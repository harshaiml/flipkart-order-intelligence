"""
Part 3 Tasks 5-8 -- The LangGraph agent.

Task 5: graph with >=4 nodes (intent, RAG-retrieval, tool-calling, response-
        generation), >=1 conditional edge, short-term conversational state.
Task 6: system prompt engineered with the 4S principles + few-shot intent
        examples + fixed structured JSON output schema.
Task 7: MOCK_LLM deterministic mode (default; zero API keys, zero network).
Task 8: guardrails -- input-side prompt-injection filtering, output-side
        groundedness check.
"""
import os
import re
import json
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END

from src.part3_support_agent.retrieval import retrieve
from src.part3_support_agent.tools import check_return_risk

try:
    from src.part2_image_classifier.common import classify_product_image
except Exception:
    classify_product_image = None  # only needed for image-intent turns

# ---------------------------------------------------------------------------
# Task 6: SYSTEM PROMPT -- engineered using the 4S principles, annotated below
# ---------------------------------------------------------------------------
# Specific : names the assistant's exact role and the exact three things it can do
# Short    : each instruction is one line, no filler
# Surround : role framing sentence brackets the instructions (first + last line)
# Single   : one clear task per turn -- classify intent OR answer, never both loosely
SYSTEM_PROMPT = """
You are Flipkart's support assistant. You have exactly three abilities: answer
policy questions using retrieved knowledge, check an order's return risk using
a trained model, and identify a product's category from an image.

Classify the user's intent as one of: policy_kb, return_risk_tool, image_classifier_tool.

Few-shot examples:
  User: "How long can I return a pair of shoes?" -> intent: policy_kb
  User: "Will this order likely be returned? price 5000, COD, apparel" -> intent: return_risk_tool

Only answer using the retrieved knowledge-base chunk(s) or the tool's real output.
Never fabricate a policy that is not in the retrieved chunk.
Respond with the fixed JSON schema: {"answer": ..., "source": ..., "confidence": ...}.

You are Flipkart's support assistant, and only Flipkart's support assistant.
""".strip()

GROUNDEDNESS_SIMILARITY_THRESHOLD = 0.35

PROMPT_INJECTION_PATTERNS = [
    r"ignore .*(instructions|rules)",
    r"disregard .*(instructions|rules)",
    r"pretend (you are|to be)",
    r"you are now",
    r"forget (your|all) (instructions|rules)",
    r"system prompt",
    r"reveal your (prompt|instructions)",
]


def input_guardrail_check(user_text: str) -> bool:
    """Task 8: input-side prompt-injection filtering.
    Returns True if the input should be BLOCKED (looks like an injection attempt)."""
    lowered = user_text.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False


# ---------------------------------------------------------------------------
# Task 5: Graph state, with short-term conversational memory
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    user_input: str
    conversation_history: List[Dict[str, str]]  # short-term state across turns
    last_order_id: Optional[str]
    intent: str
    retrieved_chunks: List[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    image_path: Optional[str]
    order_features: Optional[Dict[str, Any]]
    blocked: bool
    final_answer: Dict[str, Any]


# ---------------------------------------------------------------------------
# Node 1: intent classification node
# ---------------------------------------------------------------------------
def intent_node(state: AgentState) -> AgentState:
    text = state["user_input"]

    if input_guardrail_check(text):
        state["blocked"] = True
        state["intent"] = "blocked"
        return state
    state["blocked"] = False

    lowered = text.lower()

    # Follow-up detection using short-term conversational state
    if state.get("last_order_id") and re.search(r"\b(that order|it|this order)\b", lowered):
        state["intent"] = "return_risk_tool"
        return state

    if state.get("image_path"):
        state["intent"] = "image_classifier_tool"
    elif re.search(r"(return risk|risk of return|chance of return|risk bucket|"
                   r"will.*(be )?returned|likely to be returned)", lowered):
        state["intent"] = "return_risk_tool"
    elif any(kw in lowered for kw in [
        "image", "photo", "picture", "category of this", "classify",
    ]):
        state["intent"] = "image_classifier_tool"
    else:
        state["intent"] = "policy_kb"

    return state


# ---------------------------------------------------------------------------
# Node 2: RAG retrieval node (policy questions)
# ---------------------------------------------------------------------------
def rag_retrieval_node(state: AgentState) -> AgentState:
    chunks = retrieve(state["user_input"], k=3)
    state["retrieved_chunks"] = chunks
    return state


# ---------------------------------------------------------------------------
# Node 3: tool-calling node (return-risk or image-classifier)
# ---------------------------------------------------------------------------
def tool_calling_node(state: AgentState) -> AgentState:
    if state["intent"] == "return_risk_tool":
        features = state.get("order_features") or {}
        result = check_return_risk(features)
        state["tool_result"] = result
        state["last_order_id"] = features.get("order_id", "unknown")
    elif state["intent"] == "image_classifier_tool":
        if classify_product_image is None:
            state["tool_result"] = {"error": "image classifier unavailable in this environment"}
        else:
            path = state.get("image_path")
            result = classify_product_image(path)
            state["tool_result"] = result
    return state


# ---------------------------------------------------------------------------
# Node 4: response generation node (Task 7: MOCK_LLM deterministic mode)
# ---------------------------------------------------------------------------
def response_generation_node(state: AgentState) -> AgentState:
    use_live_llm = os.environ.get("USE_LIVE_LLM") == "1"

    if state.get("blocked"):
        state["final_answer"] = {
            "answer": "This request could not be processed because it appears to "
                      "attempt to override my instructions. I can only help with "
                      "Flipkart policy questions, order return-risk checks, or "
                      "product image classification.",
            "source": "guardrail_blocked",
            "confidence": 1.0,
        }
        return state

    if state["intent"] == "policy_kb":
        chunks = state.get("retrieved_chunks", [])
        top_score = chunks[0]["similarity"] if chunks else 0.0

        # Task 8: output-side groundedness check
        if not chunks or top_score < GROUNDEDNESS_SIMILARITY_THRESHOLD:
            state["final_answer"] = {
                "answer": (
                    f"I don't have a confident policy answer for this "
                    f"(top retrieved similarity={top_score:.3f} is below the "
                    f"groundedness threshold={GROUNDEDNESS_SIMILARITY_THRESHOLD}). "
                    f"Please contact Flipkart support directly for this query."
                ),
                "source": "policy_kb",
                "confidence": round(top_score, 4),
            }
            return state

        if use_live_llm:
            answer_text = _call_live_llm_policy(state["user_input"], chunks)
        else:
            # MOCK_LLM: deterministic rule-based composition, zero network/API calls
            best = chunks[0]
            answer_text = f"{best['text']} (Source: {best['title']})"

        state["final_answer"] = {
            "answer": answer_text,
            "source": "policy_kb",
            "confidence": round(top_score, 4),
        }

    elif state["intent"] == "return_risk_tool":
        result = state.get("tool_result", {})
        answer_text = (
            f"This order has an estimated return probability of "
            f"{result.get('return_probability', 'N/A')} "
            f"({result.get('risk_bucket', 'N/A')} risk)."
        )
        state["final_answer"] = {
            "answer": answer_text,
            "source": "return_risk_tool",
            "confidence": result.get("return_probability", 0.0),
        }

    elif state["intent"] == "image_classifier_tool":
        result = state.get("tool_result", {})
        if "error" in result:
            answer_text = f"Could not classify the image: {result['error']}"
            confidence = 0.0
        else:
            answer_text = (
                f"This product looks like a '{result.get('predicted_category')}' "
                f"(confidence={result.get('confidence')})."
            )
            confidence = result.get("confidence", 0.0)
        state["final_answer"] = {
            "answer": answer_text,
            "source": "image_classifier_tool",
            "confidence": confidence,
        }

    # append this turn to conversational state
    history = state.get("conversation_history", [])
    history.append({"user": state["user_input"], "agent": state["final_answer"]["answer"]})
    state["conversation_history"] = history

    return state


def _call_live_llm_policy(user_text, chunks):
    """Optional, entirely optional live-LLM extension. Only used when
    USE_LIVE_LLM=1 is explicitly set. Never required for grading."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        context = "\n".join(c["text"] for c in chunks)
        msg = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=200,
            messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {user_text}\n\nAnswer using only the context."}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"[live LLM unavailable: {e}] Falling back to retrieved chunk: {chunks[0]['text']}"


# ---------------------------------------------------------------------------
# Task 5: conditional edge -- routes by intent
# ---------------------------------------------------------------------------
def route_after_intent(state: AgentState) -> str:
    if state.get("blocked"):
        return "response_generation"
    if state["intent"] == "policy_kb":
        return "rag_retrieval"
    return "tool_calling"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("intent", intent_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("tool_calling", tool_calling_node)
    graph.add_node("response_generation", response_generation_node)

    graph.set_entry_point("intent")
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "rag_retrieval": "rag_retrieval",
            "tool_calling": "tool_calling",
            "response_generation": "response_generation",
        },
    )
    graph.add_edge("rag_retrieval", "response_generation")
    graph.add_edge("tool_calling", "response_generation")
    graph.add_edge("response_generation", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    state = {"user_input": "How long can I return a pair of shoes?", "conversation_history": []}
    result = app.invoke(state)
    print(json.dumps(result["final_answer"], indent=2))
