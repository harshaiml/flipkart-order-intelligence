# Flipkart Order Intelligence & Support Assistant

An end-to-end system built for Flipkart's catalog and support teams, combining
three connected components into one working demo: a return-risk scoring
model, a product-image categoriser, and a LangGraph support agent that calls
both trained models as tools while answering policy questions from a small
retrieval-augmented knowledge base.

**Author:** Harsh Vardhan Kushwaha ([@harshaiml](https://github.com/harshaiml))

---

## Project Structure

```
flipkart-order-intelligence/
├── generate_orders.py              # Part 1: seeded synthetic order dataset generator
├── orders_dataset.csv              # Part 1: generated dataset (6,000 rows)
├── requirements.txt                # All Python dependencies
│
├── src/
│   ├── part1_return_risk/          # Return-risk scoring pipeline
│   │   ├── 01_verify_data.py
│   │   ├── 02_preprocess_and_baseline.py
│   │   ├── 03_logistic_regression.py
│   │   ├── 04_random_forest.py
│   │   ├── 05_feature_importance.py
│   │   ├── 06_subgroup_analysis.py
│   │   └── 07_save_artifact.py
│   │
│   ├── part2_image_classifier/     # Product image categoriser (transfer learning)
│   │   ├── train_classifier_step1_extract.py
│   │   ├── train_classifier_step2_train.py
│   │   ├── export_samples.py
│   │   └── common.py               # shared loading logic, used by Part 3's tool
│   │
│   └── part3_support_agent/        # LangGraph support agent
│       ├── knowledge_base.py       # 13 policy documents, chunked + retrieval answer key
│       ├── retrieval.py            # sentence-transformer embeddings + FAISS index
│       ├── tools.py                # check_return_risk, classify_product_image
│       ├── agent_graph.py          # the LangGraph graph itself
│       └── run_tests.py            # generates transcripts/ + retrieval evaluation
│
├── models/
│   ├── return_risk_model.pkl       # Part 1's final tuned Random Forest pipeline
│   ├── t_star_rf.txt               # Part 1's F1-maximising threshold on the RF
│   └── product_classifier_head.pt  # Part 2's trained classifier head
│
├── data/
│   └── sample_images/              # 10 real Fashion-MNIST test images, exported as .png
│
└── transcripts/                    # Part 3's 9 recorded test conversations + retrieval eval
```

---

## Part 1 — Return-Risk Scoring Pipeline

**Goal:** flag orders that are statistically likely to be returned, before the return happens.

### How to run

```bash
pip install -r requirements.txt
python generate_orders.py                                    # generates orders_dataset.csv
python src/part1_return_risk/01_verify_data.py                # data verification
python src/part1_return_risk/02_preprocess_and_baseline.py    # preprocessing + baseline
python src/part1_return_risk/03_logistic_regression.py        # Logistic Regression + threshold sweep
python src/part1_return_risk/04_random_forest.py              # Random Forest + GridSearchCV
python src/part1_return_risk/05_feature_importance.py         # impurity + permutation importance
python src/part1_return_risk/06_subgroup_analysis.py          # subgroup recall/precision breakdown
python src/part1_return_risk/07_save_artifact.py              # saves models/return_risk_model.pkl
```

### Key results

- **Dataset:** 6,000 orders, 13 columns, 22.75% overall return rate.
- `rating_given` is missing on 13.05% of rows -- classified as **MAR** (missing at
  random, conditional on `payment_method`: COD orders are missing at ~22.8% vs
  ~6% for non-COD), not MCAR or MNAR.
- **Baseline (DummyClassifier):** 77% accuracy, but **F1 = 0.0000** for the
  `returned=1` class -- the "high accuracy, zero recall" trap.
- **Logistic Regression** (default threshold): ROC-AUC 0.625, F1 0.392. A
  threshold sweep raised recall by ~18 percentage points at a small precision cost.
- **Random Forest** (tuned via `GridSearchCV`, 5-fold `StratifiedKFold`, scored
  on ROC-AUC): best params `{max_depth: 6, n_estimators: 100}`, CV ROC-AUC
  0.618, held-out test ROC-AUC 0.614 (within 0.004 -- no overfitting).
- **Feature importance:** `payment_method`, `price_inr`, `customer_tenure_days`,
  and `discount_pct` are in the impurity-based top-5; permutation importance
  shows `delivery_distance_km` and `customer_tenure_days` lose most of their
  apparent importance once shuffled -- a textbook case of impurity-based
  importance overrating a noisy continuous feature.
- **Final artifact:** `models/return_risk_model.pkl` is one combined
  scikit-learn `Pipeline` (preprocessing + tuned Random Forest). Its own
  F1-maximising threshold, **t\*\_rf = 0.46**, is saved to `models/t_star_rf.txt`
  and anchors Part 3's risk buckets (Low / Medium / High).

---

## Part 2 — Product Image Categoriser (Transfer Learning)

**Goal:** classify a product photo into one of 10 apparel/footwear/accessory categories.

> **Note on environment:** training this part downloads ImageNet-pretrained
> ResNet-18 weights, which requires normal outbound internet access. It was
> trained on **Google Colab (free T4 GPU)** rather than a restricted sandbox.

### How to run (Google Colab recommended)

```bash
pip install -r requirements.txt
python src/part2_image_classifier/train_classifier_step1_extract.py   # downloads Fashion-MNIST + caches frozen-backbone features
python src/part2_image_classifier/train_classifier_step2_train.py     # trains the head, evaluates, saves the model
python src/part2_image_classifier/export_samples.py                   # exports 10 real test images to data/sample_images/
```

### Key results

- **Dataset:** Fashion-MNIST (60,000 train / 10,000 test), pinned to the
  canonical Zalando Research source. Stratified split: 55,000 train / 5,000
  validation / 10,000 test.
- **Approach:** ImageNet-pretrained ResNet-18 backbone, fully frozen; features
  cached once, then only a small new classifier head (512 → 128 → 10) is
  trained on the cached features -- this turns what would be an hours-long
  CPU training loop into a ~30-second run on a GPU.
- **Feature extraction alone reached 91.12% validation accuracy** -- above the
  80% bar, so no fine-tuning of backbone layers was needed.
- **Final test-set accuracy: 89.86%.**
- **Confusion patterns:** the model's biggest confusions are **Shirt ↔
  T-shirt/top** (114 cases) and **Shirt → Coat** (102 cases) -- all three are
  visually similar upper-body garments in 28x28 grayscale, which plausibly
  explains the confusion.
- **Artifact:** `models/product_classifier_head.pt` (loaded together with the
  frozen ResNet-18 backbone via `src/part2_image_classifier/common.py`).

---

## Part 3 — Flipkart Support Agent (LangGraph)

**Goal:** a single conversational agent that answers policy questions via RAG
and calls Part 1's and Part 2's real saved models as tools.

### How to run (default MOCK_LLM mode -- zero API keys, zero network calls for generation)

```bash
pip install -r requirements.txt
python src/part3_support_agent/run_tests.py
```

This regenerates all 9 transcripts in `transcripts/` and reruns the retrieval
evaluation. An optional live-LLM extension exists (`USE_LIVE_LLM=1`
environment variable) but is never required -- every acceptance criterion
passes in the default `MOCK_LLM` mode.

### Architecture

- **Knowledge base:** 13 short Flipkart-style policy documents (return
  windows by category, COD/prepaid refund timelines, delivery SLAs,
  reverse-pickup eligibility, damaged/wrong-item policies, cancellations),
  chunked sentence-wise into 37 chunks.
- **Retrieval:** each chunk embedded with `all-MiniLM-L6-v2`
  (sentence-transformers, free/local) and indexed with a FAISS
  `IndexFlatIP` (cosine similarity via normalized inner product).
- **Graph (LangGraph, 4 nodes + 1 conditional edge):**
  `intent` → (conditionally) → `rag_retrieval` **or** `tool_calling` → `response_generation`
  - **intent node:** classifies the turn as `policy_kb`, `return_risk_tool`,
    or `image_classifier_tool`; also detects prompt-injection attempts and
    follow-up references to a previously-discussed order (short-term
    conversational state).
  - **tool_calling node:** calls the *real* `check_return_risk` (loads
    `models/return_risk_model.pkl`, buckets risk relative to `t*_rf`) or
    `classify_product_image` (loads `models/product_classifier_head.pt` +
    frozen ResNet-18, run against real `.png` files in `data/sample_images/`).
  - **response_generation node:** composes the final structured JSON answer
    deterministically in `MOCK_LLM` mode (`{"answer", "source", "confidence"}`).
- **Guardrails:**
  - *Input-side:* regex-based prompt-injection filtering (blocks phrases like
    "ignore all instructions", "pretend you are...", "reveal your system prompt").
  - *Output-side:* a groundedness check refuses to answer a policy question if
    the top retrieved chunk's similarity is below 0.35, rather than letting
    the mock generator fabricate a policy.
- **System prompt** is engineered against the 4S principles (Specific, Short,
  Surround, Single) with two few-shot intent-classification examples --
  see `SYSTEM_PROMPT` in `agent_graph.py`.

### Test transcripts (`transcripts/`)

| File | Covers |
|---|---|
| `01_policy_question_apparel.json` | Policy question answered via RAG |
| `02_policy_question_cod_refund.json` | Policy question answered via RAG |
| `03_return_risk_question.json` | Calls `check_return_risk` with realistic order features |
| `04_image_classification_question.json` | Calls `classify_product_image` on a real `.png` |
| `05_multiturn_turn1.json` / `06_multiturn_turn2_followup.json` | Multi-turn state carried across turns ("that order") |
| `07_fresh_conversation_no_state.json` | Same follow-up phrase in a **fresh** conversation -- state correctly absent |
| `08_prompt_injection_attempt.json` | Injection attempt visibly blocked by the input guardrail |
| `09_ungrounded_question_refusal.json` | Off-topic question correctly refused by the groundedness check |
| `10_retrieval_evaluation.json` | Precision@3 / Recall@3 across 7 test queries |

### Retrieval evaluation results

Average **Precision@3 = 0.31**, average **Recall@3 = 0.71** across 7 test
queries (document-level, chunks deduplicated to their parent document before
scoring). Two queries ("delivery time in a small town", "reverse pickup")
retrieved semantically related but not exactly-matching documents -- reported
honestly rather than adjusted, per-query arithmetic is in
`transcripts/10_retrieval_evaluation.json`.

---

## Environment Notes

- Parts 1 and 3's retrieval/agent logic were developed and tested locally.
- Part 2's training and Part 3's `run_tests.py` (which needs the
  `all-MiniLM-L6-v2` embedding model and the ResNet-18 ImageNet backbone) were
  run on **Google Colab** (free T4 GPU), since they require downloading
  pretrained weights from the open internet.
- All acceptance criteria are satisfied in the default configuration with
  **zero paid services, zero API keys, and zero account-gated tools** --
  scikit-learn, PyTorch, sentence-transformers, and FAISS are all free and local.
