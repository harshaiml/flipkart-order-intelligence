"""
Part 3 Task 1 -- Policy knowledge base.
At least 12 short (2-4 sentence) Flipkart-style policy documents, covering:
return windows by category, COD refund timelines, delivery SLAs, and
reverse-pickup eligibility. Each document is chunked sentence-wise.

Also defines a query/relevant-document answer key (Task 1 continued) used
later by Task 10's retrieval evaluation (Precision@3 / Recall@3).
"""

POLICY_DOCUMENTS = [
    {
        "doc_id": "return_apparel_footwear",
        "title": "Return Window -- Apparel & Footwear",
        "text": (
            "Apparel and footwear items can be returned within 14 days of delivery, "
            "provided tags are intact and the item is unworn. Innerwear, socks, and "
            "customized apparel are non-returnable for hygiene reasons. Footwear must "
            "be returned in its original box to be eligible for a refund."
        ),
    },
    {
        "doc_id": "return_electronics",
        "title": "Return Window -- Electronics",
        "text": (
            "Electronics such as phones, laptops, and headphones have a 7-day return "
            "window from the date of delivery. The item must include all original "
            "accessories, manuals, and packaging. Physical damage or a missing IMEI "
            "box voids eligibility for return."
        ),
    },
    {
        "doc_id": "return_home",
        "title": "Return Window -- Home & Furniture",
        "text": (
            "Home and furniture items can be returned within 10 days of delivery if "
            "unused and in original packaging. Large furniture returns require a "
            "reverse pickup slot to be scheduled within this window. Assembled "
            "furniture may incur a re-packaging fee upon return."
        ),
    },
    {
        "doc_id": "return_beauty",
        "title": "Return Window -- Beauty Products",
        "text": (
            "Beauty and personal care products are returnable within 7 days only if "
            "the factory seal is unbroken. Opened cosmetics, perfumes, and skincare "
            "items cannot be returned due to safety regulations. Damaged-on-arrival "
            "beauty items are eligible for replacement, not refund."
        ),
    },
    {
        "doc_id": "cod_refund_timeline",
        "title": "COD Refund Timeline",
        "text": (
            "For Cash on Delivery orders, refunds are issued to the customer's bank "
            "account within 7-10 business days after the returned item passes "
            "quality inspection. COD refunds cannot be credited back to a card since "
            "no card was used at checkout. Customers must provide valid bank details "
            "in the return request."
        ),
    },
    {
        "doc_id": "prepaid_refund_timeline",
        "title": "Prepaid Refund Timeline",
        "text": (
            "Prepaid orders (Prepaid Card, Prepaid UPI, Wallet) are refunded to the "
            "original payment method within 3-5 business days after the return is "
            "approved. UPI refunds are typically the fastest, often completing within "
            "48 hours of approval."
        ),
    },
    {
        "doc_id": "delivery_sla_metro",
        "title": "Delivery SLA -- Metro Cities",
        "text": (
            "Orders shipped to metro cities are delivered within 2-4 business days "
            "under standard shipping. Express delivery, where available, targets "
            "next-day delivery for an additional fee. Delays beyond 4 days in metro "
            "areas are eligible for a delivery-delay support ticket."
        ),
    },
    {
        "doc_id": "delivery_sla_nonmetro",
        "title": "Delivery SLA -- Non-Metro & Rural Areas",
        "text": (
            "Non-metro and rural pin codes typically receive orders within 5-8 "
            "business days. Remote pin codes beyond standard courier network "
            "coverage may take up to 12 business days. Customers in these areas do "
            "not have access to express delivery options."
        ),
    },
    {
        "doc_id": "reverse_pickup_eligibility",
        "title": "Reverse Pickup Eligibility",
        "text": (
            "Reverse pickup is offered automatically for returns in pin codes "
            "covered by Flipkart's courier network. In pin codes without reverse "
            "pickup coverage, customers must self-ship the item and upload the "
            "courier receipt for refund processing. Reverse pickup attempts are "
            "made up to twice before the return request is cancelled."
        ),
    },
    {
        "doc_id": "damaged_item_policy",
        "title": "Damaged or Defective Item Policy",
        "text": (
            "Items reported as damaged or defective within 48 hours of delivery are "
            "eligible for a free replacement or full refund, regardless of the "
            "category's normal return window. Customers must upload photos of the "
            "damage at the time of raising the request. No return shipping charge "
            "applies for damaged-item claims."
        ),
    },
    {
        "doc_id": "wrong_item_policy",
        "title": "Wrong Item Delivered Policy",
        "text": (
            "If a customer receives an item different from what was ordered, a free "
            "replacement or refund is initiated immediately upon verification. This "
            "claim must be raised within 3 days of delivery. Reverse pickup for "
            "wrong-item claims is prioritized over standard return pickups."
        ),
    },
    {
        "doc_id": "cancellation_policy",
        "title": "Order Cancellation Policy",
        "text": (
            "Orders can be cancelled free of charge before they are shipped. Once an "
            "order has shipped, it can no longer be cancelled and must instead be "
            "returned after delivery following the standard return process. "
            "Cancellation refunds for prepaid orders follow the same timeline as "
            "return refunds."
        ),
    },
    {
        "doc_id": "return_shipping_fee",
        "title": "Return Shipping Fee Policy",
        "text": (
            "Standard returns due to change of mind are free of shipping charges "
            "for Flipkart Plus members, and incur a nominal fee for other "
            "customers in categories where reverse pickup is offered. Damaged, "
            "defective, or wrong-item returns never incur a shipping fee "
            "regardless of membership status."
        ),
    },
]


def chunk_document(doc):
    """Chunk a document sentence-wise (one strategy for production RAG).
    Splits on '. ' while keeping the trailing period, producing one chunk per
    sentence -- multi-sentence documents each yield more than one chunk."""
    text = doc["text"].strip()
    raw_sentences = [s.strip() for s in text.split(". ") if s.strip()]
    chunks = []
    for i, sent in enumerate(raw_sentences):
        if not sent.endswith("."):
            sent = sent + "."
        chunks.append({
            "chunk_id": f"{doc['doc_id']}__chunk{i}",
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "text": sent,
        })
    return chunks


def build_all_chunks():
    all_chunks = []
    for doc in POLICY_DOCUMENTS:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


# ---- Task 1 (continued): query / relevant-document answer key ----
# For at least 5 realistic test queries, the document(s) considered "relevant".
# This becomes the retrieval-evaluation answer key for Task 10.
RETRIEVAL_ANSWER_KEY = [
    {
        "query": "How long do I have to return a t-shirt I bought?",
        "relevant_doc_ids": ["return_apparel_footwear"],
    },
    {
        "query": "When will I get my refund for a COD order?",
        "relevant_doc_ids": ["cod_refund_timeline"],
    },
    {
        "query": "How many days does delivery take in a small town?",
        "relevant_doc_ids": ["delivery_sla_nonmetro"],
    },
    {
        "query": "Will someone come pick up my return, or do I have to ship it myself?",
        "relevant_doc_ids": ["reverse_pickup_eligibility"],
    },
    {
        "query": "The phone I received is damaged, what happens now?",
        "relevant_doc_ids": ["damaged_item_policy"],
    },
    {
        "query": "Can I return an opened perfume bottle?",
        "relevant_doc_ids": ["return_beauty"],
    },
    {
        "query": "I got the wrong product in my package, what do I do?",
        "relevant_doc_ids": ["wrong_item_policy"],
    },
]

if __name__ == "__main__":
    chunks = build_all_chunks()
    print(f"Total documents: {len(POLICY_DOCUMENTS)}")
    print(f"Total chunks: {len(chunks)}")
    for c in chunks[:5]:
        print(f"  [{c['chunk_id']}] {c['text']}")
    print(f"\nRetrieval answer key has {len(RETRIEVAL_ANSWER_KEY)} queries.")
