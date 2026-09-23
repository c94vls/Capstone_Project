"""
support_assistant/ingest.py
Corpus ingestion and local vector database indexing using ChromaDB and all-MiniLM-L6-v2.
Runs 100% locally with zero external API dependencies.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

# -------------------------------------------------------------------------
# Exact Document Corpus Definitions
# -------------------------------------------------------------------------
DOCS_CORPUS = {
    "doc_01": (
        "Delivery Policy: Zepto delivers grocery and household essentials to serviceable "
        "pin codes within 10 to 30 minutes of order confirmation, depending on the customer's "
        "delivery zone and current order volume. Standard delivery is free on orders over INR 149; "
        "orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which "
        "reserves the next available rider slot, is available at checkout for an additional INR 15. "
        "Zepto does not currently deliver to addresses outside its listed serviceable pin codes."
    ),
    "doc_02": (
        "Returns & Refunds: Grocery and perishable items may be reported for a return within 24 hours "
        "of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned "
        "within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to "
        "the original payment method within 3–5 business days, or instantly to the Zepto wallet if "
        "the customer opts for wallet credit. Personal care items that have been opened are "
        "non-returnable except in the case of a manufacturing defect. Return pickup, where required, "
        "is arranged free of cost by Zepto."
    ),
    "doc_03": (
        "Membership Tiers: Zepto offers three account tiers: Basic (free, default tier, standard "
        "delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders "
        "and 5% off select categories), and Zepto Pass+ (INR 99 per month, free priority delivery, "
        "10% off select categories, and early access to limited-time deals 24 hours before they go "
        "live to Basic and Pass members). Membership can be cancelled at any time from account settings; "
        "cancelling stops the next billing cycle but does not refund the current membership period."
    ),
    "doc_04": (
        "Order Tracking: Every Zepto order shows a live rider-tracking map from the moment it is "
        "packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time "
        "updates automatically as the rider moves. If an order's status shows no movement for more "
        "than 20 minutes past its original estimated delivery time, customers should contact support "
        "directly rather than continue waiting, since this indicates a likely delivery issue."
    ),
    "doc_05": (
        "Order Cancellation Policy: Orders can be cancelled free of cost any time before the order "
        "status changes to 'Packed', typically within the first 2 minutes of placing the order. "
        "Once an order has been packed, it can no longer be cancelled through the app, since the "
        "rider is dispatched immediately after packing given Zepto's quick-delivery model. If a "
        "packed order cannot be delivered due to a Zepto-side issue (for example, rider unavailability), "
        "the order is auto-cancelled and fully refunded without any cancellation fee."
    ),
    "doc_06": (
        "Damaged or Missing Items: If an order arrives with damaged, spoiled, or missing items, "
        "customers must report it within 24 hours of delivery through the 'Report an Issue' button "
        "on the order page. Zepto ships a free replacement or issues a full refund for damaged, "
        "spoiled, or missing items without requiring the customer to return the original item, "
        "unless the order value exceeds INR 1000, in which case a photo of the issue must be "
        "submitted through the report form before a replacement or refund is processed."
    ),
    "doc_07": (
        "Gift Cards: Zepto gift cards are available in fixed denominations of INR 100, INR 250, "
        "INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase. Gift "
        "cards are valid for 1 year from the date of issue and carry no maintenance fees. Gift card "
        "balance can be combined with one other payment method at checkout but cannot be combined "
        "with another gift card in the same transaction. Gift card balance cannot be redeemed for "
        "cash except where required by law."
    ),
    "doc_08": (
        "Customer Support Hours: Zepto customer support is available via in-app chat 24 hours a day, "
        "7 days a week, given the time-sensitive nature of quick commerce deliveries. Average in-app "
        "chat response time is under 2 minutes. Email support is also available for non-urgent queries "
        "and is answered within 24 hours on business days. Phone support is not offered."
    )
}

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")

def build_vector_store():
    # 1. Write document corpus to docs/ directory
    os.makedirs(DOCS_DIR, exist_ok=True)
    for doc_id, text in DOCS_CORPUS.items():
        file_path = os.path.join(DOCS_DIR, f"{doc_id}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    print(f"[Ingestion] Saved {len(DOCS_CORPUS)} documents into '{DOCS_DIR}'.")

    # 2. Initialize ChromaDB client and local sentence-transformers embedding function
    client = chromadb.PersistentClient(path=DB_DIR)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # 3. Create or replace collection
    collection_name = "zepto_policies"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Insert chunks into ChromaDB
    ids = list(DOCS_CORPUS.keys())
    documents = [DOCS_CORPUS[k] for k in ids]
    metadatas = [{"source": f"{k}.txt"} for k in ids]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"[Ingestion] Embedded and indexed {len(ids)} document chunks in ChromaDB collection '{collection_name}'.")

if __name__ == "__main__":
    build_vector_store()