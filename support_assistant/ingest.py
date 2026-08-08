# ---------------------------------------------------------
# Zepto Support Assistant - Document Ingestion
# ---------------------------------------------------------

import os
import glob

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. Find the docs folder
# ---------------------------------------------------------

DOCS_FOLDER = "docs"


# ---------------------------------------------------------
# 2. Load the embedding model
# ---------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")


# ---------------------------------------------------------
# 3. Create ChromaDB client
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)


# ---------------------------------------------------------
# 4. Create / get collection
# ---------------------------------------------------------

collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)


# ---------------------------------------------------------
# 5. Read all documents
# ---------------------------------------------------------

files = sorted(
    glob.glob(
        os.path.join(DOCS_FOLDER, "doc_*.txt")
    )
)

print(f"Found {len(files)} documents.")


# ---------------------------------------------------------
# 6. Store document information
# ---------------------------------------------------------

documents = []
document_ids = []


for file_path in files:

    # Get file name
    file_name = os.path.basename(file_path)

    # Read file
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read().strip()

    # One document = one chunk
    documents.append(text)

    # Use file name as document ID
    document_ids.append(file_name)


# ---------------------------------------------------------
# 7. Generate embeddings
# ---------------------------------------------------------

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    normalize_embeddings=True
).tolist()

print("Embeddings generated successfully.")


# ---------------------------------------------------------
# 8. Add documents to ChromaDB
# ---------------------------------------------------------

collection.upsert(
    ids=document_ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents stored in ChromaDB successfully.")


# ---------------------------------------------------------
# 9. Check number of stored documents
# ---------------------------------------------------------

count = collection.count()

print(f"Total documents in ChromaDB: {count}")


# ---------------------------------------------------------
# 10. Test retrieval
# ---------------------------------------------------------

test_query = "What is the delivery fee?"

print("\nTesting retrieval...")
print("Query:", test_query)


query_embedding = model.encode(
    [test_query],
    normalize_embeddings=True
).tolist()


results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)


# ---------------------------------------------------------
# 11. Print retrieved documents
# ---------------------------------------------------------

print("\nTop retrieved documents:")

for i, doc_id in enumerate(results["ids"][0]):

    print(
        f"\n{i + 1}. {doc_id}"
    )

    print(
        results["documents"][0][i][:200]
    )


print("\nIngestion completed successfully.")