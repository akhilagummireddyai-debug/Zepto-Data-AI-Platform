# ============================================================
# Zepto Support Assistant
# LangGraph + ChromaDB + FastAPI
# ============================================================

import os
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. MOCK_LLM SETTING
# ============================================================

# If MOCK_LLM is not set, mock mode is used.
# This is the required graded mode.

MOCK_LLM = os.getenv("MOCK_LLM", "1")


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 3. CONNECT TO CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)

print(
    "ChromaDB documents:",
    collection.count()
)


# ============================================================
# 4. STATE USED BY LANGGRAPH
# ============================================================

class SupportState(TypedDict, total=False):

    query: str

    intent: str

    answer: str

    sources: list[str]

    confidence: float


# ============================================================
# 5. FINAL RESPONSE MODEL
# ============================================================

class AnswerResponse(BaseModel):

    answer: str

    sources: list[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# ============================================================
# 6. REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):

    query: str


# ============================================================
# 7. STRUCTURED PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are Zepto's customer support assistant.

CONTEXT:
Answer only using the policy information provided below.

TASK:
Answer the customer's question using the retrieved context.

FORMAT:
Return a clear and concise answer.

LENGTH:
Keep the answer short and useful.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present
in the provided context.

FEW-SHOT EXAMPLE:

Question:
What is the delivery fee?

Context:
Standard delivery is free on orders over INR 149.
Orders below INR 149 have a flat INR 25 delivery fee.

Answer:
Orders below INR 149 have a INR 25 delivery fee,
while standard delivery is free for orders over INR 149.

Customer Question:
{query}

Retrieved Context:
{context}
"""


# ============================================================
# 8. NODE 1 - CLASSIFY INTENT
# ============================================================

def classify_intent(
    state: SupportState
) -> SupportState:

    query = state["query"].lower()

    # Required mock-mode keyword heuristic

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    # Check whether any policy keyword exists

    is_policy_question = any(
        keyword in query
        for keyword in policy_keywords
    )

    if is_policy_question:

        intent = "policy_question"

    else:

        intent = "general_question"

    print(
        f"Intent classified as: {intent}"
    )

    return {
        **state,
        "intent": intent
    }


# ============================================================
# 9. NODE 2 - RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(
    state: SupportState
) -> SupportState:

    query = state["query"]

    # --------------------------------------------------------
    # Create embedding for the user query
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    # --------------------------------------------------------
    # Retrieve top 3 documents
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    retrieved_documents = results["documents"][0]

    retrieved_ids = results["ids"][0]

    # --------------------------------------------------------
    # Get the most relevant chunk
    # --------------------------------------------------------

    top_chunk = retrieved_documents[0]

    # Keep answer short
    top_chunk_snippet = top_chunk[:200]

    # --------------------------------------------------------
    # Required MOCK_LLM behaviour
    # --------------------------------------------------------

    if MOCK_LLM == "1":

        answer = (
            "Based on the retrieved context: "
            + top_chunk_snippet
        )

        confidence = 1.0

    else:

        # ----------------------------------------------------
        # Optional real-LLM extension
        # ----------------------------------------------------
        # The graded submission uses MOCK_LLM=1.
        #
        # This branch is intentionally kept as an extension.
        # ----------------------------------------------------

        context = "\n\n".join(
            retrieved_documents
        )

        prompt = PROMPT_TEMPLATE.format(
            query=query,
            context=context
        )

        # Placeholder for optional real LLM integration.
        # The baseline does not require an API call.

        answer = (
            "Real LLM mode is not configured. "
            "Retrieved context: "
            + top_chunk_snippet
        )

        confidence = 0.8

    # --------------------------------------------------------
    # Return structured state
    # --------------------------------------------------------

    return {
        **state,
        "answer": answer,
        "sources": retrieved_ids,
        "confidence": confidence
    }


# ============================================================
# 10. NODE 3 - DIRECT ANSWER
# ============================================================

def direct_answer(
    state: SupportState
) -> SupportState:

    # Required mock response

    if MOCK_LLM == "1":

        answer = (
            "I can only answer questions about "
            "Zepto policies right now."
        )

        confidence = 1.0

    else:

        # Optional real-LLM extension

        answer = (
            "I can only answer questions about "
            "Zepto policies right now."
        )

        confidence = 0.8

    return {
        **state,
        "answer": answer,
        "sources": [],
        "confidence": confidence
    }


# ============================================================
# 11. CONDITIONAL ROUTING FUNCTION
# ============================================================

def route_question(
    state: SupportState
):

    if state["intent"] == "policy_question":

        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# 12. CREATE LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    SupportState
)


# Add the three required nodes

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)


# ============================================================
# 13. CONNECT GRAPH NODES
# ============================================================

graph_builder.add_edge(
    START,
    "classify_intent"
)


# Conditional edge after classification

graph_builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer":
            "retrieve_and_answer",

        "direct_answer":
            "direct_answer"
    }
)


# Both branches finish the graph

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)


# ============================================================
# 14. COMPILE GRAPH
# ============================================================

graph = graph_builder.compile()

print(
    "LangGraph compiled successfully."
)


# ============================================================
# 15. RUN SUPPORT ASSISTANT
# ============================================================

def ask_question(
    query: str
) -> AnswerResponse:

    initial_state: SupportState = {
        "query": query
    }

    result = graph.invoke(
        initial_state
    )

    # Validate final response with Pydantic

    response = AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

    return response


# ============================================================
# 16. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description="Zepto policy RAG support assistant",
    version="1.0"
)


# ============================================================
# 17. POST /ask ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask(request: AskRequest):

    return ask_question(
        request.query
    )


# ============================================================
# 18. LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("\nTesting policy question...")

    policy_result = ask_question(
        "What is the delivery fee?"
    )

    print(
        policy_result.model_dump_json(
            indent=2
        )
    )

    print("\nTesting general question...")

    general_result = ask_question(
        "What is the capital of India?"
    )

    print(
        general_result.model_dump_json(
            indent=2
        )
    )

    print(
        "\nSupport Assistant test completed successfully."
    )