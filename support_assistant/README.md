# Zepto Support Assistant

A small GenAI-style support assistant for Zepto policies using document embeddings, ChromaDB, LangGraph, Pydantic structured output, and FastAPI.

The required graded path uses deterministic offline mock logic through the `MOCK_LLM` environment variable. No LLM API key is required.

---

## 1. Project Structure

```text
support_assistant/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── ingest.py
├── main.py
├── Dockerfile
└── README.md