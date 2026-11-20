# InsightForge AI — Enterprise Business Intelligence & RAG Assistant

**InsightForge AI** is an end-to-end Business Intelligence (BI), Hybrid Retrieval-Augmented Generation (RAG), and LLMOps Evaluation suite built with **Streamlit, Python, Pandas, LangChain, FAISS, and Plotly**. It unifies structured sales analytics, dynamic user-uploaded PDF intelligence, conversational memory, natural-language data filtering, and a 4-pillar LLM quality & fairness judge in one workspace.

---

## Key Features

### 1. Interactive BI Dashboard (`Tab 1`)
- **Executive KPI Cards:** Real-time calculation of **Total Revenue**, **Average Order Value**, **Median Order Value**, and **Average Customer Satisfaction**.
- **Multi-Dimensional Filtering:** Slice sales data dynamically by **Age Group, Gender, Region, Product**, and **Keyword Search**, with one-click filter reset.
- **Visual Analytics:** Interactive Plotly charts covering monthly/yearly sales trends, growth rates, demographic breakdowns, and cross-metric correlations.

### 2. Personalized BI Assistant & Dynamic Multi-PDF RAG (`Tab 2`)
- **Bring Your Own PDFs (Live Upload + Preloaded Knowledge Base):** Drag and drop any new PDF files (up to 200 MB per file) directly in the UI to index and query them immediately alongside preloaded business reports in `Data/` and `PDF Folder/`.
- **Knowledge Source Filtering:** Scope answers to **All Sources**, specific uploaded PDFs, structured CSV sales records, or Wikipedia fallback.
- **Hybrid Retrieval & Sequential Chains:** Combines exact **Pandas statistical computation** on structured CSV data with **FAISS semantic vector search** over unstructured PDFs.
- **Live Token Streaming & Citations:** Streams responses in real time with source attributions so every insight is traceable.
- **Fairness & Safety Guardrails + Session Memory:** Applies fairness/safety checks on outputs and remembers user-specific context across turns using full-name session identification (`chat_memory.json`).
- **Tableau Bridge:** Connects insights directly with filtered BI views from the analytics tabs.

### 3. Tableau Crosstab & Natural-Language Condition Studio (`Tab 3`)
- **Ask Tableau AI (Natural-Language Auto-Filter):** Converts plain-English requests (e.g., *"Show unhappy Gen Z customers in the West who spent over 500"*) into structured filter conditions automatically.
- **Custom Condition Filter Builder:** Build multi-column rules using operators (`>`, `<`, `==`, `!=`, `between`, `contains`) across numeric and categorical fields.
- **Dynamic Crosstab Studio:** Generate multi-dimensional pivot/crosstab summaries for deeper cohort and regional analysis.

### 4. RAG & LLM Evaluation Suite (`Tab 4`)
- **4-Pillar Quality & Fairness Judge (1–5 Scale):** Scores every evaluated response across **Correctness**, **Faithfulness**, **Completeness**, and **Fairness**.
- **Live Auto-Evaluation Log:** Automatically evaluates and logs every question asked in **Tab 2 (Ask InsightForge AI)** in real time.
- **Standalone Automated Benchmark Suite:** Runs a 3-case standardized benchmark stress test without polluting personal chat history.
- **Custom Question Stress-Tester:** Test any custom prompt—**with or without a Ground Truth reference answer**—to inspect both the generated RAG output and its 4-pillar quality breakdown.

### 5. Dual-Mode Resilience (Active LLM + Zero-Crash Local Mode)
- **Active LLM Mode:** Supports OpenAI (`gpt-4o-mini`) or any OpenAI-compatible provider via custom **Base URL**, **Model Name**, and **API Key** in the sidebar or `.env`.
- **Zero-Crash Local Fallback Mode:** Runs completely offline/without an API key using local Pandas analytics, FAISS vector retrieval, Wikipedia lookup, and deterministic heuristic evaluation—ensuring every tab works reliably without crashes.

---

## Project Architecture

```text
InsightForge-AI/
├── main.py                  # Primary 4-tab Streamlit application entry point
├── app.py                   # Companion/alternate application entry module
├── tableau.py               # Crosstab & conditional filtering helpers
├── requirements.txt         # Python package dependencies
├── sales_data.csv           # Core retail/business sales dataset
├── Data/                    # Preloaded reference datasets and business PDFs
├── PDF Folder/              # Additional knowledge-base PDF documents
└── src/
    ├── config.py            # Environment variables, model defaults, and path settings
    ├── data_engine.py       # Pandas data loading, KPI computation, and multi-rule filtering
    ├── rag_engine.py        # Dynamic PDF ingestion, FAISS indexing, hybrid retrieval, streaming & memory
    ├── visuals.py           # Plotly chart builders for trends, demographics, and correlations
    └── evaluator.py         # 4-pillar LLM-as-a-Judge & local heuristic fallback evaluator
