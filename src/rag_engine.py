import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    LangChainDocument = None


@dataclass
class FallbackDocument:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def make_doc(text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
    meta = metadata or {}
    if LangChainDocument is not None:
        return LangChainDocument(page_content=text, metadata=meta)
    return FallbackDocument(page_content=text, metadata=meta)


try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    PyPDFLoader = None
    FAISS = None
    HuggingFaceEmbeddings = None
    RecursiveCharacterTextSplitter = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "PDF Folder")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
MEMORY_FILE = os.path.join(BASE_DIR, "chat_memory.json")


class SalesDataEngine:
    def __init__(self, csv_path: Optional[str] = None):
        candidate_paths = [
            csv_path,
            os.path.join(BASE_DIR, "sales_data.csv"),
            os.path.join(BASE_DIR, "Data", "sales_data.csv"),
            os.path.join(BASE_DIR, "Data", "sales_data (1).csv"),
        ]
        resolved_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
        self.df = pd.read_csv(resolved_path) if resolved_path else pd.DataFrame()
        self._normalize_columns()

    def _normalize_columns(self):
        if self.df.empty:
            return
        rename_map = {}
        for col in self.df.columns:
            low = col.strip().lower()
            if low in {"sales", "total_sales", "revenue", "amount"} and "Sales" not in self.df.columns:
                rename_map[col] = "Sales"
            elif low in {"region", "regions"} and "Region" not in self.df.columns:
                rename_map[col] = "Region"
            elif low in {"product", "products", "item"} and "Product" not in self.df.columns:
                rename_map[col] = "Product"
            elif low in {"gender", "customer_gender"} and "Customer_Gender" not in self.df.columns:
                rename_map[col] = "Customer_Gender"
            elif low in {"age_group", "age"} and "Age_Group" not in self.df.columns:
                rename_map[col] = "Age_Group"
            elif low in {"satisfaction", "customer_satisfaction", "rating"} and "Customer_Satisfaction" not in self.df.columns:
                rename_map[col] = "Customer_Satisfaction"
        if rename_map:
            self.df.rename(columns=rename_map, inplace=True)

        if "Sales" in self.df.columns:
            self.df["Sales"] = pd.to_numeric(self.df["Sales"], errors="coerce")
        if "Customer_Satisfaction" in self.df.columns:
            self.df["Customer_Satisfaction"] = pd.to_numeric(self.df["Customer_Satisfaction"], errors="coerce")

    def summarize_dataframe(self, df: Optional[pd.DataFrame] = None, scope_name: str = "Sales Dataset") -> str:
        target = self.df if df is None else df
        if target is None or target.empty:
            return f"Structured Data Summary ({scope_name}): No matching rows found for the requested filter."

        lines = [f"Structured Data Summary ({scope_name}): {len(target):,} matching orders"]

        if "Sales" in target.columns and target["Sales"].notna().any():
            lines.append(
                f"- Total Sales: ${target['Sales'].sum():,.2f} | "
                f"Avg Order: ${target['Sales'].mean():,.2f} | "
                f"Median: ${target['Sales'].median():,.2f}"
            )

        if "Customer_Satisfaction" in target.columns and target["Customer_Satisfaction"].notna().any():
            lines.append(f"- Avg Customer Satisfaction: {target['Customer_Satisfaction'].mean():.2f} / 5.0")

        if "Region" in target.columns and "Sales" in target.columns:
            reg_map = target.groupby("Region")["Sales"].sum().round(0).astype(int).to_dict()
            lines.append(f"- Sales by Region: {reg_map}")

        if "Product" in target.columns and "Sales" in target.columns:
            prod_map = target.groupby("Product")["Sales"].sum().round(0).astype(int).to_dict()
            lines.append(f"- Sales by Product: {prod_map}")

        if "Customer_Gender" in target.columns and "Sales" in target.columns:
            gen_stats = (
                target.groupby("Customer_Gender")["Sales"]
                .agg(count="count", sum="sum", mean="mean")
                .round({"sum": 0, "mean": 2})
            )
            gen_dict = {
                k: {"count": int(v["count"]), "sum": int(v["sum"]), "mean": float(v["mean"])}
                for k, v in gen_stats.iterrows()
            }
            lines.append(f"- Gender Breakdown: {gen_dict}")

        return "\n".join(lines)

    def build_regional_strategy(self, df: Optional[pd.DataFrame] = None) -> str:
        target = self.df if df is None or df.empty else df
        if target.empty or "Region" not in target.columns or "Sales" not in target.columns:
            return "Regional sales data is unavailable to generate a 2-step strategy."

        reg_summary = (
            target.groupby("Region", observed=False)
            .agg(
                Total_Sales=("Sales", "sum"),
                Orders=("Sales", "count"),
                Avg_Order=("Sales", "mean"),
                Avg_Sat=("Customer_Satisfaction", "mean"),
            )
            .sort_values("Total_Sales", ascending=False)
        )

        top_region = reg_summary.index[0]
        bottom_region = reg_summary.index[-1]

        step1_lines = ["### Step 1: Regional Performance Diagnosis (Analyst Agent)"]
        for reg, row in reg_summary.iterrows():
            reg_df = target[target["Region"] == reg]
            top_prod = (
                reg_df.groupby("Product")["Sales"].sum().idxmax()
                if "Product" in reg_df.columns and not reg_df.empty
                else "N/A"
            )
            weak_prod = (
                reg_df.groupby("Product")["Sales"].sum().idxmin()
                if "Product" in reg_df.columns and not reg_df.empty
                else "N/A"
            )
            step1_lines.append(
                f"- **{reg}**: ${row['Total_Sales']:,.2f} across {int(row['Orders']):,} orders "
                f"(Avg Order: ${row['Avg_Order']:,.2f} | Satisfaction: {row['Avg_Sat']:.2f}/5.0 | "
                f"Leader: {top_prod}, Laggard: {weak_prod})"
            )

        step2_lines = ["\n### Step 2: Actionable Growth Recommendations by Region (Strategist Agent)"]
        for reg, row in reg_summary.iterrows():
            reg_df = target[target["Region"] == reg]
            top_prod = (
                reg_df.groupby("Product")["Sales"].sum().idxmax()
                if "Product" in reg_df.columns and not reg_df.empty
                else "top SKU"
            )
            weak_prod = (
                reg_df.groupby("Product")["Sales"].sum().idxmin()
                if "Product" in reg_df.columns and not reg_df.empty
                else "underperforming SKU"
            )

            if reg == top_region:
                action = (
                    f"Protect market leadership by introducing loyalty bundles pairing **{top_prod}** "
                    f"with **{weak_prod}** to lift average order value above ${row['Avg_Order']:,.2f}."
                )
            elif reg == bottom_region:
                action = (
                    f"Priority turnaround market (lowest total volume): run targeted regional promotions on "
                    f"**{weak_prod}** and audit post-purchase friction to lift satisfaction from {row['Avg_Sat']:.2f}/5.0."
                )
            else:
                action = (
                    f"Scale mid-tier momentum by expanding inventory & ad spend on **{top_prod}** while running "
                    f"cross-sell discounts on **{weak_prod}** to close the gap with {top_region}."
                )
            step2_lines.append(f"- **{reg} Strategy:** {action}")

        return "\n".join(step1_lines + step2_lines)


class InsightForgeRAG:
    def __init__(self, csv_path: Optional[str] = None):
        self.data_engine = SalesDataEngine(csv_path=csv_path)
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.model_name = "gpt-4o-mini"
        self.last_pdf_error: str = ""
        self._init_embeddings_and_store()
        self._init_llm()

    def _init_llm(self):
        load_dotenv()
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            self.llm = None
            return
        try:
            from langchain_openai import ChatOpenAI

            base_url = (os.getenv("OPENAI_API_BASE") or "").strip() or None
            self.model_name = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"

            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "api_key": api_key,
                "temperature": 0.2,
                "timeout": 20,
                "max_retries": 1,
            }
            if base_url:
                kwargs["base_url"] = base_url

            self.llm = ChatOpenAI(**kwargs)
        except Exception:
            self.llm = None

    def _init_embeddings_and_store(self):
        if HuggingFaceEmbeddings is None or FAISS is None:
            self.last_pdf_error = "langchain_huggingface or faiss is not imported in the active Python environment."
            return
        try:
            if self.embeddings is None:
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self._load_or_build_vector_store()
        except Exception as exc:
            self.last_pdf_error = f"Embedding/FAISS initialization failed: {exc}"
            self.vector_store = None

    @staticmethod
    def _clean_text(raw: str) -> str:
        return re.sub(r"\s+", " ", (raw or "")).strip()

    def _split_docs(self, pages: List[Any]) -> List[Any]:
        if not pages:
            return []
        if RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
            return splitter.split_documents(pages)

        chunks: List[Any] = []
        for p in pages:
            txt = p.page_content
            step = 750
            for start in range(0, len(txt), step):
                part = txt[start : start + 900].strip()
                if part:
                    chunks.append(make_doc(part, dict(p.metadata)))
        return chunks

    def _load_pdf_pages_with_fallback(self, pdf_path: str, fname: str) -> List[Any]:
        pages: List[Any] = []

        if PyPDFLoader is not None:
            try:
                loaded = PyPDFLoader(pdf_path).load()
                for idx, d in enumerate(loaded):
                    cleaned = self._clean_text(d.page_content)
                    if cleaned:
                        meta = dict(d.metadata or {})
                        meta["source_file"] = fname
                        raw_page = meta.get("page", idx)
                        meta["page"] = (int(raw_page) + 1) if isinstance(raw_page, int) and raw_page >= 0 else (idx + 1)
                        pages.append(make_doc(cleaned, meta))
                if pages:
                    return pages
            except Exception:
                pass

        try:
            import pypdf

            reader = pypdf.PdfReader(pdf_path)
            for idx, page in enumerate(reader.pages, start=1):
                cleaned = self._clean_text(page.extract_text())
                if cleaned:
                    pages.append(make_doc(cleaned, {"source_file": fname, "page": idx}))
            if pages:
                return pages
        except Exception:
            pass

        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    cleaned = self._clean_text(page.extract_text())
                    if cleaned:
                        pages.append(make_doc(cleaned, {"source_file": fname, "page": idx}))
            if pages:
                return pages
        except Exception:
            pass

        return pages

    def _extract_pdf_docs(self, folder_path: str) -> List[Any]:
        docs: List[Any] = []
        if not os.path.isdir(folder_path):
            return docs

        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(".pdf"):
                continue
            fpath = os.path.join(folder_path, fname)
            loaded = self._load_pdf_pages_with_fallback(fpath, fname)
            if loaded:
                docs.extend(self._split_docs(loaded))
        return docs

    def build_uploaded_pdf_store(self, uploaded_files: List[Any]) -> Tuple[Any, List[str]]:
        self.last_pdf_error = ""
        if not uploaded_files:
            return None, []

        names = [getattr(uf, "name", "uploaded.pdf") for uf in uploaded_files]

        if self.embeddings is None or FAISS is None:
            self._init_embeddings_and_store()

        if self.embeddings is None or FAISS is None:
            if not self.last_pdf_error:
                self.last_pdf_error = "Embeddings or FAISS is unavailable in the current environment."
            return None, names

        docs: List[Any] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for uf, fname in zip(uploaded_files, names):
                safe_name = os.path.basename(fname) or "uploaded.pdf"
                tmp_path = os.path.join(tmpdir, safe_name)
                raw_bytes = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()
                with open(tmp_path, "wb") as f:
                    f.write(raw_bytes)
                loaded = self._load_pdf_pages_with_fallback(tmp_path, safe_name)
                if loaded:
                    docs.extend(self._split_docs(loaded))

        if not docs:
            self.last_pdf_error = "No selectable text could be extracted (the PDF may be scanned/image-based)."
            return None, names

        try:
            store = FAISS.from_documents(docs, self.embeddings)
            return store, names
        except Exception as exc:
            self.last_pdf_error = f"Failed to build FAISS index from uploaded PDF: {exc}"
            return None, names

    def _load_or_build_vector_store(self):
        if self.embeddings is None or FAISS is None:
            return
        os.makedirs(INDEX_DIR, exist_ok=True)
        index_file = os.path.join(INDEX_DIR, "index.faiss")
        if os.path.exists(index_file):
            try:
                self.vector_store = FAISS.load_local(
                    INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True
                )
                return
            except Exception:
                pass

        docs: List[Any] = []
        for folder in {PDF_DIR, os.path.join(BASE_DIR, "Data"), BASE_DIR}:
            docs.extend(self._extract_pdf_docs(folder))

        if docs:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(INDEX_DIR)

    def audit_data_gaps(
        self,
        custom_df: Optional[pd.DataFrame] = None,
        uploaded_names: Optional[List[str]] = None,
        uploaded_files: Optional[List[Any]] = None,
        source_mode: str = "All Sources",
    ) -> str:
        lines: List[str] = ["### Uploaded PDF & Data Gaps Audit"]

        pdf_texts: List[Tuple[str, int, str]] = []
        if uploaded_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                for uf in uploaded_files:
                    fname = os.path.basename(getattr(uf, "name", "uploaded.pdf")) or "uploaded.pdf"
                    tmp_path = os.path.join(tmpdir, fname)
                    raw_bytes = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()
                    with open(tmp_path, "wb") as f:
                        f.write(raw_bytes)
                    pages = self._load_pdf_pages_with_fallback(tmp_path, fname)
                    combined = " ".join(p.page_content for p in pages)
                    pdf_texts.append((fname, len(pages), combined))
        elif uploaded_names:
            for fname in uploaded_names:
                pdf_texts.append((fname, 0, ""))

        if pdf_texts:
            lines.append("#### 1. Uploaded PDF Coverage & Missing Information")
            for fname, page_count, full_txt in pdf_texts:
                words = full_txt.split()
                word_count = len(words)
                if word_count == 0:
                    lines.append(
                        f"- **{fname}**: **0 words extracted** — likely scanned/image-only PDF or non-selectable text. "
                        f"**Gap:** Needs OCR or a text-selectable version before findings can be audited."
                    )
                    continue

                sample_preview = " ".join(words[:45]) + ("..." if word_count > 45 else "")
                missing_hints: List[str] = []
                low_txt = full_txt.lower()
                if not any(k in low_txt for k in ["metric", "kpi", "target", "benchmark", "score", "threshold", "%"]):
                    missing_hints.append("quantitative metrics / KPIs / targets")
                if not any(k in low_txt for k in ["timeline", "deadline", "milestone", "schedule", "date", "phase"]):
                    missing_hints.append("timelines / milestones / dates")
                if not any(k in low_txt for k in ["risk", "limitation", "assumption", "constraint", "dependency"]):
                    missing_hints.append("assumptions / risks / constraints")
                if not any(k in low_txt for k in ["owner", "stakeholder", "role", "team", "deliverable"]):
                    missing_hints.append("owners / deliverables / acceptance criteria")

                lines.append(
                    f"- **{fname}** ({page_count} page(s) with text, ~{word_count:,} words): "
                    f"Preview: *{sample_preview}*"
                )
                if missing_hints:
                    lines.append(f"  - **Potential PDF Gaps:** Missing or unclear {', '.join(missing_hints)}.")
                else:
                    lines.append(
                        "  - **Potential PDF Gaps:** Core metrics, timeline, risk, and deliverable terms appear present; "
                        "verify exact figures against source tables."
                    )
        else:
            lines.append("#### 1. Uploaded PDF Coverage")
            lines.append("- No PDFs are currently uploaded in Tab 2. Upload a PDF to audit its extracted content and gaps.")

        if source_mode != "Only Uploaded PDF(s)":
            target_df = custom_df if (custom_df is not None and not custom_df.empty) else self.data_engine.df
            lines.append("\n#### 2. Supporting Sales CSV & Vector Store Status")
            if target_df is None or target_df.empty:
                lines.append("- **Sales CSV:** No rows loaded.")
            else:
                null_cols = {col: int(cnt) for col, cnt in target_df.isna().sum().items() if cnt > 0}
                lines.append(
                    f"- **Sales CSV:** {len(target_df):,} rows × {len(target_df.columns)} cols | "
                    f"Nulls: {null_cols if null_cols else 'None'}"
                )
            has_disk_index = os.path.exists(os.path.join(INDEX_DIR, "index.faiss"))
            lines.append(f"- **FAISS Index on Disk:** {'Present' if has_disk_index else 'Not built yet'}")

        return "\n".join(lines)

    def load_memory(self) -> Dict[str, Any]:
        if not os.path.exists(MEMORY_FILE):
            return {}
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_memory(self, memory_data: Dict[str, Any]):
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get_user_history(self, user_name: str) -> List[Dict[str, Any]]:
        key = (user_name or "default_user").strip()
        memory = self.load_memory()
        records = memory.get(key, [])
        return records if isinstance(records, list) else []

    def save_turn_to_memory(
        self,
        user_name: str,
        question: str,
        answer: str,
        engine_label: str = "",
        scope_label: str = "",
        source_details: Optional[List[Dict[str, str]]] = None,
        reliability_label: str = "",
    ):
        key = (user_name or "default_user").strip()
        memory = self.load_memory()
        history = memory.get(key, [])
        if not isinstance(history, list):
            history = []
        history.append({
            "question": question,
            "answer": answer,
            "engine_label": engine_label,
            "scope_label": scope_label,
            "source_details": source_details or [],
            "reliability_label": reliability_label,
        })
        memory[key] = history
        self.save_memory(memory)

    def clear_user_history(self, user_name: str):
        key = (user_name or "default_user").strip()
        memory = self.load_memory()
        if key in memory:
            del memory[key]
        self.save_memory(memory)

    clear_user_memory = clear_user_history

    def parse_nl_to_conditions(self, query: str) -> Dict[str, Any]:
        q = (query or "").lower()
        conditions: List[Dict[str, str]] = []

        regions_found = [r.capitalize() for r in ["west", "east", "north", "south"] if re.search(rf"\b{r}\b", q)]
        prods_found = [p.title() for p in ["widget a", "widget b", "widget c", "widget d"] if re.search(rf"\b{p}\b", q)]

        for reg in regions_found:
            conditions.append({"col": "Region", "op": "equals (==)", "val1": reg, "val2": ""})

        for prod in prods_found:
            conditions.append({"col": "Product", "op": "equals (==)", "val1": prod, "val2": ""})

        if re.search(r"\bfemale\b|\bwomen\b", q):
            conditions.append({"col": "Customer_Gender", "op": "equals (==)", "val1": "Female", "val2": ""})
        if re.search(r"\bmale\b|\bmen\b", q):
            conditions.append({"col": "Customer_Gender", "op": "equals (==)", "val1": "Male", "val2": ""})

        age_map = [
            (r"\bgen z\b|\b18-25\b", "18-25"),
            (r"\byoung adult\b|\b26-35\b", "26-35"),
            (r"\bmid-career\b|\b36-50\b", "36-50"),
            (r"\bsenior\b|\b51\+\b", "51+"),
        ]
        for pat, label in age_map:
            if re.search(pat, q):
                conditions.append({"col": "Age_Group", "op": "contains", "val1": label, "val2": ""})

        if re.search(r"\bunhappy\b|\blow satisfaction\b|\bdissatisfied\b", q):
            conditions.append({"col": "Customer_Satisfaction", "op": "less than (<)", "val1": "2.5", "val2": ""})
        if re.search(r"(?<!un)\bhappy\b|\bhigh satisfaction\b|(?<!dis)\bsatisfied\b", q):
            conditions.append({"col": "Customer_Satisfaction", "op": "greater or equal (>=)", "val1": "4.0", "val2": ""})

        for m in re.finditer(r"(?:over|above|greater than|>|more than)\s*\$?(\d+(?:\.\d+)?)", q):
            conditions.append({"col": "Sales", "op": "greater than (>)", "val1": m.group(1), "val2": ""})

        for m in re.finditer(r"(?:under|below|less than|<)\s*\$?(\d+(?:\.\d+)?)", q):
            conditions.append({"col": "Sales", "op": "less than (<)", "val1": m.group(1), "val2": ""})

        same_col_multi = len(regions_found) > 1 or len(prods_found) > 1
        match_mode = "ANY (OR)" if (re.search(r"\bor\b", q) or same_col_multi) else "ALL (AND)"
        return {"conditions": conditions, "match_mode": match_mode}

    def _is_sales_query(self, question: str, source_mode: str = "All Sources") -> bool:
        if source_mode == "Current Tableau Filtered Slice":
            return True
        if source_mode == "Only Uploaded PDF(s)":
            return False

        sales_keywords = {
            "sales", "walmart", "region", "regions", "west", "east", "north", "south",
            "product", "products", "widget", "customer", "customers", "revenue",
            "order", "orders", "satisfaction", "gender", "male", "female", "age",
            "slice", "dataset", "csv", "recommendation", "strategy",
        }
        q_words = set(re.findall(r"\w+", (question or "").lower()))
        return bool(q_words & sales_keywords)

    def _is_regional_strategy_query(self, question: str) -> bool:
        q = (question or "").lower()
        return ("region" in q or "regional" in q) and any(
            w in q for w in ["recommend", "recommendation", "strategy", "improve", "grow", "action"]
        )

    def _filter_df_from_query(self, question: str) -> Tuple[pd.DataFrame, str]:
        df = self.data_engine.df.copy()
        if df.empty:
            return df, "Sales Dataset"

        q = (question or "").lower()
        applied: List[str] = []

        regs = [r.capitalize() for r in ["west", "east", "north", "south"] if re.search(rf"\b{r}\b", q)]
        if regs and "Region" in df.columns:
            df = df[df["Region"].astype(str).isin(regs)]
            applied.append(f"Region={','.join(regs)}")

        prods = [p.title() for p in ["widget a", "widget b", "widget c", "widget d"] if re.search(rf"\b{p}\b", q)]
        if prods and "Product" in df.columns:
            df = df[df["Product"].astype(str).isin(prods)]
            applied.append(f"Product={','.join(prods)}")

        if re.search(r"\bfemale\b|\bwomen\b", q) and "Customer_Gender" in df.columns:
            df = df[df["Customer_Gender"].astype(str).str.lower() == "female"]
            applied.append("Gender=Female")
        elif re.search(r"\bmale\b|\bmen\b", q) and "Customer_Gender" in df.columns:
            df = df[df["Customer_Gender"].astype(str).str.lower() == "male"]
            applied.append("Gender=Male")

        if applied:
            return df, f"Filtered Walmart Sales ({' | '.join(applied)})"
        return df, "Full Walmart Sales Dataset"

    def _fetch_wikipedia_summary(self, query: str) -> Tuple[str, str]:
        try:
            clean_q = re.sub(
                r"\b(anything on|tell me about|what is|what are|who is|how is|explain|define|full form of|full form|stands for|meaning of|according to wikipedia|wikipedia)\b",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip(" ?.,!")
            search_term = clean_q if len(clean_q) >= 2 else query

            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": search_term,
                "format": "json",
                "srlimit": 5,
            }
            headers = {"User-Agent": "InsightForgeAI/1.0 (local-dev@insightforge.ai)"}
            resp = requests.get(search_url, params=params, headers=headers, timeout=6)
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
            if not results:
                return "", f"Wikipedia search returned no results for '{search_term}'."

            ordered = sorted(
                results,
                key=lambda item: 0 if search_term.lower() in item.get("title", "").lower() else 1,
            )

            for item in ordered:
                title = item.get("title", "")
                if not title:
                    continue
                sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
                sresp = requests.get(sum_url, headers=headers, timeout=6)
                sresp.raise_for_status()
                extract = sresp.json().get("extract", "")
                if extract and "may refer to:" not in extract.lower():
                    exact_note = (
                        ""
                        if search_term.lower() in title.lower()
                        else f" (no standalone article titled '{search_term}', returned closest match)"
                    )
                    reason = (
                        f"No PDF chunk matched; queried Wikipedia for '{search_term}' "
                        f"-> matched article '{title}'{exact_note}"
                    )
                    formatted = f"[Wikipedia — {title}]\n*Why Wikipedia gave this answer: {reason}*\n\n{extract}"
                    return formatted, reason

            return "", f"Wikipedia search found pages for '{search_term}', but no non-disambiguation summary was returned."
        except Exception as exc:
            return "", f"Wikipedia lookup failed ({type(exc).__name__}: {exc})."

    def query(
        self,
        question: str,
        source_mode: str = "All Sources",
        custom_df: Optional[pd.DataFrame] = None,
        use_wiki_fallback: bool = True,
        uploaded_store: Any = None,
    ) -> Dict[str, Any]:
        needs_sales_data = self._is_sales_query(question, source_mode=source_mode)
        is_strategy = self._is_regional_strategy_query(question)

        stats_context = ""
        strategy_output = ""
        scope_label = (
            "Uploaded PDFs Only (CSV Skipped)"
            if source_mode == "Only Uploaded PDF(s)"
            else "PDF / External Knowledge Only (CSV Skipped)"
        )
        chain_type = "Standard RAG + Fairness Guardrail Chain"

        if needs_sales_data:
            if source_mode == "Current Tableau Filtered Slice" and custom_df is not None:
                active_df = custom_df
                scope_label = f"Tableau Filtered Slice ({len(active_df):,} rows)"
            else:
                active_df, scope_label = self._filter_df_from_query(question)

            stats_context = self.data_engine.summarize_dataframe(active_df, scope_name=scope_label)
            if is_strategy:
                chain_type = "2-Step Sequential Chain (Regional Analyst -> Strategist)"
                strategy_output = self.data_engine.build_regional_strategy(active_df)

        doc_snippets: List[str] = []
        source_details: List[Dict[str, str]] = []
        wiki_reason = ""

        if source_mode == "Only Uploaded PDF(s)":
            active_store = uploaded_store
            max_distance = 2.25
        else:
            active_store = uploaded_store if uploaded_store is not None else self.vector_store
            max_distance = 1.15

        if active_store is not None and not needs_sales_data:
            try:
                hits = active_store.similarity_search_with_score(question, k=4)
                for doc, score in hits:
                    if score <= max_distance:
                        fname = doc.metadata.get("source_file", "Uploaded_PDF.pdf")
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content.strip()
                        doc_snippets.append(f"**[{fname} — Page {page}]**: {snippet}")
                        source_details.append({
                            "Source": fname,
                            "Page": str(page),
                            "Relevance": f"FAISS L2 distance = {score:.3f}",
                        })
            except Exception:
                pass

        if not doc_snippets and use_wiki_fallback and not needs_sales_data and source_mode != "Only Uploaded PDF(s)":
            wiki_txt, wiki_reason = self._fetch_wikipedia_summary(question)
            if wiki_txt:
                doc_snippets.append(wiki_txt)
                source_details.append({
                    "Source": "Wikipedia API",
                    "Page": "Web",
                    "Relevance": wiki_reason,
                })

        fallback_local = strategy_output or stats_context or "\n\n".join(doc_snippets)

        if self.llm is not None and (stats_context or doc_snippets):
            prompt = (
                "You are InsightForge AI, an objective enterprise BI assistant. "
                "Standardize and synthesize the retrieved context into a clear, accurate, well-structured response "
                "without demographic bias. Do not invent facts beyond the provided context.\n\n"
                f"Context:\n{stats_context}\n{strategy_output}\n" + "\n".join(doc_snippets) +
                f"\n\nUser Question: {question}"
            )
            try:
                resp = self.llm.invoke(prompt)
                answer = resp.content
            except Exception as exc:
                err_msg = f"**LLM Error ({type(exc).__name__}):** {exc}"
                answer = f"{err_msg}\n\n{fallback_local}" if fallback_local else err_msg
        elif self.llm is not None and source_mode != "Only Uploaded PDF(s)":
            scope_label = "General LLM Knowledge (No PDF or Wikipedia match)"
            chain_type = "Direct General LLM Fallback + Fairness Guardrail"
            general_prompt = (
                "You are InsightForge AI, an objective enterprise BI assistant. "
                "Neither local PDF/CSV sources nor Wikipedia returned a match for this query. "
                "Answer clearly and accurately from general knowledge without demographic bias.\n\n"
                f"User Question: {question}"
            )
            try:
                resp = self.llm.invoke(general_prompt)
                answer = resp.content
            except Exception as exc:
                answer = f"**LLM Error ({type(exc).__name__}):** {exc}"
        else:
            parts: List[str] = []
            if is_strategy and strategy_output:
                parts.append(strategy_output)
            elif stats_context:
                parts.append(stats_context)
            if doc_snippets:
                if source_mode == "Only Uploaded PDF(s)":
                    parts.append("### Extracted Passages from Uploaded PDF(s)\n" + "\n\n".join(doc_snippets))
                else:
                    parts.append("\n\n".join(doc_snippets))
            if not parts:
                if source_mode == "Only Uploaded PDF(s)" and uploaded_store is None:
                    detail = f" ({self.last_pdf_error})" if self.last_pdf_error else ""
                    parts.append(
                        f"No searchable text was indexed from the uploaded PDF(s){detail}. "
                        "Try a text-based PDF (such as one from the Data folder) to verify extraction."
                    )
                elif wiki_reason:
                    parts.append(f"No matching PDF records found. {wiki_reason}")
                else:
                    parts.append(
                        "No matching local CSV/PDF/Wikipedia records found, and no LLM API key is configured. "
                        "Add an API key in the sidebar or `.env` to enable general LLM answers."
                    )
            answer = "\n\n".join(parts)

        return {
            "answer": answer,
            "scope_label": scope_label,
            "chain_type": chain_type,
            "stats_used": stats_context,
            "doc_context": "\n\n".join(doc_snippets),
            "source_details": source_details,
            "engine_label": f"LLM ({self.model_name})" if self.llm else "InsightForgeRAG (Local Mode — No API Key)",
        }

    def stream_ask(
        self,
        question: str,
        full_name: str = "",
        uploaded_store: Any = None,
        source_mode: str = "All Sources",
        custom_df: Optional[pd.DataFrame] = None,
        use_wiki_fallback: bool = True,
    ) -> Dict[str, Any]:
        res = self.query(
            question=question,
            source_mode=source_mode,
            custom_df=custom_df,
            use_wiki_fallback=use_wiki_fallback,
            uploaded_store=uploaded_store,
        )

        def _token_stream() -> Generator[str, None, None]:
            for chunk in re.split(r"(\s+)", res["answer"]):
                if chunk:
                    yield chunk

        res["stream"] = _token_stream()
        return res