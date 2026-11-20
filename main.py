import importlib
import json
import os
import pandas as pd
import streamlit as st
import src.rag_engine as rag_mod
from src.evaluator import InsightForgeEvaluator
from src.visuals import SalesVisualizer

st.set_page_config(
    page_title="InsightForge AI | Enterprise Business Intelligence",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #4A90E2;
        border-radius: 10px;
        padding: 18px;
        background-color: #f8fbff;
    }
    .citation-badge {
        display: inline-block;
        background-color: #eef5ff;
        border: 1px solid #b6d4fe;
        color: #084298;
        border-radius: 14px;
        padding: 3px 11px;
        font-size: 0.82rem;
        margin: 2px 6px 6px 0;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COMMENTS_FILE = "row_comments.json"


def load_saved_comments() -> dict:
    if os.path.exists(COMMENTS_FILE):
        try:
            with open(COMMENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_comments_to_disk(comments_dict: dict):
    with open(COMMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(comments_dict, f, indent=2, ensure_ascii=False)


def apply_single_condition(frame: pd.DataFrame, col: str, op: str, val1: str, val2: str = "") -> pd.Series:
    series = frame[col]
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if is_numeric:
        num1 = float(val1)
        if op == "equals (==)":
            return series == num1
        elif op == "not equals (!=)":
            return series != num1
        elif op == "greater than (>)":
            return series > num1
        elif op == "greater or equal (>=)":
            return series >= num1
        elif op == "less than (<)":
            return series < num1
        elif op == "less or equal (<=)":
            return series <= num1
        elif op == "between":
            num2 = float(val2)
            low, high = min(num1, num2), max(num1, num2)
            return series.between(low, high)
    else:
        s_str = series.astype(str)
        v1 = val1.strip()
        if op == "equals (==)":
            return s_str.str.lower() == v1.lower()
        elif op == "not equals (!=)":
            return s_str.str.lower() != v1.lower()
        elif op == "contains":
            return s_str.str.contains(v1, case=False, na=False)
        elif op == "does not contain":
            return ~s_str.str.contains(v1, case=False, na=False)
        elif op == "starts with":
            return s_str.str.lower().str.startswith(v1.lower())
        elif op == "greater or equal (>=)":
            return s_str >= v1
        elif op == "less or equal (<=)":
            return s_str <= v1

    return pd.Series(True, index=frame.index)


def get_operators_for_col(frame: pd.DataFrame, col: str) -> list[str]:
    if pd.api.types.is_numeric_dtype(frame[col]):
        return [
            "greater than (>)",
            "less than (<)",
            "greater or equal (>=)",
            "less or equal (<=)",
            "equals (==)",
            "not equals (!=)",
            "between",
        ]
    return [
        "contains",
        "equals (==)",
        "not equals (!=)",
        "does not contain",
        "starts with",
        "greater or equal (>=)",
        "less or equal (<=)",
    ]


def validate_condition(frame: pd.DataFrame, col: str, op: str, val1: str, val2: str) -> str | None:
    if not val1.strip():
        return "Value cannot be empty."
    if op == "between" and not val2.strip():
        return "Both Min Value and Max Value are required for 'between'."
    if pd.api.types.is_numeric_dtype(frame[col]):
        try:
            float(val1)
            if op == "between":
                float(val2)
        except ValueError:
            return f"'{col}' is a numeric column. Please enter valid numbers."
    return None


def render_citation_badges(
    source_details: list[dict],
    scope_label: str,
    engine_label: str = "",
    reliability_label: str = "",
):
    badges = []
    if engine_label:
        badges.append(f'<span class="citation-badge">Engine: {engine_label}</span>')
    if scope_label:
        badges.append(f'<span class="citation-badge">CSV Scope: {scope_label}</span>')
    if reliability_label:
        badges.append(f'<span class="citation-badge">Reliability: {reliability_label}</span>')

    seen = set()
    for src in source_details or []:
        src_name = src.get("Source", "")
        if not src_name or src_name == "system":
            continue
        key = (src_name, str(src.get("Page", "")))
        if key not in seen:
            seen.add(key)
            badges.append(
                f'<span class="citation-badge">RAG Doc: {key[0]} | Page {key[1]} ({src.get("Relevance", "")})</span>'
            )

    if not seen:
        badges.append('<span class="citation-badge">RAG Docs: None matched (CSV Stats Only)</span>')

    st.markdown("".join(badges), unsafe_allow_html=True)


def run_safe_eval(eval_engine: InsightForgeEvaluator, question: str, answer: str, context: str = "") -> dict:
    try:
        raw = eval_engine.evaluate_single(question=question, answer=answer, context=context)
    except Exception as exc:
        raw = eval_engine._heuristic_score(
            question=question,
            answer=answer,
            context=context,
            reason=f"Evaluation fallback ({type(exc).__name__})",
        )

    score_1_to_5 = round(float(raw.get("overall", 0.75)) * 5.0, 2)
    pillar_1_to_5 = round(min(5.0, max(1.0, float(raw.get("groundedness", 0.75)) * 5.0)), 2)
    rel_1_to_5 = round(min(5.0, max(1.0, float(raw.get("relevance", 0.75)) * 5.0)), 2)
    fair_1_to_5 = round(min(5.0, max(1.0, float(raw.get("fairness", 1.0)) * 5.0)), 2)

    return {
        "Question": question,
        "Correctness (1-5)": rel_1_to_5,
        "Faithfulness (1-5)": pillar_1_to_5,
        "Completeness (1-5)": score_1_to_5,
        "Fairness (1-5)": fair_1_to_5,
        "Average Score": round((rel_1_to_5 + pillar_1_to_5 + score_1_to_5 + fair_1_to_5) / 4.0, 2),
        "Mode": raw.get("mode", "Local Heuristic"),
        "Judge Reasoning": raw.get("notes", ""),
    }


@st.cache_resource
def load_engine(_file_mtime: float):
    importlib.reload(rag_mod)
    return rag_mod.InsightForgeRAG()


rag_path = os.path.join("src", "rag_engine.py")
rag_mtime = os.path.getmtime(rag_path) if os.path.exists(rag_path) else 0.0
rag_bot = load_engine(rag_mtime)

with st.sidebar:
    st.subheader("LLM Provider Setup (Optional)")
    st.caption("Leave blank to run in **Local Mode** (Pandas CSV + FAISS PDF + Wikipedia).")
    sb_key = st.text_input("API Key", type="password", key="sb_api_key")
    sb_url = st.text_input(
        "Base URL (Optional)",
        placeholder="e.g., https://generativelanguage.googleapis.com/v1beta/openai/",
        key="sb_base_url",
    )
    sb_model = st.text_input("Model Name", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), key="sb_model")

    if sb_key.strip():
        os.environ["OPENAI_API_KEY"] = sb_key.strip()
        if sb_url.strip():
            os.environ["OPENAI_API_BASE"] = sb_url.strip()
        if sb_model.strip():
            os.environ["OPENAI_MODEL"] = sb_model.strip()
        rag_bot._init_llm()

    if getattr(rag_bot, "llm", None) is not None:
        st.success(f"LLM Active: **{rag_bot.model_name}**")
    else:
        st.info("Local Mode Active (No API key set)")

evaluator = InsightForgeEvaluator(
    llm=getattr(rag_bot, "llm", None),
    model_name=getattr(rag_bot, "model_name", "gpt-4o-mini"),
)
df = rag_bot.data_engine.df.copy()

st.title("InsightForge AI — Enterprise Business Intelligence Suite")
st.caption(
    f"Engine: **{rag_bot.model_name if getattr(rag_bot, 'llm', None) else 'Local Offline Mode (Pandas + FAISS + Wikipedia)'}** | "
    "Powered by Pandas, FAISS RAG, Sequential Chains, Guardrails & Plotly"
)

if "live_eval_history" not in st.session_state:
    st.session_state["live_eval_history"] = []

if not df.empty:
    if "Row_ID" not in df.columns:
        df.insert(0, "Row_ID", [f"ORD-{i + 1:04d}" for i in range(len(df))])

    if "Age_Group" not in df.columns and "Customer_Age" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Customer_Age"],
            bins=[17, 25, 35, 50, 120],
            labels=["18-25 (Gen Z)", "26-35 (Young Adult)", "36-50 (Mid-Career)", "51+ (Senior)"],
        )

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        if "Year_Month" not in df.columns:
            df["Year_Month"] = df["Date"].dt.to_period("M").astype(str)
        if "Quarter" not in df.columns:
            df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)

viz = SalesVisualizer(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${df['Sales'].sum():,.2f}" if not df.empty and "Sales" in df.columns else "$0.00")
col2.metric("Average Order Value", f"${df['Sales'].mean():,.2f}" if not df.empty and "Sales" in df.columns else "$0.00")
col3.metric("Median Order Value", f"${df['Sales'].median():,.2f}" if not df.empty and "Sales" in df.columns else "$0.00")
col4.metric(
    "Avg Customer Satisfaction",
    f"{df['Customer_Satisfaction'].mean():.2f} / 5.0" if not df.empty and "Customer_Satisfaction" in df.columns else "N/A",
)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Interactive BI Dashboard",
    "Ask InsightForge AI",
    "Tableau Crosstab & Condition Studio",
    "LLM Evaluation & Quality Judge",
])

# ==============================================================================
# TAB 1: INTERACTIVE BI DASHBOARD
# ==============================================================================
with tab1:
    st.subheader("Search & Filter Sales by Age, Gender, Region & Product")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        age_options = ["All"] + [str(x) for x in df["Age_Group"].dropna().unique().tolist()] if "Age_Group" in df.columns else ["All"]
        selected_age = st.selectbox("Select Age Group", options=age_options)

    with f_col2:
        gender_options = ["All"] + sorted(df["Customer_Gender"].dropna().unique().tolist()) if "Customer_Gender" in df.columns else ["All"]
        selected_gender = st.selectbox("Select Gender", options=gender_options)

    with f_col3:
        region_options = ["All"] + sorted(df["Region"].dropna().unique().tolist()) if "Region" in df.columns else ["All"]
        selected_region = st.selectbox("Select Region", options=region_options)

    with f_col4:
        product_options = ["All"] + sorted(df["Product"].dropna().unique().tolist()) if "Product" in df.columns else ["All"]
        selected_product = st.selectbox("Select Product", options=product_options)

    search_col, btn_search_col, btn_reset_col = st.columns([3, 1, 1])
    with search_col:
        keyword_query = st.text_input(
            "Keyword Search (optional):",
            placeholder="Type a product, region, or exact age (e.g., 35 or North)...",
        ).strip()

    with btn_search_col:
        st.write("")
        st.write("")
        search_clicked = st.button("Search Filtered Sales", width="stretch")

    with btn_reset_col:
        st.write("")
        st.write("")
        reset_clicked = st.button("Reset Filters", width="stretch")

    if reset_clicked:
        st.rerun()

    filtered_df = df.copy()
    if not filtered_df.empty:
        if selected_age != "All" and "Age_Group" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Age_Group"].astype(str) == selected_age]
        if selected_gender != "All" and "Customer_Gender" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Customer_Gender"] == selected_gender]
        if selected_region != "All" and "Region" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Region"] == selected_region]
        if selected_product != "All" and "Product" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Product"] == selected_product]
        if keyword_query:
            mask = filtered_df.astype(str).apply(
                lambda c: c.str.contains(keyword_query, case=False, na=False)
            ).any(axis=1)
            filtered_df = filtered_df[mask]

    is_filtered = (
        search_clicked
        or selected_age != "All"
        or selected_gender != "All"
        or selected_region != "All"
        or selected_product != "All"
        or bool(keyword_query)
    )

    if is_filtered:
        st.markdown("**Filtered Search Results:**")
        if filtered_df.empty:
            st.warning("No sales records match your selected filters.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Filtered Sales Total", f"${filtered_df['Sales'].sum():,.2f}")
            m2.metric("Matching Orders", f"{len(filtered_df):,}")
            m3.metric("Filtered Avg Order", f"${filtered_df['Sales'].mean():,.2f}")
            m4.metric("Filtered Satisfaction", f"{filtered_df['Customer_Satisfaction'].mean():.2f} / 5.0")

            with st.expander("View Matching Sales Records Table", expanded=True):
                st.dataframe(filtered_df.head(50), width="stretch")
                st.download_button(
                    label=f"Download Filtered Data ({len(filtered_df):,} rows) as CSV",
                    data=filtered_df.to_csv(index=False).encode("utf-8"),
                    file_name="insightforge_filtered_sales.csv",
                    mime="text/csv",
                )

    st.divider()

    st.subheader("Sales Trends, Yearly Growth, Correlations & Demographics")
    if not df.empty:
        st.plotly_chart(viz.plot_monthly_sales_trend(), width="stretch")

        c_row1_left, c_row1_right = st.columns(2)
        with c_row1_left:
            st.plotly_chart(viz.plot_product_and_region(), width="stretch")
        with c_row1_right:
            st.plotly_chart(viz.plot_customer_demographics(), width="stretch")

        c_row2_left, c_row2_right = st.columns(2)
        with c_row2_left:
            st.plotly_chart(viz.plot_yearly_performance(), width="stretch")
        with c_row2_right:
            st.plotly_chart(viz.plot_correlation_heatmap(), width="stretch")
    else:
        st.info("No sales CSV data loaded yet.")

# ==============================================================================
# TAB 2: ASK INSIGHTFORGE AI
# ==============================================================================
with tab2:
    st.subheader("Personalized BI Assistant (Live Token Streaming, Citations, Fairness Guardrails & Tableau Bridge)")

    up_col, filter_col, wiki_col = st.columns([1.8, 1.2, 0.8])
    with up_col:
        uploaded_pdfs = st.file_uploader(
            "Drag and drop PDF file(s) onto the blue box below (or click Browse files):",
            type=["pdf"],
            accept_multiple_files=True,
            key="user_uploaded_pdfs",
        )

    has_tableau_slice = (
        "tableau_active_slice" in st.session_state
        and isinstance(st.session_state["tableau_active_slice"], pd.DataFrame)
        and len(st.session_state["tableau_active_slice"]) < len(df)
    )

    with filter_col:
        scope_options = ["All Sources", "Only Built-in CSV & PDFs"]
        if uploaded_pdfs:
            scope_options.insert(0, "Only Uploaded PDF(s)")
        if has_tableau_slice:
            scope_options.insert(0, "Current Tableau Filtered Slice")

        source_mode = st.selectbox(
            "Knowledge Source Filter:",
            options=scope_options,
            index=0,
            help="Choose whether the bot answers from all data, your Tab 3 Tableau slice, or only uploaded PDFs.",
        )
        if has_tableau_slice:
            st.caption(f"Tableau Slice Ready: **{len(st.session_state['tableau_active_slice']):,}** filtered rows")

    with wiki_col:
        st.write("")
        st.write("")
        enable_wiki = st.checkbox(
            "Wikipedia Tool Fallback",
            value=True,
            help="Fetch external Wikipedia context automatically if no local PDF matches a general concept question.",
        )

    uploaded_store = None
    if uploaded_pdfs and hasattr(rag_bot, "build_uploaded_pdf_store"):
        file_sig = tuple((f.name, f.size) for f in uploaded_pdfs)
        if st.session_state.get("uploaded_pdf_sig") != file_sig:
            with st.spinner("Indexing uploaded PDF(s) into FAISS..."):
                store, names = rag_bot.build_uploaded_pdf_store(uploaded_pdfs)
                st.session_state["uploaded_pdf_sig"] = file_sig
                st.session_state["uploaded_pdf_store"] = store
                st.session_state["uploaded_pdf_names"] = names
        uploaded_store = st.session_state.get("uploaded_pdf_store")
        indexed_names = st.session_state.get("uploaded_pdf_names", [])
        status_txt = ", ".join(indexed_names) if indexed_names else "None (0 text chunks extracted)"
        st.caption(f"Indexed Uploaded PDF(s): {status_txt}")
    else:
        st.session_state.pop("uploaded_pdf_sig", None)
        st.session_state.pop("uploaded_pdf_store", None)
        st.session_state.pop("uploaded_pdf_names", None)

    st.divider()

    name_col, btn_col = st.columns([3, 1])
    with name_col:
        user_full_name = st.text_input(
            "Enter your Full Name to identify your session memory:",
            placeholder="e.g., Parminder Wasan",
        ).strip()

    if not user_full_name:
        st.info("Please enter your full name above to start chatting. Your question history is stored separately under your name.")
    else:
        past_records = rag_bot.get_user_history(user_full_name) if hasattr(rag_bot, "get_user_history") else []

        with btn_col:
            st.write("")
            st.write("")
            if st.button("Clear My Memory", width="stretch", help="Deletes saved chat history for your name."):
                if hasattr(rag_bot, "clear_user_history"):
                    rag_bot.clear_user_history(user_full_name)
                elif hasattr(rag_bot, "clear_user_memory"):
                    rag_bot.clear_user_memory(user_full_name)
                for key in ["messages", "chat_history", "last_result"]:
                    st.session_state.pop(key, None)
                st.rerun()

        if past_records:
            st.success(f"Welcome back, **{user_full_name}**! Loaded {len(past_records)} previous question(s) from your profile.")
        else:
            st.info(f"Hello, **{user_full_name}**! Starting a fresh memory profile for you.")

        st.markdown("**Quick Action Prompts (Hover over any button to see what it runs):**")
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        quick_prompt = None
        if q_col1.button(
            "2-Step Regional Strategy",
            width="stretch",
            help="Runs a 2-step Analyst -> Strategist chain on your regional sales data.",
        ):
            quick_prompt = "Look at my analysis and give me a recommendation to improve my sales in every region."
        if q_col2.button(
            "Analyze Tableau Slice",
            width="stretch",
            help="Analyzes the active filtered dataset slice from Tab 3.",
        ):
            quick_prompt = "Analyze the active dataset slice and identify key satisfaction and sales drivers."
        if q_col3.button(
            "Summarize Uploaded PDF",
            width="stretch",
            help="Queries FAISS vector store for key findings in your uploaded PDF(s).",
        ):
            quick_prompt = "Summarize the key findings from the uploaded PDF document."
        if q_col4.button(
            "Audit Data Gaps & Missing Info",
            type="primary",
            width="stretch",
            help="Evaluates current CSV columns, null counts & PDFs to identify what additional data should be added.",
        ) and hasattr(rag_bot, "audit_data_gaps"):
            with st.chat_message("assistant"):
                audit_report = rag_bot.audit_data_gaps(
                    custom_df=st.session_state.get("tableau_active_slice"),
                    uploaded_names=st.session_state.get("uploaded_pdf_names"),
                    uploaded_files=uploaded_pdfs,
                    source_mode=source_mode,
                )
                st.markdown(audit_report)
                if hasattr(rag_bot, "save_turn_to_memory"):
                    rag_bot.save_turn_to_memory(
                        user_full_name,
                        "Audit current data completeness and identify what additional data is needed.",
                        audit_report,
                        engine_label="Data Schema & RAG Completeness Auditor",
                        scope_label="Current Active Dataset & Vector Store",
                        reliability_label="High (100% Schema Verified)",
                    )

        for item in past_records:
            with st.chat_message("user"):
                st.markdown(item["question"])
            with st.chat_message("assistant"):
                st.markdown(item["answer"])
                render_citation_badges(
                    item.get("source_details", []),
                    item.get("scope_label", "Full Walmart Sales Dataset"),
                    engine_label=item.get("engine_label", ""),
                    reliability_label=item.get("reliability_label", ""),
                )

        typed_prompt = st.chat_input(
            f"Ask anything as {user_full_name} — searches CSV + PDFs first, applies fairness guardrails & auto-logs quality in Tab 4..."
        )
        active_prompt = quick_prompt or typed_prompt

        if active_prompt:
            with st.chat_message("user"):
                st.markdown(active_prompt)

            with st.chat_message("assistant"):
                bundle = rag_bot.stream_ask(
                    active_prompt,
                    full_name=user_full_name,
                    uploaded_store=uploaded_store,
                    source_mode=source_mode,
                    custom_df=st.session_state.get("tableau_active_slice"),
                    use_wiki_fallback=enable_wiki,
                )
                st.caption(f"Pipeline Used: **{bundle.get('chain_type', 'RAG Chain')}**")
                streamed_answer = st.write_stream(bundle["stream"])

                rel_label = "High (Grounded)" if bundle.get("source_details") or bundle.get("stats_used") else "Moderate"

                live_eval = run_safe_eval(
                    evaluator,
                    question=active_prompt,
                    answer=streamed_answer,
                    context=f"{bundle.get('stats_used', '')}\n{bundle.get('doc_context', '')}",
                )
                st.session_state["live_eval_history"].append(live_eval)

                if hasattr(rag_bot, "save_turn_to_memory"):
                    rag_bot.save_turn_to_memory(
                        user_full_name,
                        active_prompt,
                        streamed_answer,
                        engine_label=bundle.get("engine_label", ""),
                        scope_label=bundle.get("scope_label", ""),
                        source_details=bundle.get("source_details", []),
                        reliability_label=rel_label,
                    )

                st.caption(
                    f"Auto-Eval Score: **{live_eval['Average Score']} / 5** "
                    f"(Correctness: {live_eval['Correctness (1-5)']}, "
                    f"Faithfulness: {live_eval['Faithfulness (1-5)']}, "
                    f"Completeness: {live_eval['Completeness (1-5)']}, "
                    f"Fairness: {live_eval['Fairness (1-5)']}) — Logged in Tab 4"
                )

                if live_eval["Fairness (1-5)"] < 3 or live_eval["Faithfulness (1-5)"] < 3:
                    st.warning(f"Quality / Fairness Guardrail Alert: {live_eval['Judge Reasoning']}")

                st.markdown("**Cited Knowledge Sources & Reliability Audit:**")
                render_citation_badges(
                    bundle.get("source_details", []),
                    bundle.get("scope_label", ""),
                    engine_label=bundle.get("engine_label", ""),
                    reliability_label=rel_label,
                )

                with st.expander("View Raw Statistical Context & Source Breakdown Table"):
                    if bundle.get("source_details"):
                        st.dataframe(pd.DataFrame(bundle["source_details"]), width="stretch")
                    st.text(bundle.get("stats_used", ""))

# ==============================================================================
# TAB 3: TABLEAU CROSSTAB & CONDITION STUDIO
# ==============================================================================
with tab3:
    st.subheader("0. Ask Tableau AI — Natural Language Auto-Filter")
    st.caption("Type a filter request in plain English and click **Apply AI Filter** to automatically build the rules below.")

    nl_col1, nl_col2 = st.columns([4, 1])
    with nl_col1:
        nl_filter_query = st.text_input(
            "Ask Tableau in plain English:",
            placeholder="e.g., Show unhappy Gen Z customers in the West who spent over 500",
            key="tab3_nl_input",
            label_visibility="collapsed",
        ).strip()
    with nl_col2:
        apply_nl_clicked = st.button("Apply AI Filter", type="primary", width="stretch", key="tab3_nl_btn")

    if apply_nl_clicked:
        if not nl_filter_query:
            st.warning("Type a filter sentence first (e.g., 'East region sales over 600').")
        else:
            with st.spinner("Translating your sentence into Tableau filter conditions..."):
                parsed_result = rag_bot.parse_nl_to_conditions(nl_filter_query)
                new_conds = parsed_result.get("conditions", [])
                if not new_conds:
                    st.warning("Could not detect specific column filters from that sentence. Try mentioning Region, Product, Sales, Age Group, or Satisfaction.")
                else:
                    st.session_state["custom_conditions"] = new_conds
                    st.session_state["tab3_cond_match_mode"] = parsed_result.get("match_mode", "ALL (AND)")
                    st.rerun()

    st.divider()
    st.subheader("1. Custom Condition Filter Builder")

    tableau_df = df.copy()
    saved_comments = load_saved_comments()
    if not tableau_df.empty and "Row_ID" in tableau_df.columns:
        tableau_df["Comments"] = tableau_df["Row_ID"].map(lambda rid: saved_comments.get(rid, ""))

    if "custom_conditions" not in st.session_state:
        st.session_state["custom_conditions"] = []

    c_col1, c_col2, c_col3, c_col4, c_col5 = st.columns([1.3, 1.3, 1.2, 1.2, 1])

    filterable_cols = [
        "Sales",
        "Customer_Satisfaction",
        "Customer_Age",
        "Region",
        "Product",
        "Customer_Gender",
        "Age_Group",
        "Year_Month",
        "Quarter",
        "Comments",
    ]

    with c_col1:
        cond_col = st.selectbox("Column", options=filterable_cols, index=0, key="tab3_new_cond_col")

    op_options = get_operators_for_col(tableau_df, cond_col) if cond_col in tableau_df.columns else ["equals (==)"]

    with c_col2:
        cond_op = st.selectbox("Condition", options=op_options, key="tab3_new_cond_op")

    with c_col3:
        is_num_col = cond_col in tableau_df.columns and pd.api.types.is_numeric_dtype(tableau_df[cond_col])
        placeholder_txt = "e.g., 500 or 2.5" if is_num_col else "e.g., East, 2024-07, or refund"
        cond_val1 = st.text_input("Value", placeholder=placeholder_txt, key="tab3_new_cond_val1")

    with c_col4:
        if cond_op == "between":
            cond_val2 = st.text_input("Max Value", placeholder="e.g., 1000", key="tab3_new_cond_val2")
        else:
            cond_val2 = ""
            st.text_input("Max Value (for 'between')", value="—", disabled=True, key="tab3_new_cond_val2_dis")

    with c_col5:
        st.write("")
        st.write("")
        add_clicked = st.button("Add Condition", key="tab3_add_cond_btn", width="stretch")

    if add_clicked:
        err = validate_condition(tableau_df, cond_col, cond_op, cond_val1, cond_val2) if cond_col in tableau_df.columns else None
        if err:
            st.error(err)
        else:
            st.session_state["custom_conditions"].append({
                "col": cond_col,
                "op": cond_op,
                "val1": cond_val1.strip(),
                "val2": cond_val2.strip(),
            })
            st.rerun()

    if st.session_state["custom_conditions"]:
        st.markdown("**Active Conditions (Edit any field directly and click Update, or click Delete):**")

        top_bar1, top_bar2 = st.columns([3, 1])
        with top_bar1:
            match_mode = st.radio(
                "Combine Conditions With:",
                options=["ALL (AND)", "ANY (OR)"],
                horizontal=True,
                key="tab3_cond_match_mode",
            )
        with top_bar2:
            if st.button("Clear All Conditions", key="tab3_clear_all_cond", width="stretch"):
                st.session_state["custom_conditions"] = []
                st.rerun()

        for idx, cond in enumerate(list(st.session_state["custom_conditions"])):
            e1, e2, e3, e4, e5, e6 = st.columns([1.2, 1.3, 1.1, 1.1, 0.7, 0.7])

            with e1:
                col_idx = filterable_cols.index(cond["col"]) if cond["col"] in filterable_cols else 0
                edit_col = st.selectbox(
                    f"Col #{idx + 1}",
                    options=filterable_cols,
                    index=col_idx,
                    key=f"tab3_edit_col_{idx}",
                    label_visibility="collapsed",
                )

            edit_ops = get_operators_for_col(tableau_df, edit_col) if edit_col in tableau_df.columns else ["equals (==)"]
            op_idx = edit_ops.index(cond["op"]) if cond["op"] in edit_ops else 0

            with e2:
                edit_op = st.selectbox(
                    f"Op #{idx + 1}",
                    options=edit_ops,
                    index=op_idx,
                    key=f"tab3_edit_op_{idx}",
                    label_visibility="collapsed",
                )

            with e3:
                edit_val1 = st.text_input(
                    f"Val1 #{idx + 1}",
                    value=cond["val1"],
                    key=f"tab3_edit_val1_{idx}",
                    label_visibility="collapsed",
                )

            with e4:
                if edit_op == "between":
                    edit_val2 = st.text_input(
                        f"Val2 #{idx + 1}",
                        value=cond.get("val2", ""),
                        placeholder="Max",
                        key=f"tab3_edit_val2_{idx}",
                        label_visibility="collapsed",
                    )
                else:
                    edit_val2 = ""
                    st.text_input(
                        f"Val2_dis #{idx + 1}",
                        value="—",
                        disabled=True,
                        key=f"tab3_edit_val2_dis_{idx}",
                        label_visibility="collapsed",
                    )

            with e5:
                if st.button("Update", key=f"tab3_upd_btn_{idx}", width="stretch"):
                    err = validate_condition(tableau_df, edit_col, edit_op, edit_val1, edit_val2) if edit_col in tableau_df.columns else None
                    if err:
                        st.error(f"Rule #{idx + 1}: {err}")
                    else:
                        st.session_state["custom_conditions"][idx] = {
                            "col": edit_col,
                            "op": edit_op,
                            "val1": edit_val1.strip(),
                            "val2": edit_val2.strip(),
                        }
                        st.rerun()

            with e6:
                if st.button("Delete", key=f"tab3_del_btn_{idx}", width="stretch"):
                    st.session_state["custom_conditions"].pop(idx)
                    st.rerun()

        valid_conds = [c for c in st.session_state["custom_conditions"] if c["col"] in tableau_df.columns]
        if valid_conds and not tableau_df.empty:
            masks = [
                apply_single_condition(tableau_df, c["col"], c["op"], c["val1"], c.get("val2", ""))
                for c in valid_conds
            ]
            combined_mask = masks[0]
            for m in masks[1:]:
                combined_mask = (combined_mask & m) if match_mode == "ALL (AND)" else (combined_mask | m)
            tableau_df = tableau_df[combined_mask]

    st.session_state["tableau_active_slice"] = tableau_df.copy()

    st.divider()
    st.subheader("2. Tableau Rows & Columns Pivot Shelf")
    p1, p2 = st.columns(2)
    dimensions = [d for d in ["Region", "Product", "Age_Group", "Customer_Gender", "Quarter", "Year_Month"] if d in df.columns]

    with p1:
        row_dim = st.selectbox("Rows Dimension", options=dimensions or ["Region"], index=0, key="tab3_row_dim")
    with p2:
        col_options = ["None (Summary Table)"] + [d for d in dimensions if d != row_dim]
        col_dim = st.selectbox("Columns Dimension (Pivot)", options=col_options, index=min(1, len(col_options) - 1), key="tab3_col_dim")

    tk1, tk2, tk3, tk4 = st.columns(4)
    tk1.metric("Matching Sales ($)", f"${tableau_df['Sales'].sum():,.2f}" if not tableau_df.empty and "Sales" in tableau_df.columns else "$0.00")
    tk2.metric("Matching Orders", f"{len(tableau_df):,}")
    tk3.metric("Avg Sale ($)", f"${tableau_df['Sales'].mean():,.2f}" if not tableau_df.empty and "Sales" in tableau_df.columns else "$0.00")
    tk4.metric(
        "Avg Satisfaction",
        f"{tableau_df['Customer_Satisfaction'].mean():.2f} / 5.0" if not tableau_df.empty and "Customer_Satisfaction" in tableau_df.columns else "N/A",
    )

    if tableau_df.empty:
        st.warning("No rows match your custom conditions. Edit a rule above or click 'Clear All Conditions'.")
    else:
        st.subheader(f"3. Exact Numbers Matrix ({row_dim} × {col_dim}) — Click Any Row to Drill Down")

        if col_dim == "None (Summary Table)":
            matrix_df = (
                tableau_df.groupby(row_dim, as_index=False, observed=False)
                .agg(
                    Total_Sales=("Sales", "sum"),
                    Order_Count=("Sales", "count"),
                    Avg_Order_Value=("Sales", "mean"),
                    Median_Order=("Sales", "median"),
                    Avg_Satisfaction=("Customer_Satisfaction", "mean"),
                )
                .sort_values("Total_Sales", ascending=False)
            )
            grand_total = pd.DataFrame([{
                row_dim: "Grand Total",
                "Total_Sales": tableau_df["Sales"].sum(),
                "Order_Count": len(tableau_df),
                "Avg_Order_Value": tableau_df["Sales"].mean(),
                "Median_Order": tableau_df["Sales"].median(),
                "Avg_Satisfaction": tableau_df["Customer_Satisfaction"].mean(),
            }])
            display_matrix = pd.concat([matrix_df, grand_total], ignore_index=True)
        else:
            pivot_df = pd.pivot_table(
                tableau_df,
                values="Sales",
                index=row_dim,
                columns=col_dim,
                aggfunc="sum",
                fill_value=0.0,
                observed=False,
            )
            pivot_df["Row Total ($)"] = pivot_df.sum(axis=1)
            pivot_df["Order Count"] = tableau_df.groupby(row_dim, observed=False)["Sales"].count()
            pivot_df["Avg Satisfaction"] = tableau_df.groupby(row_dim, observed=False)["Customer_Satisfaction"].mean().round(2)
            pivot_df = pivot_df.reset_index()

            total_row = {row_dim: "Grand Total"}
            for c in pivot_df.columns:
                if c == row_dim:
                    continue
                elif c == "Avg Satisfaction":
                    total_row[c] = round(tableau_df["Customer_Satisfaction"].mean(), 2)
                else:
                    total_row[c] = pivot_df[c].sum()
            display_matrix = pd.concat([pivot_df, pd.DataFrame([total_row])], ignore_index=True)

        format_dict = {}
        for c in display_matrix.columns:
            if c == row_dim:
                continue
            elif "Count" in str(c):
                format_dict[c] = "{:,.0f}"
            elif "Satisfaction" in str(c):
                format_dict[c] = "{:.2f}"
            else:
                format_dict[c] = "${:,.2f}"

        matrix_event = st.dataframe(
            display_matrix.style.format(format_dict),
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="tab3_matrix_df",
        )

        dl_m1, dl_m2 = st.columns([3, 1])
        with dl_m2:
            st.download_button(
                label="Download Numbers Matrix CSV",
                data=display_matrix.to_csv(index=False).encode("utf-8"),
                file_name=f"tableau_matrix_{row_dim}.csv",
                mime="text/csv",
                width="stretch",
                key="tab3_dl_matrix",
            )

        st.divider()

        drill_df = tableau_df.copy()
        selected_rows = matrix_event.get("selection", {}).get("rows", []) if matrix_event else []
        active_label = "All Condition-Matched Orders"

        if selected_rows:
            picked_val = str(display_matrix.iloc[selected_rows[0]][row_dim])
            if picked_val != "Grand Total":
                drill_df = drill_df[drill_df[row_dim].astype(str) == picked_val]
                active_label = f"{row_dim} = {picked_val}"

        st.subheader(f"4. Individual Sales Records with Editable Comments ({active_label} — {len(drill_df):,} orders)")
        st.caption("Double-click any cell in the **Comments** column to type a note, tick **Select** for specific rows, and click **Save Comments**.")

        editor_df = drill_df.copy()
        editor_df.insert(0, "Select", False)
        data_columns = [c for c in editor_df.columns if c not in ("Select", "Comments")]

        edited_df = st.data_editor(
            editor_df,
            width="stretch",
            hide_index=True,
            disabled=data_columns,
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", help="Check rows to download only those specific sales", default=False),
                "Row_ID": st.column_config.TextColumn("Row ID"),
                "Sales": st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
                "Customer_Satisfaction": st.column_config.NumberColumn("Satisfaction", format="%.4f"),
                "Comments": st.column_config.TextColumn("Comments (Editable)", width="large", help="Double-click to add or edit notes for this row"),
            },
            key="tab3_sales_comments_editor",
        )

        btn_save_col, btn_filter_notes_col, btn_export_col = st.columns([1.2, 1.3, 1.5])

        with btn_save_col:
            if st.button("Save Comments", type="primary", key="tab3_save_comments", width="stretch"):
                updated_comments = load_saved_comments()
                for _, r in edited_df.iterrows():
                    rid = str(r["Row_ID"])
                    note = str(r["Comments"]).strip() if pd.notna(r["Comments"]) else ""
                    if note:
                        updated_comments[rid] = note
                    elif rid in updated_comments:
                        del updated_comments[rid]
                save_comments_to_disk(updated_comments)
                st.success(f"Saved! Total rows with comments across dataset: **{len(updated_comments)}**.")

        with btn_filter_notes_col:
            only_commented = st.checkbox("Export only rows with Comments", key="tab3_only_commented")

        selected_mask = edited_df["Select"] == True
        if selected_mask.any():
            export_df = edited_df[selected_mask].drop(columns=["Select"])
            dl_label = f"Download Selected {len(export_df)} Row(s) with Comments CSV"
            dl_filename = "selected_sales_with_comments.csv"
        else:
            export_df = edited_df.drop(columns=["Select"])
            if only_commented:
                export_df = export_df[export_df["Comments"].astype(str).str.strip() != ""]
            dl_label = f"Download {len(export_df):,} Row(s) with Comments CSV"
            dl_filename = "tableau_sales_with_comments.csv"

        with btn_export_col:
            st.download_button(
                label=dl_label,
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=dl_filename,
                mime="text/csv",
                width="stretch",
                key="tab3_dl_sales",
            )

# ==============================================================================
# TAB 4: LLM EVALUATION & QUALITY JUDGE
# ==============================================================================
with tab4:
    st.subheader("RAG & LLM Evaluation Suite (4-Pillar Quality & Fairness Guardrails)")

    if getattr(rag_bot, "llm", None) is not None:
        st.success(f"**Active Judge Engine: `{rag_bot.model_name}`** — LLM-assisted quality evaluation active.")
    else:
        st.info(
            "**Active Judge Engine: Local Offline Heuristic + Guardrail Rules** — "
            "Add an API key in the sidebar or `.env` to enable LLM-assisted evaluation."
        )

    st.markdown("### 1. Live Auto-Evaluation Log (Connected to Tab 2 Chat)")
    if st.session_state["live_eval_history"]:
        live_df = pd.DataFrame(st.session_state["live_eval_history"])
        st.dataframe(live_df, width="stretch", hide_index=True)
        if st.button(
            "Clear Live Chat Evaluation Log",
            help="Clears the auto-logged 4-pillar evaluation rows collected from Tab 2 during this session.",
        ):
            st.session_state["live_eval_history"] = []
            st.rerun()
    else:
        st.info("Ask any question in **Tab 2 (Ask InsightForge AI)** — its 4-pillar quality & fairness scores will appear here automatically.")

    st.divider()
    st.markdown("### 2. Standalone Benchmark & Custom Stress-Test Suite")
    st.caption(
        "Scores **Correctness**, **Faithfulness**, **Completeness**, and **Fairness** (1–5 scale) "
        "without saving test runs to your personal Tab 2 chat history."
    )

    benchmark_cases = [
        {
            "question": "What are the top selling product and best performing region in the dataset?",
            "ground_truth": "Widget A is the top selling product by both revenue and volume, and West is the highest-performing region while East lags.",
        },
        {
            "question": "Compare sales performance between Female and Male customers objectively.",
            "ground_truth": f"Female and Male customer sales are balanced across the dataset with total revenue of ${df['Sales'].sum():,.2f} and average satisfaction of {df['Customer_Satisfaction'].mean():.2f} out of 5.",
        },
        {
            "question": "How does the managerial approach differ from the technical approach in Business Intelligence?",
            "ground_truth": "The managerial approach views BI as integrating internal and external data to support management decision-making, whereas the technical approach focuses on tools and infrastructure.",
        },
    ]

    if st.button(
        "Run Automated Benchmark Evaluation (3 Test Cases)",
        type="primary",
        help="Runs 3 standard sales, demographic fairness, and BI concept questions through Tab 2's RAG pipeline and grades all 4 pillars (1–5).",
    ):
        eval_rows = []
        progress = st.progress(0)
        for idx, case in enumerate(benchmark_cases):
            res = rag_bot.query(
                case["question"],
                source_mode="Only Built-in CSV & PDFs",
                use_wiki_fallback=False,
            )
            score_row = run_safe_eval(
                evaluator,
                question=case["question"],
                answer=res["answer"],
                context=f"{res.get('scope_label', '')}\n{case['ground_truth']}",
            )
            score_row["Generated Answer"] = res["answer"]
            eval_rows.append(score_row)
            progress.progress((idx + 1) / len(benchmark_cases))

        eval_df = pd.DataFrame(eval_rows)
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Avg Correctness", f"{eval_df['Correctness (1-5)'].mean():.2f} / 5.0")
        e2.metric("Avg Faithfulness", f"{eval_df['Faithfulness (1-5)'].mean():.2f} / 5.0")
        e3.metric("Avg Completeness", f"{eval_df['Completeness (1-5)'].mean():.2f} / 5.0")
        e4.metric("Avg Fairness", f"{eval_df['Fairness (1-5)'].mean():.2f} / 5.0")
        e5.metric("Overall Quality", f"{eval_df['Average Score'].mean():.2f} / 5.0")

        st.dataframe(eval_df, width="stretch", hide_index=True)
        st.download_button(
            label="Download Evaluation Report CSV",
            data=eval_df.to_csv(index=False).encode("utf-8"),
            file_name="insightforge_llm_eval_report.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("**Evaluate a Custom Question (With or Without Ground Truth):**")
    custom_q = st.text_input(
        "1. Test Question (Required):",
        placeholder="e.g., Which customer age group has the highest average spend and how do regions compare?",
        help="Enter the exact question you want the RAG bot to answer and get graded on.",
    )
    custom_gt = st.text_area(
        "2. Expected Ground Truth Answer (Optional — leave blank for automatic Reference-Free verification):",
        placeholder=(
            "e.g., The 18-25 age group records the highest average order value, and West leads overall regional sales.\n"
            "• Leave BLANK to automatically verify numbers, completeness & fairness directly against your CSV/PDFs.\n"
            "• Fill IN to compare the bot's answer against your exact expected answer."
        ),
        height=100,
    )

    if st.button(
        "Evaluate Custom Test Case (4-Pillar 1–5 Score)",
        help="Step 1: Generates an answer via Tab 2 RAG. Step 2: Grades Correctness, Faithfulness, Completeness & Fairness.",
    ):
        if not custom_q.strip():
            st.warning("Enter a Test Question in Box 1 first.")
        else:
            with st.spinner("Generating RAG response and scoring 4-pillar quality & fairness..."):
                res = rag_bot.query(custom_q.strip(), use_wiki_fallback=True)
                scored = run_safe_eval(
                    evaluator,
                    question=custom_q.strip(),
                    answer=res["answer"],
                    context=f"{res.get('scope_label', '')}\n{custom_gt.strip()}",
                )
                st.dataframe(pd.DataFrame([scored]), width="stretch", hide_index=True)
                st.markdown("**Generated RAG Response:**")
                st.info(res["answer"])