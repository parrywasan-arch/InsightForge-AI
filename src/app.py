import pandas as pd
import streamlit as st
from src.config import USE_AZURE
from src.rag_engine import InsightForgeRAG
from src.visuals import SalesVisualizer

st.set_page_config(
    page_title="InsightForge AI | Business Intelligence",
    layout="wide",
)

st.title("InsightForge AI — Enterprise Business Intelligence Assistant")
st.caption(
    f"Mode: **{'Azure OpenAI (Lab)' if USE_AZURE else 'Local Offline Test Mode'}** | "
    "Powered by Pandas, FAISS RAG & Plotly"
)


@st.cache_resource
def load_engine():
    return InsightForgeRAG()


rag_bot = load_engine()
df = rag_bot.data_engine.df.copy()

if "Age_Group" not in df.columns and "Customer_Age" in df.columns:
    df["Age_Group"] = pd.cut(
        df["Customer_Age"],
        bins=[17, 25, 35, 50, 120],
        labels=["18-25 (Gen Z)", "26-35 (Young Adult)", "36-50 (Mid-Career)", "51+ (Senior)"],
    )

viz = SalesVisualizer(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${df['Sales'].sum():,.2f}")
col2.metric("Average Order Value", f"${df['Sales'].mean():,.2f}")
col3.metric("Median Order Value", f"${df['Sales'].median():,.2f}")
col4.metric("Avg Customer Satisfaction", f"{df['Customer_Satisfaction'].mean():.2f} / 5.0")

st.divider()

tab1, tab2 = st.tabs(["Interactive BI Dashboard", "Ask InsightForge AI"])

with tab1:
    st.subheader("Search & Filter Sales by Age, Gender, Region & Product")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        age_options = ["All"] + [str(x) for x in df["Age_Group"].dropna().unique().tolist()]
        selected_age = st.selectbox("Select Age Group", options=age_options)

    with f_col2:
        gender_options = ["All"] + sorted(df["Customer_Gender"].dropna().unique().tolist())
        selected_gender = st.selectbox("Select Gender", options=gender_options)

    with f_col3:
        region_options = ["All"] + sorted(df["Region"].dropna().unique().tolist())
        selected_region = st.selectbox("Select Region", options=region_options)

    with f_col4:
        product_options = ["All"] + sorted(df["Product"].dropna().unique().tolist())
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
        search_clicked = st.button("Search Filtered Sales", use_container_width=True)

    with btn_reset_col:
        st.write("")
        st.write("")
        reset_clicked = st.button("Reset Filters", use_container_width=True)

    if reset_clicked:
        st.rerun()

    filtered_df = df.copy()
    if selected_age != "All":
        filtered_df = filtered_df[filtered_df["Age_Group"].astype(str) == selected_age]
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df["Customer_Gender"] == selected_gender]
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]
    if selected_product != "All":
        filtered_df = filtered_df[filtered_df["Product"] == selected_product]
    if keyword_query:
        mask = filtered_df.astype(str).apply(
            lambda col: col.str.contains(keyword_query, case=False, na=False)
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
            st.warning("No sales records match your selected Age, Gender, Region, or Product filters.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Filtered Sales Total", f"${filtered_df['Sales'].sum():,.2f}")
            m2.metric("Matching Orders", f"{len(filtered_df):,}")
            m3.metric("Filtered Avg Order", f"${filtered_df['Sales'].mean():,.2f}")
            m4.metric("Filtered Satisfaction", f"{filtered_df['Customer_Satisfaction'].mean():.2f} / 5.0")

            with st.expander("View Matching Sales Records Table", expanded=True):
                st.dataframe(filtered_df.head(50), use_container_width=True)

    st.divider()

    st.subheader("Sales Trends, Regional Breakdown & Customer Demographics")
    st.plotly_chart(viz.plot_monthly_sales_trend(), use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(viz.plot_product_and_region(), use_container_width=True)
    with chart_col2:
        st.plotly_chart(viz.plot_customer_demographics(), use_container_width=True)

with tab2:
    st.subheader("Personalized BI Assistant (User-Isolated Memory)")

    name_col, btn_col = st.columns([3, 1])
    with name_col:
        user_full_name = st.text_input(
            "Enter your Full Name to identify your session memory:",
            placeholder="e.g., Parminder Wasan",
        ).strip()

    if not user_full_name:
        st.info("Please enter your full name above to start chatting. Your question history is stored separately under your name.")
    else:
        past_records = rag_bot.get_user_history(user_full_name)

        with btn_col:
            st.write("")
            st.write("")
            if st.button("Clear My Memory", use_container_width=True):
                rag_bot.clear_user_memory(user_full_name)
                st.rerun()

        if past_records:
            st.success(f"Welcome back, **{user_full_name}**! Loaded {len(past_records)} previous question(s) from your profile.")
        else:
            st.info(f"Hello, **{user_full_name}**! Starting a fresh memory profile for you.")

        st.markdown("**Quick Prompts:**")
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        quick_prompt = None
        if q_col1.button("Sales by Age & Gender", use_container_width=True):
            quick_prompt = "Show total sales and average spend by customer age group and gender."
        if q_col2.button("Top Products & Regions", use_container_width=True):
            quick_prompt = "Which products and regions generate the highest sales revenue?"
        if q_col3.button("Walmart Doc Summary", use_container_width=True):
            quick_prompt = "Give me key findings only from the Walmart document."
        if q_col4.button("Next (Continue Answer)", use_container_width=True):
            quick_prompt = "next"

        for item in past_records:
            with st.chat_message("user"):
                st.markdown(item["question"])
            with st.chat_message("assistant"):
                st.markdown(item["answer"])

        typed_prompt = st.chat_input(f"Ask a question as {user_full_name} (or type 'next' to continue)...")
        active_prompt = quick_prompt or typed_prompt

        if active_prompt:
            with st.chat_message("user"):
                st.markdown(active_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Checking your profile memory & analyzing data..."):
                    result = rag_bot.ask(active_prompt, full_name=user_full_name)
                    st.markdown(result["answer"])

                    with st.expander("View Raw Statistical Context & Source Breakdown Table"):
                        if result.get("source_details"):
                            st.dataframe(pd.DataFrame(result["source_details"]), use_container_width=True)
                        st.text(result.get("stats_used", ""))