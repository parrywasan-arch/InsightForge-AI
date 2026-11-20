import streamlit as st
from src.config import USE_AZURE
from src.rag_engine import InsightForgeRAG
from src.visuals import SalesVisualizer

# 1. Page configuration (pure text, no image/icon dependencies)
st.set_page_config(
    page_title="InsightForge AI | Business Intelligence",
    layout="wide",
)

st.title("InsightForge AI — Enterprise Business Intelligence Assistant")
st.caption(
    f"Mode: **{'Azure OpenAI (Lab)' if USE_AZURE else 'Local Offline Test Mode'}** | "
    "Powered by Pandas, FAISS RAG & Plotly"
)


# 2. Initialize & cache the RAG engine so PDFs aren't re-indexed on every click
@st.cache_resource
def load_engine():
    return InsightForgeRAG()


rag_bot = load_engine()
df = rag_bot.data_engine.df
viz = SalesVisualizer(df)

# 3. Top KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${df['Sales'].sum():,.2f}")
col2.metric("Average Order Value", f"${df['Sales'].mean():,.2f}")
col3.metric("Median Order Value", f"${df['Sales'].median():,.2f}")
col4.metric("Avg Customer Satisfaction", f"{df['Customer_Satisfaction'].mean():.2f} / 5.0")

st.divider()

# 4. Two-Tab Layout: Visual Dashboard & AI Analyst Chat
tab1, tab2 = st.tabs(["Interactive BI Dashboard", "Ask InsightForge AI"])

with tab1:
    st.subheader("Sales Trends, Regional Breakdown & Customer Demographics")
    st.plotly_chart(viz.plot_monthly_sales_trend(), use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(viz.plot_product_and_region(), use_container_width=True)
    with chart_col2:
        st.plotly_chart(viz.plot_customer_demographics(), use_container_width=True)

with tab2:
    st.subheader("Chat with Your Sales Data & BI Research Documents")

    # Initialize chat message history in Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box
    if prompt := st.chat_input("Ask about sales trends, products, regions, or BI strategies..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing sales statistics & BI literature..."):
                result = rag_bot.ask(prompt)
                st.markdown(result["answer"])

                with st.expander("View Retrieved Sales Stats & PDF Sources"):
                    st.text(result["stats_used"])
                    st.write("**PDF Sources Matched:**", list(set(result["pdf_sources"])))

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})