import pandas as pd
from langchain_core.documents import Document
from src.config import CSV_PATH


class SalesDataEngine:
    """Handles Pandas statistical aggregations and custom retrieval for sales_data.csv."""

    def __init__(self, csv_path: str = CSV_PATH):
        self.df = pd.read_csv(csv_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df["Year_Month"] = self.df["Date"].dt.to_period("M").astype(str)
        self.df["Quarter"] = self.df["Date"].dt.to_period("Q").astype(str)

        # Create customer age brackets for segmentation
        bins = [0, 25, 35, 50, 100]
        labels = ["18-25 (Gen Z)", "26-35 (Young Adult)", "36-50 (Mid-Career)", "51+ (Senior)"]
        self.df["Age_Group"] = pd.cut(self.df["Customer_Age"], bins=bins, labels=labels)

    def get_overall_stats(self) -> str:
        """4. Statistical measures (mean, median, min, max, std) for Sales & Satisfaction."""
        sales = self.df["Sales"]
        sat = self.df["Customer_Satisfaction"]
        return (
            "=== OVERALL STATISTICAL SUMMARY (2,500 Transactions) ===\n"
            f"- Total Revenue: ${sales.sum():,.2f}\n"
            f"- Sales Mean: ${sales.mean():,.2f} | Median: ${sales.median():,.2f} | Std Dev: ${sales.std():,.2f}\n"
            f"- Sales Min: ${sales.min():,.2f} | Max: ${sales.max():,.2f}\n"
            f"- Customer Satisfaction Mean: {sat.mean():.2f}/5.0 | Median: {sat.median():.2f}/5.0 | Std Dev: {sat.std():.2f}\n"
        )

    def get_time_period_summary(self) -> str:
        """1. Sales performance by time period (Quarterly & Top/Bottom Months)."""
        quarterly = self.df.groupby("Quarter")["Sales"].agg(["count", "sum", "mean", "median"]).round(2)
        monthly = self.df.groupby("Year_Month")["Sales"].sum()
        best_month = f"{monthly.idxmax()} (${monthly.max():,.2f})"
        worst_month = f"{monthly.idxmin()} (${monthly.min():,.2f})"

        return (
            "=== SALES PERFORMANCE BY TIME PERIOD ===\n"
            f"- Best Performing Month: {best_month}\n"
            f"- Lowest Performing Month: {worst_month}\n"
            f"- Quarterly Breakdown:\n{quarterly.to_string()}\n"
        )

    def get_product_and_region_summary(self) -> str:
        """2. Product and regional analysis."""
        prod = (
            self.df.groupby("Product")
            .agg(
                Total_Sales=("Sales", "sum"),
                Avg_Sales=("Sales", "mean"),
                Median_Sales=("Sales", "median"),
                Std_Sales=("Sales", "std"),
                Avg_Satisfaction=("Customer_Satisfaction", "mean"),
            )
            .round(2)
        )
        reg = (
            self.df.groupby("Region")
            .agg(
                Total_Sales=("Sales", "sum"),
                Avg_Sales=("Sales", "mean"),
                Median_Sales=("Sales", "median"),
                Avg_Satisfaction=("Customer_Satisfaction", "mean"),
            )
            .round(2)
        )
        return (
            "=== PRODUCT PERFORMANCE ANALYSIS ===\n"
            f"{prod.to_string()}\n\n"
            "=== REGIONAL PERFORMANCE ANALYSIS ===\n"
            f"{reg.to_string()}\n"
        )

    def get_customer_segmentation_summary(self) -> str:
        """3. Customer segmentation by demographics (Age Group & Gender)."""
        age_seg = (
            self.df.groupby("Age_Group", observed=False)
            .agg(
                Customers=("Sales", "count"),
                Total_Sales=("Sales", "sum"),
                Avg_Sales=("Sales", "mean"),
                Avg_Satisfaction=("Customer_Satisfaction", "mean"),
            )
            .round(2)
        )
        gender_seg = (
            self.df.groupby("Customer_Gender")
            .agg(
                Customers=("Sales", "count"),
                Total_Sales=("Sales", "sum"),
                Avg_Sales=("Sales", "mean"),
                Avg_Satisfaction=("Customer_Satisfaction", "mean"),
            )
            .round(2)
        )
        return (
            "=== CUSTOMER SEGMENTATION BY AGE GROUP ===\n"
            f"{age_seg.to_string()}\n\n"
            "=== CUSTOMER SEGMENTATION BY GENDER ===\n"
            f"{gender_seg.to_string()}\n"
        )

    def retrieve_relevant_stats(self, query: str) -> list[Document]:
        """Custom retriever that routes user questions to targeted Pandas statistics."""
        q = query.lower()
        sections = [self.get_overall_stats()]

        if any(w in q for w in ["time", "month", "quarter", "trend", "year", "date", "season", "growth"]):
            sections.append(self.get_time_period_summary())
        if any(w in q for w in ["product", "widget", "region", "north", "south", "east", "west", "area"]):
            sections.append(self.get_product_and_region_summary())
        if any(w in q for w in ["customer", "age", "gender", "male", "female", "demographic", "segment", "satisfaction"]):
            sections.append(self.get_customer_segmentation_summary())

        # If query is broad (e.g., "Give me a full business summary"), include all sections
        if len(sections) == 1:
            sections.extend([
                self.get_time_period_summary(),
                self.get_product_and_region_summary(),
                self.get_customer_segmentation_summary(),
            ])

        combined_text = "\n\n".join(sections)
        return [Document(page_content=combined_text, metadata={"source": "sales_data.csv (Pandas Engine)"})]


if __name__ == "__main__":
    engine = SalesDataEngine()
    docs = engine.retrieve_relevant_stats("Which product and region perform best?")
    print(docs[0].page_content)