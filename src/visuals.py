import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class SalesVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            self.df["Year"] = self.df["Date"].dt.year.astype(str)
            if "Year_Month" not in self.df.columns:
                self.df["Year_Month"] = self.df["Date"].dt.to_period("M").astype(str)

    def plot_monthly_sales_trend(self):
        monthly = (
            self.df.groupby("Year_Month", as_index=False, observed=False)
            .agg(Total_Sales=("Sales", "sum"), Orders=("Sales", "count"))
            .sort_values("Year_Month")
        )
        fig = px.line(
            monthly,
            x="Year_Month",
            y="Total_Sales",
            markers=True,
            title="Monthly Sales Revenue Trend ($)",
            labels={"Year_Month": "Month", "Total_Sales": "Total Sales ($)"},
        )
        fig.update_layout(hovermode="x unified", height=380)
        return fig

    def plot_product_and_region(self):
        grouped = (
            self.df.groupby(["Region", "Product"], as_index=False, observed=False)["Sales"]
            .sum()
        )
        fig = px.bar(
            grouped,
            x="Region",
            y="Sales",
            color="Product",
            barmode="group",
            title="Sales by Region & Product ($)",
            labels={"Sales": "Total Sales ($)"},
        )
        fig.update_layout(height=380)
        return fig

    def plot_customer_demographics(self):
        demo = (
            self.df.groupby(["Age_Group", "Customer_Gender"], as_index=False, observed=False)["Sales"]
            .mean()
        )
        fig = px.bar(
            demo,
            x="Age_Group",
            y="Sales",
            color="Customer_Gender",
            barmode="group",
            title="Average Order Value by Age Group & Gender ($)",
            labels={"Sales": "Avg Order Value ($)", "Age_Group": "Age Group"},
        )
        fig.update_layout(height=380)
        return fig

    def plot_yearly_performance(self):
        yearly = (
            self.df.groupby("Year", as_index=False, observed=False)
            .agg(
                Total_Sales=("Sales", "sum"),
                Order_Count=("Sales", "count"),
                Avg_Satisfaction=("Customer_Satisfaction", "mean"),
            )
            .sort_values("Year")
        )
        fig = px.bar(
            yearly,
            x="Year",
            y="Total_Sales",
            text="Order_Count",
            color="Avg_Satisfaction",
            color_continuous_scale="Blues",
            title="Year-over-Year Total Sales ($), Order Volume & Satisfaction",
            labels={
                "Total_Sales": "Total Sales ($)",
                "Avg_Satisfaction": "Avg Rating",
            },
        )
        fig.update_traces(texttemplate="%{text:,} orders", textposition="outside")
        fig.update_layout(height=380)
        return fig

    def plot_correlation_heatmap(self):
        num_cols = [c for c in ["Sales", "Customer_Age", "Customer_Satisfaction"] if c in self.df.columns]
        corr = self.df[num_cols].corr().round(3)
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                text=corr.values,
                texttemplate="%{text:.3f}",
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
            )
        )
        fig.update_layout(
            title="Numeric Feature Correlation Matrix (Sales, Age & Satisfaction)",
            height=380,
        )
        return fig