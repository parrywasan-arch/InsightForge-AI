import json
import os
import pandas as pd
import streamlit as st
from src.data_engine import SalesDataEngine

st.set_page_config(
    page_title="InsightForge | Tableau Crosstab, Conditions & Row Comments",
    layout="wide",
)

st.title("InsightForge — Tableau Crosstab, Condition Builder & Row Comments Studio")
st.caption("Add, edit, or delete custom conditions, inspect exact pivot numbers, add row comments, and download as CSV.")

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


@st.cache_data
def load_enriched_data():
    engine = SalesDataEngine()
    df = engine.df.copy()
    df.insert(0, "Row_ID", [f"ORD-{i + 1:04d}" for i in range(len(df))])

    if "Age_Group" not in df.columns and "Customer_Age" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Customer_Age"],
            bins=[17, 25, 35, 50, 120],
            labels=["18-25 (Gen Z)", "26-35 (Young Adult)", "36-50 (Mid-Career)", "51+ (Senior)"],
        )
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year_Month"] = df["Date"].dt.to_period("M").astype(str)
        df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    return df


def apply_single_condition(frame: pd.DataFrame, col: str, op: str, val1: str, val2: str = "") -> pd.Series:
    """Evaluates a user-defined condition on any numeric, text, or date column."""
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
    """Returns None if valid, otherwise an error message."""
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


df = load_enriched_data()
saved_comments = load_saved_comments()
df["Comments"] = df["Row_ID"].map(lambda rid: saved_comments.get(rid, ""))

if "custom_conditions" not in st.session_state:
    st.session_state["custom_conditions"] = []

# --- 1. Custom Condition Builder (Add, Edit Inline, or Delete Individual Rules) ---
st.subheader("1. Custom Condition Filter Builder")

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
    cond_col = st.selectbox("Column", options=filterable_cols, index=0, key="new_cond_col")

op_options = get_operators_for_col(df, cond_col)

with c_col2:
    cond_op = st.selectbox("Condition", options=op_options, key="new_cond_op")

with c_col3:
    is_num_col = pd.api.types.is_numeric_dtype(df[cond_col])
    placeholder_txt = "e.g., 500 or 2.5" if is_num_col else "e.g., East, 2024-07, or refund"
    cond_val1 = st.text_input("Value", placeholder=placeholder_txt, key="new_cond_val1")

with c_col4:
    if cond_op == "between":
        cond_val2 = st.text_input("Max Value", placeholder="e.g., 1000", key="new_cond_val2")
    else:
        cond_val2 = ""
        st.text_input("Max Value (for 'between')", value="—", disabled=True, key="new_cond_val2_disabled")

with c_col5:
    st.write("")
    st.write("")
    add_clicked = st.button("Add Condition", use_container_width=True)

if add_clicked:
    err = validate_condition(df, cond_col, cond_op, cond_val1, cond_val2)
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

# Active Conditions Editor (Modify or Delete Any Condition Directly)
filtered_df = df.copy()

if st.session_state["custom_conditions"]:
    st.markdown("**Active Conditions (Edit any field directly and click Update, or click Delete):**")

    top_bar1, top_bar2 = st.columns([3, 1])
    with top_bar1:
        match_mode = st.radio(
            "Combine Conditions With:",
            options=["ALL (AND)", "ANY (OR)"],
            horizontal=True,
            key="cond_match_mode",
        )
    with top_bar2:
        if st.button("Clear All Conditions", use_container_width=True):
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
                key=f"edit_col_{idx}",
                label_visibility="collapsed",
            )

        edit_ops = get_operators_for_col(df, edit_col)
        op_idx = edit_ops.index(cond["op"]) if cond["op"] in edit_ops else 0

        with e2:
            edit_op = st.selectbox(
                f"Op #{idx + 1}",
                options=edit_ops,
                index=op_idx,
                key=f"edit_op_{idx}",
                label_visibility="collapsed",
            )

        with e3:
            edit_val1 = st.text_input(
                f"Val1 #{idx + 1}",
                value=cond["val1"],
                key=f"edit_val1_{idx}",
                label_visibility="collapsed",
            )

        with e4:
            if edit_op == "between":
                edit_val2 = st.text_input(
                    f"Val2 #{idx + 1}",
                    value=cond.get("val2", ""),
                    placeholder="Max",
                    key=f"edit_val2_{idx}",
                    label_visibility="collapsed",
                )
            else:
                edit_val2 = ""
                st.text_input(
                    f"Val2_dis #{idx + 1}",
                    value="—",
                    disabled=True,
                    key=f"edit_val2_dis_{idx}",
                    label_visibility="collapsed",
                )

        with e5:
            if st.button("Update", key=f"upd_btn_{idx}", use_container_width=True):
                err = validate_condition(df, edit_col, edit_op, edit_val1, edit_val2)
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
            if st.button("Delete", key=f"del_btn_{idx}", use_container_width=True):
                st.session_state["custom_conditions"].pop(idx)
                st.rerun()

    # Apply all validated active conditions
    masks = [
        apply_single_condition(filtered_df, c["col"], c["op"], c["val1"], c.get("val2", ""))
        for c in st.session_state["custom_conditions"]
    ]
    combined_mask = masks[0]
    for m in masks[1:]:
        combined_mask = (combined_mask & m) if match_mode == "ALL (AND)" else (combined_mask | m)
    filtered_df = filtered_df[combined_mask]

# --- 2. Tableau Rows / Columns Pivot Configuration ---
st.divider()
st.subheader("2. Tableau Rows & Columns Pivot Shelf")
p1, p2 = st.columns(2)
dimensions = ["Region", "Product", "Age_Group", "Customer_Gender", "Quarter", "Year_Month"]

with p1:
    row_dim = st.selectbox("Rows Dimension", options=dimensions, index=0)
with p2:
    col_options = ["None (Summary Table)"] + [d for d in dimensions if d != row_dim]
    col_dim = st.selectbox("Columns Dimension (Pivot)", options=col_options, index=1)

# KPI Strip
k1, k2, k3, k4 = st.columns(4)
k1.metric("Matching Sales ($)", f"${filtered_df['Sales'].sum():,.2f}")
k2.metric("Matching Orders", f"{len(filtered_df):,}")
k3.metric("Avg Sale ($)", f"${filtered_df['Sales'].mean():,.2f}" if not filtered_df.empty else "$0.00")
k4.metric("Avg Satisfaction", f"{filtered_df['Customer_Satisfaction'].mean():.2f} / 5.0" if not filtered_df.empty else "N/A")

if filtered_df.empty:
    st.warning("No rows match your custom conditions. Edit a rule above or click 'Clear All Conditions'.")
    st.stop()

# --- 3. Exact Numbers Matrix (Click Row to Drill Down) ---
st.subheader(f"3. Exact Numbers Matrix ({row_dim} × {col_dim}) — Click Any Row to Drill Down")

if col_dim == "None (Summary Table)":
    matrix_df = (
        filtered_df.groupby(row_dim, as_index=False, observed=False)
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
        "Total_Sales": filtered_df["Sales"].sum(),
        "Order_Count": len(filtered_df),
        "Avg_Order_Value": filtered_df["Sales"].mean(),
        "Median_Order": filtered_df["Sales"].median(),
        "Avg_Satisfaction": filtered_df["Customer_Satisfaction"].mean(),
    }])
    display_matrix = pd.concat([matrix_df, grand_total], ignore_index=True)
else:
    pivot_df = pd.pivot_table(
        filtered_df,
        values="Sales",
        index=row_dim,
        columns=col_dim,
        aggfunc="sum",
        fill_value=0.0,
        observed=False,
    )
    pivot_df["Row Total ($)"] = pivot_df.sum(axis=1)
    pivot_df["Order Count"] = filtered_df.groupby(row_dim, observed=False)["Sales"].count()
    pivot_df["Avg Satisfaction"] = filtered_df.groupby(row_dim, observed=False)["Customer_Satisfaction"].mean().round(2)
    pivot_df = pivot_df.reset_index()

    total_row = {row_dim: "Grand Total"}
    for c in pivot_df.columns:
        if c == row_dim:
            continue
        elif c == "Avg Satisfaction":
            total_row[c] = round(filtered_df["Customer_Satisfaction"].mean(), 2)
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
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

dl_m1, dl_m2 = st.columns([3, 1])
with dl_m2:
    st.download_button(
        label="Download Numbers Matrix CSV",
        data=display_matrix.to_csv(index=False).encode("utf-8"),
        file_name=f"tableau_matrix_{row_dim}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# --- 4. Editable Individual Sales Table (Add Comments, Save & Download Rows) ---
drill_df = filtered_df.copy()
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
    use_container_width=True,
    hide_index=True,
    disabled=data_columns,
    column_config={
        "Select": st.column_config.CheckboxColumn("Select", help="Check rows to download only those specific sales", default=False),
        "Row_ID": st.column_config.TextColumn("Row ID"),
        "Sales": st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
        "Customer_Satisfaction": st.column_config.NumberColumn("Satisfaction", format="%.4f"),
        "Comments": st.column_config.TextColumn("Comments (Editable)", width="large", help="Double-click to add or edit notes for this row"),
    },
    key="sales_comments_editor",
)

btn_save_col, btn_filter_notes_col, btn_export_col = st.columns([1.2, 1.3, 1.5])

with btn_save_col:
    if st.button("Save Comments", type="primary", use_container_width=True):
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
    only_commented = st.checkbox("Export only rows with Comments")

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
        use_container_width=True,
    )