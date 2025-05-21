from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import to_rgb

st.set_page_config(layout="wide")


# Function to load data from multiple CSV files and collect all unique metric names
@st.cache_data
def load_data(uploaded_files: list) -> Tuple[pd.DataFrame, set[str]]:
    dataframes = []
    all_columns = set()
    for file in uploaded_files:
        try:
            if file.name.endswith(".csv"):
                file_data = pd.read_csv(file)
            else:
                file_data = pd.read_parquet(file)

            # Check if 'target' column exists
            if "target" not in file_data.columns:
                st.error(f"File {file.name} needs to be a metrics file.")
                continue

        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

        file_data = file_data.loc[:, ~file_data.columns.str.contains("^Unnamed")]
        file_data["Source"] = file.name  # Add a column to track the source file
        all_columns.update(file_data.columns)
        dataframes.append(file_data)

    if dataframes:
        return pd.concat(dataframes, ignore_index=True), all_columns
    else:
        st.warning(
            "No valid files were uploaded. Please upload files containing the 'target' column."
        )
        return pd.DataFrame(), set()


# Function to apply color gradient similar to Excel conditional formatting
def apply_gradient(
    val: float, min_val: float, max_val: float, min_color: str, max_color: str
) -> str:
    min_rgb = np.array(to_rgb(min_color)) * 255
    max_rgb = np.array(to_rgb(max_color)) * 255
    if pd.isna(val):
        return ""
    norm_val = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    color = (min_rgb * (1 - norm_val)) + (max_rgb * norm_val)
    red, green, blue = color.astype(int)
    return f"background-color: rgb({red}, {green}, {blue})"


def main() -> None:
    # Main function to run the app
    st.markdown(
        "<h2 style='text-align: center; font-weight: normal;'>Sensor Metrics Comparison</h2>",
        unsafe_allow_html=True,
    )

    # Sidebar options
    st.sidebar.header("Upload Data")
    uploaded_files = st.sidebar.file_uploader(
        "Upload csv or parquet files", accept_multiple_files=True, type=["csv", "parquet"]
    )

    if uploaded_files:
        # Load data and get all unique columns
        data, all_columns = load_data(uploaded_files)
        all_columns = list(all_columns - {"target", "Source"})

        if not all_columns:
            st.warning("No metrics available for selection.")

        # Select a single metric
        st.sidebar.header("Metric and Color Selection")
        selected_metric = st.sidebar.radio("Select Metric to Display", all_columns)

        if selected_metric:
            # Filter data to include only the files that have the selected metric
            data = data.dropna(subset=[selected_metric])

            # Color selection
            min_color = st.sidebar.color_picker("Color for minimum value", "#66c266")
            max_color = st.sidebar.color_picker("Color for maximum value", "#ff6666")

            # Pivot data
            pivot_table = data.pivot_table(index="target", columns="Source", values=selected_metric)

            if pivot_table.empty:
                st.warning("The pivot table is empty. Please check your data.")
                return

            # Compute min and max for selected metric
            min_val = pivot_table.min().min()
            max_val = pivot_table.max().max()

            # Apply color formatting
            styled_df = pivot_table.style.applymap(
                lambda x: apply_gradient(x, min_val, max_val, min_color, max_color)
            )
            styled_df = styled_df.format(precision=4)
            st.dataframe(styled_df, height=800, width=1500, use_container_width=True)
        else:
            st.warning("Please select a metric to display.")
    else:
        st.write("")
        st.markdown(
            """<p style='text-align: center;'>Please select a load case metrics CSV/Parquet'
             'file to proceed</p>""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
