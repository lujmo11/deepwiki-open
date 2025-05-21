from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score


def calculate_general_metrics(
    actuals: np.array,
    predictions: np.array,
) -> Dict[str, float]:
    errors_all = predictions - actuals
    errors_all_norm = errors_all / actuals

    # Actuals can be 0 - convert inf to nan
    errors_all_norm[np.isinf(errors_all_norm)] = np.nan

    e_mean_norm = np.nanmean(errors_all_norm)
    e_std_norm = np.nanstd(errors_all_norm)
    mae = mean_absolute_error(predictions, actuals)
    mae_norm = np.abs(mae / np.nanmean(actuals))
    mane = np.nanmean(np.abs(errors_all_norm))
    r2 = r2_score(actuals, predictions)
    e_max_norm = np.max(np.abs(errors_all_norm))

    metrics = {
        "e_mean_norm": e_mean_norm,
        "e_std_norm": e_std_norm,
        "e_max_norm": e_max_norm,
        "mae": mae,
        "mae_norm": mae_norm,
        "mane": mane,
        "r2": r2,
    }

    return metrics


@st.cache_data
def load_data(file):  # noqa
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".parquet"):
        return pd.read_parquet(file)
    else:
        st.error("Unsupported file type.")
        return None


def compute_metric_matrix(
    df: pd.DataFrame, actual_col: str, pred_col: str, metric: str, bins: int
) -> np.ndarray:
    matrix = np.full((bins, bins), np.nan)
    for i in range(bins):
        for j in range(bins):
            bin_data = df[(df["x_bin"] == i) & (df["y_bin"] == j)]
            if not bin_data.empty:
                actual = bin_data[actual_col].to_numpy()
                predicted = bin_data[pred_col].to_numpy()
                metrics = calculate_general_metrics(actual, predicted)
                matrix[j, i] = metrics[metric]
    return matrix


# Heatmap plot
def plot_heatmap(
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    x_col: str,
    y_col: str,
    metric: str,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        matrix[::-1],
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=x_labels,
        yticklabels=y_labels[::-1],
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Metric: {metric.upper()}")
    # Add margins next to image
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.pyplot(fig)


st.markdown(
    "<h2 style='text-align: center; font-weight: normal;'>Performance Metrics Heatmap</h2>",
    unsafe_allow_html=True,
)

# Sidebar for data upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Parquet file", type=["csv", "parquet"])

# Load data
if uploaded_file is not None:
    df_inp = load_data(uploaded_file)
    st.session_state["df_inp"] = df_inp
    st.session_state["file_name"] = uploaded_file.name
elif "df_inp" in st.session_state:
    df_inp = st.session_state["df_inp"]
    st.sidebar.write(f"Using previously uploaded file: {st.session_state['file_name']}")
else:
    df_inp = None
    st.markdown(
        """<p style='text-align: center;'>Please select 'test_data_including_predictions' 
        CSV/Parquet file to proceed.</p>""",
        unsafe_allow_html=True,
    )

# Proceed if data is available
if df_inp is not None:
    numeric_columns = df_inp.select_dtypes(include=[np.number]).columns.tolist()
    pred_columns = [col for col in df_inp.columns if col.endswith("_pred")]

    if not numeric_columns:
        st.error("The uploaded file has no numeric columns.")
    elif not pred_columns:
        st.error("No columns ending with '_pred' found in the file.")
    else:
        # Sidebar inputs
        x_col = st.sidebar.selectbox("X-axis", numeric_columns)
        y_col = st.sidebar.selectbox(
            "Y-axis", numeric_columns, index=1 if len(numeric_columns) > 1 else 0
        )
        pred_col = st.sidebar.selectbox("Target", pred_columns)
        metric_options = ["e_mean_norm", "e_std_norm", "mae", "mae_norm", "mane", "r2"]
        metric = st.sidebar.selectbox("Performance Metric", metric_options, index=3)
        bins = st.sidebar.slider("Number of Bins", min_value=2, max_value=20, value=8, step=1)

        actual_col = pred_col.replace("_pred", "")
        if actual_col not in df_inp.columns:
            st.error(f"Actual column '{actual_col}' corresponding to '{pred_col}' not found.")
        else:
            df_inp["x_bin"] = pd.cut(df_inp[x_col], bins=bins, labels=False, include_lowest=True)
            df_inp["y_bin"] = pd.cut(df_inp[y_col], bins=bins, labels=False, include_lowest=True)
            x_labels = df_inp.groupby("x_bin")[x_col].mean().round(2).astype(str).tolist()
            y_labels = df_inp.groupby("y_bin")[y_col].mean().round(2).astype(str).tolist()
            metric_matrix = compute_metric_matrix(df_inp, actual_col, pred_col, metric, bins)

            data_min = np.nanmin(metric_matrix)
            data_max = np.nanmax(metric_matrix)
            st.sidebar.markdown("### Adjust Color Scale")
            vmin = st.sidebar.number_input("Minimum value (vmin)", value=float(data_min))
            vmax = st.sidebar.number_input("Maximum value (vmax)", value=float(data_max))

            if vmin >= vmax:
                st.error("vmin must be less than vmax")
            else:
                plot_heatmap(metric_matrix, x_labels, y_labels, x_col, y_col, metric, vmin, vmax)
