import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events


@st.cache_data
def load_data(file: str) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_parquet(file)


st.markdown(
    "<h2 style='text-align: center; font-weight: normal;'>Scatter Matrix</h2>",
    unsafe_allow_html=True,
)

st.sidebar.header("Upload Data")
file = st.sidebar.file_uploader("Choose a CSV or Parquet file", type=["csv", "parquet"])

if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = []
if "plot_height" not in st.session_state:
    st.session_state["plot_height"] = 1100


if file is not None:
    st.session_state["data"] = load_data(file)
    st.session_state["file_name"] = file.name
    data = st.session_state["data"]
elif "data" in st.session_state:
    data = st.session_state["data"]
    st.sidebar.write(f"Using previously uploaded file: {st.session_state['file_name']}")
else:
    data = None

if data is not None:
    st.sidebar.header("Feature Selection")
    features = data.columns.tolist()

    selected_features = st.sidebar.multiselect(
        "Select features:", features, default=st.session_state["selected_features"]
    )

    plot_height = st.sidebar.slider(
        "Select plot height (px)",
        min_value=400,
        max_value=3000,
        value=st.session_state["plot_height"],
    )

    if st.sidebar.button("Save settings"):
        st.session_state["selected_features"] = selected_features
        st.session_state["plot_height"] = plot_height

    if selected_features:
        fig = px.scatter_matrix(
            data,
            dimensions=selected_features,
            title="Scatter Matrix of Selected Features",
        )

        fig.update_traces(
            marker_color="black",
            marker_size=5,
            unselected_marker=dict(opacity=0.3, color="orangered"),
        )

        fig.update_layout(
            clickmode="event+select",
            autosize=True,
            plot_bgcolor="ivory",
            paper_bgcolor="ivory",
        )

        selected_points = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            key="scatter_matrix",
            override_width="100%",
            override_height=plot_height,
        )

        if selected_points:
            st.write("You clicked the following point(s):")
            st.write(selected_points)
        else:
            st.write("Click on points in the scatter matrix to select them.")
    else:
        st.write("Please select at least one feature to display the scatter matrix.")
else:
    st.write("")
    st.markdown(
        "<p style='text-align: center;'>Please select a CSV/Parquet file to proceed</p>",
        unsafe_allow_html=True,
    )
