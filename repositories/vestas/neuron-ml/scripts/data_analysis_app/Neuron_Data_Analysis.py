import random

import streamlit as st

st.set_page_config(page_title="Neuron Data Analysis", layout="wide")

st.markdown("<h1 style='text-align: center;'>Neuron Data Analysis</h1>", unsafe_allow_html=True)


quote_lib = [
    ("It is a capital mistake to theorize before one has data.", "Sherlock Holmes"),
    ("Without data, well then its just like your opinion man.", "The Dude"),
    ("Mo Data Less Problems", "The Notorious B.I.G."),
    ("I find your lack of data disturbing.", "Darth Vader"),
]

quote, author = random.choice(quote_lib)

st.markdown(
    f"""  
    <div style='text-align: center; margin: 20px 0;'>  
        <h4 style='color: darkorange; font-weight: normal;'><i>"{quote}"</i> – {author}</h4>  
    </div>  
""",
    unsafe_allow_html=True,
)
