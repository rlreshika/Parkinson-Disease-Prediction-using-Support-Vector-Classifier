import streamlit as st
import altair as alt
import pandas as pd
import time
import base64
import numpy as np
st.set_page_config(page_title = "Parkinson Disease Prediction", page_icon = "🧠", layout="wide")
# HEADER SECTION    

col1, col2, col3 = st.columns(3)

with col1:
    """
    # 192 Parkinson 
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, 
    sed do eiusmod tempor incididunt ut labore et dolore 
    magna aliqua. Volutpat sed cras ornare arcu dui vivamus.
    """
with col2:
    """
    # 29 Non Parkinson
    Stuff aligned to the right
    """
    st.button("➡️")

# This is for page background----------------------------
page_bg_image = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://images.unsplash.com/photo-1604079628040-94301bb21b91?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80");
background-size: cover; 

background-position: top left;
backgound-repeat: no-repeat;
}
[data-testid = "stHeader"]{
background-color:rgba(0,0,0,0);
}
[data-testid = "stToolbar"]{
right = 2rem;
}
</style>

"""
st.markdown(page_bg_image,unsafe_allow_html=True)

# This is for sidebar----------------------------

with st.sidebar:
    st.title("PARKINSON'S DISEASE PREDICTION🧠🧠🧠")
    while st.button("📑DESCRIPTION"):
        print("Hello")
    st.button("📈ANALYSIS")
    st.button("🤖PREDICTION")
    st.subheader("Links and resources")
    st.button("GITHUB CODE LINK")
    st.button("RESEARCH PAPER")
    
        # "https://github.com/sugam21/Parkinson-Disease-Prediction-by-analysing-various-machine-learning-algorithms.git"
# from streamlit_card import card
# hasClicked = card(
#         title="Hello World!",
#         text="Some description",
#         image="http://placekitten.com/200/300",
#         url="https://github.com/gamcoh/st-card"
from PIL import Image
col1,col2,col3 = st.columns((.333,.333,.333))
with col1:
    voice_image = Image.open(r'C:\Users\sugam\Downloads\—Pngtree—sound wave vector ilustration in_6341442.png')
    st.image(voice_image,width=300)
with col2:
    walking_image = Image.open(r"C:\Users\sugam\Downloads\pngfind.com-walking-png-16471.png")
    st.image(walking_image,width=100)
with col3:
    brain_image = Image.open(r"C:\Users\sugam\Downloads\pngfind.com-brain-png-3271826.png")
    st.image(brain_image,width=200)


# chart_data = pd.DataFrame(
#     np.random.randn(200, 3),
#     columns=['a', 'b', 'c'])


# st.vega_lite_chart(chart_data, {
#     'mark': {'type': 'circle', 'tooltip': True},
#     'encoding': {
#         'x': {'field': 'a', 'type': 'quantitative'},
#         'y': {'field': 'b', 'type': 'quantitative'},
#         'size': {'field': 'c', 'type': 'quantitative'},
#         'color': {'field': 'c', 'type': 'quantitative'},
#     }})


import streamlit as st
from vega_datasets import data

source = data.cars()

chart = {
    "mark": "point",
    "encoding": {
        "x": {
            "field": "Horsepower",
            "type": "quantitative",
        },
        "y": {
            "field": "Miles_per_Gallon",
            "type": "quantitative",
        },
        "color": {"field": "Origin", "type": "nominal"},
        "shape": {"field": "Origin", "type": "nominal"},
    },
}

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Vega-Lite native theme"])

with tab1:
    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
    st.vega_lite_chart(
        source, chart, theme="streamlit", use_container_width=True
    )
with tab2:
    st.vega_lite_chart(
        source, chart, theme=None, use_container_width=True
    )