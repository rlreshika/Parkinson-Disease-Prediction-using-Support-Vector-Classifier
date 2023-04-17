import streamlit as st
import altair as alt
import pandas as pd
import time
import base64
import numpy as np

st.set_page_config(
    page_title="Parkinson Disease Prediction", page_icon="🧠", layout="wide"
)

# HEADER SECTION
st.title("What is Parkinson's Disesase ?")
st.markdown(
    r"""**Parkinson's disease is a neurodegenerative disorder that affects the central nervous system, particularly the brain. 
    It is characterized by the loss of dopamine-producing neurons in the substantia nigra, a region of the brain that is responsible for controlling movement.
    This loss of neurons leads to a variety of physical and cognitive symptoms that can significantly impact a person's quality of life.
    The exact cause of Parkinson's disease is not yet fully understood, but it is believed to be a combination of genetic and environmental factors.
    The disease is most commonly diagnosed in people over the age of 60, although it can occur earlier in life in some cases. 
    In addition to the loss of dopamine-producing neurons, Parkinson's disease is also characterized by the presence of Lewy bodies in the brain.
    Lewy bodies are abnormal protein deposits that accumulate in the brain cells of people with Parkinson's disease.**"""
)
st.title("Symptoms")
st.markdown(r"""**Tremors: Tremors or shaking are one of the most well-known symptoms of Parkinson's disease. They usually start in the hands or fingers and can spread to the arms, legs, and other parts of the body.

Rigidity: Parkinson's disease can cause stiffness and rigidity in the muscles, making it difficult to move freely. This can lead to a stooped posture, difficulty in initiating movements, and a decrease in facial expressions.

Bradykinesia: Bradykinesia refers to the slowness of movement and is another hallmark symptom of Parkinson's disease. It can take longer to start or complete movements, such as walking, turning, or getting out of a chair.

Balance and Coordination Problems: Parkinson's disease can cause problems with balance and coordination, making it difficult to walk, stand, or perform other activities.

Speech and Swallowing Problems: Parkinson's disease can cause speech and swallowing difficulties, such as slurred speech, soft voice, and difficulty in chewing and swallowing.

Loss of Smell: Parkinson's disease can cause a loss of sense of smell or a reduced ability to detect odors.

Sleep Disorders: Many people with Parkinson's disease have sleep disorders, such as insomnia, restless leg syndrome, and sleep apnea.

Cognitive Changes: Parkinson's disease can also cause cognitive changes, such as memory loss, difficulty in concentrating, and problems with decision-making. **
""")
col1,col2 = st.columns(2)
with col1: 
    st.title("Our Dataset")
with col2:
    st.title("Features")

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
# from PIL import Image
# col1,col2,col3 = st.columns((.333,.333,.333))
# with col1:
#     voice_image = Image.open(r'C:\Users\sugam\Downloads\—Pngtree—sound wave vector ilustration in_6341442.png')
#     st.image(voice_image,width=300)
# with col2:
#     walking_image = Image.open(r"C:\Users\sugam\Downloads\pngfind.com-walking-png-16471.png")
#     st.image(walking_image,width=100)
# with col3:
#     brain_image = Image.open(r"C:\Users\sugam\Downloads\pngfind.com-brain-png-3271826.png")
#     st.image(brain_image,width=200)


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
    st.vega_lite_chart(source, chart, theme="streamlit", use_container_width=True)
with tab2:
    st.vega_lite_chart(source, chart, theme=None, use_container_width=True)
