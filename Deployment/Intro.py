
import json
import requests
  
import streamlit as st
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="Introduction", page_icon="✒️", layout="wide"
)


def animation(url):
    url = requests.get(url)
    url_json = dict()
    if url.status_code == 200:
        return url.json()
    else:
        print("Error in URL")

st.markdown("<h1 style='text-align: center; color: #0096ff; font-size:120px;'>Voice2Parkinson.ai<br></h1>", unsafe_allow_html=True)

st.title(":green[I  N  T  R  O]")
col1, col2 = st.columns(2)
with col1:
    url_brain = animation("https://assets4.lottiefiles.com/packages/lf20_n8y71jlq.json")
    st_lottie(url_brain,
          height=400,  
          width=400,
          speed=1,  
          loop=True,  
          quality='high',
          key='Brain' 
          )
    text = """<h3>Parkinson's disease is a progressive disorder that affects the nervous system and the parts of the body controlled by the nerves.<br> <br>
    Although Parkinson's disease can't be cured, medications might significantly improve your symptoms.</h2>"""
    st.markdown(text,unsafe_allow_html=True)
with col2:
    st.video("https://github.com/sugam21/Parkinson-Disease-Prediction-using-Support-Vector-Classifier/raw/master/video.mp4", format="video/mp4", start_time=0)

st.title(":green[S  Y  M  P  T  O  M  S] 😷 🤒 💉")
col1,col2,col3,col4 = st.columns(4)
with col1:
    url_symptomFirst = animation("https://assets9.lottiefiles.com/packages/lf20_owkzfxim.json")
    st_lottie(url_symptomFirst,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='voice' 
          )
    st.markdown("<h3>SPEECH CHANGES",unsafe_allow_html=True)

    text = "**:blue[You may speak softly, quickly, slur or hesitate before talking. Your speech may be more of a monotone rather than have the usual speech patterns]**"
    st.markdown(text,unsafe_allow_html=True)
    
with col2:
    url_symptomSecond = animation("https://assets7.lottiefiles.com/packages/lf20_AUPTpuHyla.json")
    st_lottie(url_symptomSecond,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='shaking' 
          )  
    st.markdown("<h3>SHAKING OF BODY",unsafe_allow_html=True)
    
    text = """**:blue[A tremor, or rhythmic shaking, usually begins in a limb, often your hand or fingers.
    You may rub your thumb and forefinger back and forth. This is known as a pill-rolling tremor.
    Your hand may tremble when it's at rest. The shaking may decrease when you are performing tasks]**"""
    st.markdown(text,unsafe_allow_html=True)
with col3:
    url_symptomThird = animation("https://assets8.lottiefiles.com/packages/lf20_yswp4uj3.json")
    st_lottie(url_symptomThird,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='writing' 
          )  
    st.markdown("<h3>WRITING CHANGES",unsafe_allow_html=True)
    text = """**:blue[It may become hard to write, and your writing may appear small]**"""
    st.markdown(text,unsafe_allow_html=True)

with col4:
    url_symptomFourth = animation("https://assets2.lottiefiles.com/packages/lf20_zm1z76.json")
    st_lottie(url_symptomFourth,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='muscle' 
          )  
    st.markdown("<h3>RIGID MUSCLES",unsafe_allow_html=True)
    text = """**:blue[Muscle stiffness may occur in any part of your body. The stiff muscles can be painful and limit your range of motion]**"""

    st.markdown(text,unsafe_allow_html=True)

st.title(":green[A P P R O A C H] 💡💡")
col1, col2, col3,col4 = st.columns([30,30,10,30])
with col1:
    model1 = animation("https://assets4.lottiefiles.com/private_files/lf30_8npirptd.json")
    st_lottie(model1,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='model1' 
          )  
with col2:
    model2 = animation("https://assets8.lottiefiles.com/packages/lf20_bXsnEx.json")
    st_lottie(model2,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='model2' 
          )  
with col3:
    st.markdown("<h3>➡️<br>➡️<br>➡️<br>➡️<br>➡️<h3>",unsafe_allow_html=True)
with col4:
    st.markdown("<h3> We have developed a model which takes features of voice as input and based on the features predicts whether the person is affected with Parkinson's or not.",unsafe_allow_html=True)
st.title(":green[I M P L E M E N T A T I O N] 📃 📄 ")

col1, col2 = st.columns(2)
with col1:
    hospital =  animation("https://assets2.lottiefiles.com/packages/lf20_mR5H7A.json")
    st_lottie(hospital,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='hospital' 
          )  
    st.markdown("<h3>HOSPITAL",unsafe_allow_html=True)
    text = """**One model is for doctors so that they can manually input the voice features**"""
    st.markdown(text,unsafe_allow_html=True)

with col2:
    hospital =  animation("https://assets10.lottiefiles.com/private_files/lf30_39yndvy8.json")
    st_lottie(hospital,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='mobile' 
          )  
    st.markdown("<h3>MOBILE APP",unsafe_allow_html=True)
    text = """**Another one for the people who cannot affort or who are not fit to travel to the hospital.
    They can record their own voice and find out within seconds if they have the disease or not**"""
    st.markdown(text,unsafe_allow_html=True)

st.title(":green[A D V A N T A G E S] 👍👍")
col1, col2,col3,col4 = st.columns(4)
with col1:
    accessibility = animation("https://assets1.lottiefiles.com/packages/lf20_y9wnxr3h.json")
    st_lottie(accessibility,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='accessibility' 
          )  
    st.markdown("<h3>EASE OF USE",unsafe_allow_html=True)

with col2:
    fast = animation("https://assets3.lottiefiles.com/packages/lf20_o4C0VO.json")
    st_lottie(fast,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='fast' 
          )  
    st.markdown("<h3>FAST RESULT",unsafe_allow_html=True)
with col3:
    accuract = animation("https://assets4.lottiefiles.com/packages/lf20_h9XCaLBwg0.json")
    st_lottie(accuract,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='accuracy' 
          )  
    st.markdown("<h3>ACCURATE PREDICTION",unsafe_allow_html=True)

with col4:
    cost = animation("https://assets3.lottiefiles.com/packages/lf20_yfsxyqxp.json")
    st_lottie(cost,
          height=200,  
          width=200,
          speed=1,  
          loop=True,  
          quality='high',
          key='cost' 
          )  
    st.markdown("<h3>FREE OF COST",unsafe_allow_html=True)
    
