import streamlit as st
import pandas as pd
import numpy as np
import webbrowser
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import parselmouth
import librosa
import pydub
import pyin
import seaborn as sns
from audio_recorder_streamlit import audio_recorder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import *
import plotly.express as px
import time



st.set_page_config(
    page_title="Parkinson Disease Prediction", page_icon="🧠", layout="wide"
)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.pinimg.com/originals/06/10/99/0610994f6f058e33e73b9d71553f0d5e.png");
background-size: 180%;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader]{{
background-color: rgba(0,0,0,0)
}}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

with st.sidebar:
    st.title("PARKINSON'S DISEASE PREDICTION🧠🧠🧠")
    st.subheader("Links and resources")
    if st.button("GITHUB CODE LINK"):
        url = 'https://github.com/sugam21/Parkinson-Disease-Prediction-using-Support-Vector-Classifier'
        webbrowser.open_new_tab(url)
    if st.button("RESEARCH PAPER"):
        url = "https://www.overleaf.com/read/zfcvdgdqwqys"
        webbrowser.open_new_tab(url)




def dataImport():
    url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url,encoding='latin1')
    X = df.drop(["status","name"],axis=1).copy()
    y = df["status"].copy()
    X_scaled = MinMaxScaler().fit_transform(X)
    sel = VarianceThreshold(threshold=0.01)
    sel.fit(X)
    selected_columns = sel.get_support()
    X_new = X.loc[:,selected_columns]
    # Scalling the values using standard scaler
    X_train,X_test,y_train,y_test = train_test_split(X_new,y,
                                                    test_size=0.2,
                                                    shuffle=True,stratify=y,random_state=0)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = SVC(C=100,kernel="rbf")
    model.fit(X_train,y_train)
    parkinson = X_new[y==1].describe().loc[["min","max"],:]
    nonparkinson = X_new[y==0].describe().loc[["min","max"],:]

    return model,scaler,X,y,X_train,X_test,y_train,y_test,parkinson,nonparkinson


def features_sound(audio_path):

        # Loading with ParselMouth to extract information about frequencies
        snd = parselmouth.Sound(audio_path)

        # Extract the pitch contour
        pitch = snd.to_pitch()

        non_silent_parts = pitch.selected_array['frequency'][pitch.selected_array['frequency'] != 0]

        # Compute the mean fundamental frequency
        mean_f0 = non_silent_parts.mean()

        # Compute the maximum fundamental frequency
        max_f0 = non_silent_parts.max()

        # Compute the maximum fundamental frequency
        min_f0 = non_silent_parts.min()

        # Loading audio from Audiosegment so that silent parts can be removed while calculating shimmer

        # Load the audio file
        audio = pydub.AudioSegment.from_wav(audio_path)

        # Split the audio into non-silent parts
        non_silent_audio = pydub.silence.split_on_silence(audio, min_silence_len=500, silence_thresh=-50)

        # Concatenate the non-silent parts into a single audio file
        output_audio = non_silent_audio[0]
        for audio_part in non_silent_audio[1:]:
            output_audio += audio_part

        # Calculate the local shimmer in dB for the output audio
        local_shimmer = output_audio.dBFS - output_audio.low_pass_filter(1000).dBFS

        # Extracting harmonicity and then hnr from it
        harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, min_f0, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

        # Load the audio file from librosa as it provides undisputed calculation of D2 & RDPE & spread1
        audio, sr = librosa.load(audio_path)

        # Calculate the Short-Time Fourier Transform (STFT) of the audio
        stft = librosa.stft(audio)

        # Calculate the magnitude and phase of the STFT
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Calculate the D2 feature
        d2 = np.sum(magnitude**2) / np.sum(phase**2)

        # Calculate the RDPE feature
        rpde = np.sum(magnitude**2) / np.sum(magnitude**2 + phase**2)

        # Calculate the Spread1 feature
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=75, fmax=600, sr=sr)
        spread1 = librosa.feature.spectral_contrast(y=audio, sr=sr, fmin=75, n_bands=1, quantile=0.02)
        spread1 = spread1[0, 0]


        return np.array([mean_f0, max_f0, min_f0, local_shimmer, hnr, rpde, -spread1, d2])



model,scaler,X,y,X_train,X_test,y_train,y_test,parkinson,nonparkinson = dataImport()


tab1, tab2 = st.tabs(["Manually Input","Voice Record"])
with tab1:
    st.header("Manual Input")
    slider, box  = st.columns([4, 2])

    # Functions to connect slider and box
    def update_slider():
        st.session_state.a = st.session_state.a1
        st.session_state.b = st.session_state.b1
        st.session_state.c = st.session_state.c1
        st.session_state.d = st.session_state.d1
        st.session_state.e = st.session_state.e1
        st.session_state.f = st.session_state.f1
        st.session_state.g = st.session_state.g1
        st.session_state.h = st.session_state.h1
    def update_numin():
        st.session_state.a1 = st.session_state.a
        st.session_state.b1 = st.session_state.b
        st.session_state.c1 = st.session_state.c
        st.session_state.d1 = st.session_state.d
        st.session_state.e1 = st.session_state.e
        st.session_state.f1 = st.session_state.f
        st.session_state.g1 = st.session_state.g
        st.session_state.h1 = st.session_state.h

    with slider:
        a = st.slider("MDVP\: Fo(Hz)",min_value=80.000,max_value = 270.000,value=154.229,step=0.001, key = 'a', format="%.3f", on_change = update_numin, help="Average Vocal Fundamental Frequency")
        b = st.slider("MDVP\: Fhi(Hz)",min_value=100.000,max_value = 600.000,value=197.105,step=0.001, key = 'b', format="%.3f",  on_change = update_numin, help="Maximum Vocal Fundamental Frequency")
        c = st.slider("MDVP\: Flo(Hz)",min_value=60.000,max_value = 240.000,value=116.325,step=0.001, key = 'c', format="%.3f",  on_change = update_numin, help="Minimum Vocal Fundamental Frequency")
        d = st.slider("MDVP\: Shimmer(dB)",min_value=0.010,max_value = 2.000,value=0.282,step=0.001, key = 'd', format="%.3f",  on_change = update_numin, help="Local Shimmer in dB")
        e = st.slider("HNR",min_value=2.000,max_value = 40.000,value=21.886,step=0.001, format="%.3f", key = 'e',  on_change = update_numin, help="Harmonics-to-Noise Ratio")
        f = st.slider("RPDE",min_value=0.100000,max_value = 1.000000,value=0.498536,step=0.000001, format="%.6f", key = 'f',  on_change = update_numin, help="Recurrence Period Density Entropy")
        g = st.slider("spread1",min_value=-10.00000,max_value = -0.0000001,value=-5.684397,step=0.000001, format="%.6f", key = 'g',  on_change = update_numin, help="Spread of Fundamental Frequency Contour")
        h = st.slider("D2",min_value=0.000000,max_value = 4.000000,value=2.381826,step=0.000001, format="%.6f", key = 'h',  on_change = update_numin, help="Correlation Dimension")
    
    with box:
        a1 = st.number_input("MDVP\: Fo(Hz)",min_value=80.000,max_value = 270.000,value=154.229,step=0.001, format="%.3f", key = 'a1', on_change = update_slider, help="Average Vocal Fundamental Frequency")
        st.write("\n")
        b1 = st.number_input("MDVP\: Fhi(Hz)",min_value=100.000,max_value = 600.000,value=197.105,step=0.001, format="%.3f", key = 'b1',  on_change = update_slider, help="Maximum Vocal Fundamental Frequency")
        st.write("\n")
        c1 = st.number_input("MDVP\: Flo(Hz)",min_value=60.000,max_value = 240.000,value=116.325,step=0.001, format="%.3f", key = 'c1', on_change = update_slider, help="Minimum Vocal Fundamental Frequency")
        st.write("\n")
        d1 = st.number_input("MDVP\: Shimmer(dB)",min_value=0.010,max_value = 2.000,value=0.282,step=0.001, format="%.3f", key = 'd1', on_change = update_slider, help="Local Shimmer in dB")
        st.write("\n")
        e1 = st.number_input("HNR",min_value=2.000,max_value = 40.000,value=21.886,step=0.001, format="%.5f", key = 'e1', on_change = update_slider, help="Harmonics-to-Noise Ratio")
        st.write("\n")
        f1 = st.number_input("RPDE",min_value=0.100000,max_value = 1.000000,value=0.498536,step=0.000001, format="%.6f", key = 'f1', on_change = update_slider, help="Recurrence Period Density Entropy")
        st.write("\n")
        g1 = st.number_input("spread1",min_value=-10.00000,max_value = -0.0000001,value=-5.684397,step=0.000001, format="%.6f", key = 'g1', on_change = update_slider, help="Spread of Fundamental Frequency Contour")
        st.write("\n")
        h1 = st.number_input("D2",min_value=0.000000,max_value = 4.000000,value=2.381826,step=0.000001, format="%.6f", key = 'h1', on_change = update_slider, help="Correlation Dimension")
        st.write("\n")

    input = np.array([a1,b1,c1,d1,e1,f1,g1,h1]).reshape(1,-1)
    input_scaled = scaler.transform(input)
    prediction = model.predict(input_scaled)
    if st.button("Predict"):
        with st.spinner("Wait for your result"):
                time.sleep(5)
        if prediction == 1:
            st.success("No Parkinson")
        else:
            st.warning("Parkinson")
    
with tab2:
    recorded_sound = None
    st.header("Voice Recording")
    audio_bytes = audio_recorder(text="Click on Icon to Record")

    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        
        wav_file = open("audio.wav", "wb")
        wav_file.write(audio_bytes)
        wav_file.close()
        
        recorded_sound = features_sound("audio.wav")

        if st.button("Predict",key = "Voice_Rec"):
            with st.spinner("Wait for your result"):
                time.sleep(5)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(recorded_sound)
            
            with col2:
                st.dataframe(parkinson.T)
            
            with col3:
                st.dataframe(nonparkinson.T)

            recorded_scaled = scaler.transform(recorded_sound.reshape(1, -1))
            prediction = model.predict(recorded_scaled)
            if prediction == 1:
                st.success("No Parkinson")
            else:
                st.warning("Parkinson")
