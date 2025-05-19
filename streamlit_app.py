import streamlit as st
import torch
import torchaudio
import os

from utils.arabicTTSwrapper import ArabicTTSWrapper

st.set_page_config(page_title="Synthèse Vocale Arabe", layout="centered")
st.title("🗣️ Synthèse Vocale Arabe")
st.markdown("Entrez une phrase en arabe, sélectionnez un modèle, et écoutez la sortie audio.")

# 📌 Lien vers outil de Tashkīl
with st.expander("🧰 أداة تشكيل النصوص العربية"):
    st.markdown(
        """
        🔤 يمكنك استخدام [هذا الرابط](https://www.horofk.com/apps/tashkil/index.php) لتشكيل النصوص العربية قبل إدخالها هنا.
        """,
        unsafe_allow_html=True
    )

# 📥 Zone de texte
text = st.text_area("✍️ Texte arabe à synthétiser :", height=100)

# 🔀 Choix du modèle
model_options = {
    "Modèle personnalisé": "custom_model",
    "Modèle pré-entraîné": "pretrained_model"
}
model_label = st.selectbox("🎛️ Choisir le modèle TTS :", list(model_options.keys()))
model_key = model_options[model_label]

# 🧠 Instancier le wrapper TTS (singleton)
@st.cache_resource
def load_tts_system():
    return ArabicTTSWrapper()

tts_system = load_tts_system()

# 📤 Actions
col1, col2 = st.columns(2)

if col1.button("📜 Afficher les phonèmes"):
    if text.strip():
        try:
            _, phonemes = tts_system.synthesize(text, model_key)
            st.success("📢 Phonèmes détectés :")
            st.write(phonemes)
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
    else:
        st.warning("⚠️ Veuillez entrer du texte.")

if col2.button("🔊 Générer et écouter l’audio"):
    if text.strip():
        try:
            wav, _ = tts_system.synthesize(text, model_key)
            audio_file = "tts_streamlit_output.wav"
            torchaudio.save(audio_file, wav.unsqueeze(0), 22050)
            st.success("✅ Résultat audio :")
            st.audio(audio_file, format="audio/wav")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
    else:
        st.warning("⚠️ Veuillez entrer du texte.")
