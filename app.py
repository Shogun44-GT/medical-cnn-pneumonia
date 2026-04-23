import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import CNNBaseline

# ─── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Détection de Pneumonie",
    page_icon="🫁",
    layout="centered"
)

# ─── Chargement du modèle (mis en cache) ──────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNNBaseline()
    checkpoint = os.path.join(os.path.dirname(__file__),
                              "outputs", "checkpoints", "best_model.pt")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# ─── Prétraitement identique à l'entraînement ─────────────────────────────────
def preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

# ─── Interface ────────────────────────────────────────────────────────────────
st.title("🫁 Détection de Pneumonie")
st.markdown("**Modèle** : CNN Baseline entraîné sur le dataset Chest X-Ray (Kaggle)")
st.markdown("---")

# Chargement du modèle
try:
    model, device = load_model()
    st.success(f"✅ Modèle chargé — Device : {device}")
except Exception as e:
    st.error(f"❌ Erreur de chargement du modèle : {e}")
    st.stop()

# Upload de l'image
st.subheader("📤 Charger une radiographie thoracique")
uploaded = st.file_uploader("Formats acceptés : JPG, JPEG, PNG",
                             type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Radiographie chargée", use_column_width=True)

    with col2:
        st.subheader("🔍 Analyse")
        with st.spinner("Analyse en cours..."):
            tensor = preprocess(image).to(device)
            with torch.no_grad():
                prob = model(tensor).item()

        pct_pneumonie = prob * 100
        pct_normal    = (1 - prob) * 100

        # Résultat principal
        if prob >= 0.5:
            st.error(f"### ⚠️ PNEUMONIE détectée")
            st.metric("Probabilité PNEUMONIE", f"{pct_pneumonie:.1f}%")
        else:
            st.success(f"### ✅ NORMAL")
            st.metric("Probabilité NORMAL", f"{pct_normal:.1f}%")

        # Jauge de probabilité
        st.markdown("**Probabilité PNEUMONIE :**")
        st.progress(float(prob))
        st.caption(f"{pct_pneumonie:.1f}% PNEUMONIE  |  {pct_normal:.1f}% NORMAL")

        # Graphique en barres
        fig, ax = plt.subplots(figsize=(4, 2.5))
        bars = ax.barh(["NORMAL", "PNEUMONIE"],
                       [pct_normal, pct_pneumonie],
                       color=["#1C7293", "#F96167"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probabilité (%)")
        for bar, val in zip(bars, [pct_normal, pct_pneumonie]):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=11)
        ax.set_title("Probabilités de classification")
        plt.tight_layout()
        st.pyplot(fig)

    # Avertissement médical
    st.markdown("---")
    st.warning(
        "⚠️ **Avertissement** : Ce modèle est un outil de recherche académique. "
        "Il ne remplace pas l'avis d'un professionnel de santé. "
        "Tout diagnostic doit être confirmé par un médecin."
    )

else:
    st.info("👆 Chargez une radiographie thoracique pour obtenir une prédiction.")

    # Informations sur le modèle
    st.markdown("---")
    st.subheader("📊 Performances du modèle (test set)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",    "80%")
    col2.metric("Recall",      "96%",  help="Sensibilité PNEUMONIE")
    col3.metric("Spécificité", "53%")
    col4.metric("AUC",         "0.90")
