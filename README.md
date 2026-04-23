# 🫁 Détection de Pneumonie — CNN Baseline

> Classification binaire de radiographies thoraciques (Normal vs Pneumonie)  
> Projet Deep Learning — B3 IA & Big Data — ECE Paris — Avril 2026

---

## 📋 Description

Ce projet implémente un pipeline complet de Deep Learning pour la détection automatique de pneumonie sur des radiographies thoraciques, à partir du dataset public **Chest X-Ray Images (Pneumonia)** disponible sur Kaggle.

L'objectif n'est pas de remplacer le médecin, mais de proposer un **outil d'aide à la décision** basé sur un CNN entraîné from scratch.

---

## 🏆 Résultats

| Métrique | Valeur |
|----------|--------|
| Accuracy | **80%** |
| Recall PNEUMONIA | **96%** |
| Spécificité | 53% |
| F1-score | 0.86 |
| AUC | **0.90** |

> ✅ Le modèle détecte **96% des cas de pneumonie** (seulement 16 manqués sur 390)

---

## 📁 Structure du projet

```
medical-cnn-pneumonia/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py        # Chargement, prétraitement, augmentation
│   ├── model.py          # Architecture CNN baseline
│   ├── train.py          # Boucle d'entraînement (Adam + BCE + Early Stopping)
│   └── eval.py           # Évaluation (Accuracy, Recall, F1, AUC, ROC)
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploration des données
│   ├── 02_training_cnn.ipynb # Entraînement du modèle
│   └── 03_evaluation.ipynb   # Évaluation finale
│
├── outputs/
│   ├── figures/              # Courbes, matrice de confusion, ROC
│   └── checkpoints/          # Meilleur modèle sauvegardé (best_model.pt)
│
├── reports/                  # Présentation PowerPoint
├── app.py                    # Interface Streamlit interactive
├── requirements.txt
└── README.md
```

---

## 🧠 Architecture CNN

```
Image 224×224
     │
     ▼
┌─────────────────┐
│ Conv(32) + ReLU │  → MaxPool → 112×112
├─────────────────┤
│ Conv(64) + ReLU │  → MaxPool → 56×56
├─────────────────┤
│ Conv(128)+ ReLU │  → MaxPool → 28×28
├─────────────────┤
│    Flatten      │
├─────────────────┤
│ Dense(128)      │  + Dropout(0.5)
├─────────────────┤
│ Dense(1)+Sigmoid│  → Probabilité PNEUMONIE ∈ [0, 1]
└─────────────────┘

Paramètres totaux : 12 938 561
```

**Paramètres d'entraînement :**
- Optimiseur : Adam (lr = 10⁻³)
- Perte : Binary Cross-Entropy
- Batch size : 32 | Époques : 20 max
- Early Stopping (patience = 5)
- GPU : NVIDIA (CUDA)

---

## 📦 Installation

**Prérequis : Python 3.11**

```bash
# Cloner le dépôt
git clone https://github.com/Shogun44-GT/medical-cnn-pneumonia.git
cd medical-cnn-pneumonia

# Installer les dépendances
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib scikit-learn streamlit Pillow jupyter
```

---

## 📂 Dataset

Télécharger le dataset depuis Kaggle :  
🔗 [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Placer les données dans la structure suivante :
```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

Mettre à jour le chemin `DATA_DIR` dans `src/dataset.py`.

---

## 🚀 Utilisation

### Entraînement
```bash
python -m src.train
```

### Évaluation
```bash
python -m src.eval
```

### Interface Streamlit
```bash
streamlit run app.py
```

### Notebooks
Ouvrir dans PyCharm ou Jupyter et exécuter les cellules dans l'ordre :
1. `01_eda.ipynb` — Exploration
2. `02_training_cnn.ipynb` — Entraînement
3. `03_evaluation.ipynb` — Évaluation

---

## 🖥️ Interface Streamlit

L'interface permet de charger n'importe quelle radiographie thoracique et d'obtenir instantanément :
- La classe prédite (NORMAL / PNEUMONIE)
- La probabilité associée
- Un graphique de visualisation

```bash
streamlit run app.py
```

---

## 📊 Figures générées

| Figure | Description |
|--------|-------------|
| `courbes_entrainement.png` | Loss et Accuracy par époque |
| `matrice_confusion.png` | TN / FP / FN / TP sur le test set |
| `courbe_roc.png` | Courbe ROC (AUC = 0.90) |
| `repartition_classes.png` | Déséquilibre des classes |
| `exemples_images.png` | Exemples de radiographies |
| `faux_negatifs.png` | Analyse des erreurs critiques |

---

## ⚠️ Avertissement médical

Ce modèle est un **outil de recherche académique**. Il ne remplace pas l'avis d'un professionnel de santé. Tout diagnostic doit être confirmé par un médecin.

---

## 🔧 Extensions possibles

- [ ] Transfert d'apprentissage (ResNet18, DenseNet121, EfficientNet)
- [ ] Optimisation du seuil de décision via la courbe ROC
- [ ] Grad-CAM pour la visualisation des zones d'intérêt
- [x] Interface Streamlit de démonstration ✅

---

## 📚 Références

- Kaggle — [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- LeCun, Bengio, Hinton — *Deep Learning*, Nature, 2015
- PyTorch Documentation — https://pytorch.org/docs

---

## 👤 Auteur

**Joan Andy Mballa Nsengue**  
B3 IA & Big Data — ECE Paris  
[GitHub](https://github.com/Shogun44-GT)
