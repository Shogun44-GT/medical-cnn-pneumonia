import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)

from src.dataset import get_dataloaders, get_class_names
from src.model import CNNBaseline

CHECKPOINT_PATH = os.path.join("outputs", "checkpoints", "best_model.pt")
FIGURES_PATH    = os.path.join("outputs", "figures")


def evaluate():
    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # --------------------------------------------------------
    # Chargement du meilleur modèle
    # --------------------------------------------------------
    model = CNNBaseline()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Modèle chargé depuis : {CHECKPOINT_PATH}")

    # --------------------------------------------------------
    # Données test
    # --------------------------------------------------------
    _, _, test_loader = get_dataloaders(batch_size=32)
    class_names = get_class_names()
    print(f"Classes : {class_names}")

    # --------------------------------------------------------
    # Prédictions sur le test set
    # --------------------------------------------------------
    all_labels  = []
    all_probs   = []
    all_preds   = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs   = outputs.squeeze().cpu().numpy()
            preds   = (outputs >= 0.5).squeeze().cpu().numpy().astype(int)

            all_probs.extend(probs if probs.ndim > 0 else [probs.item()])
            all_preds.extend(preds if preds.ndim > 0 else [preds.item()])
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)

    # --------------------------------------------------------
    # Métriques principales
    # --------------------------------------------------------
    print("\n" + "="*50)
    print("RÉSULTATS SUR LE TEST SET")
    print("="*50)
    print(classification_report(all_labels, all_preds,
                                target_names=class_names))

    auc = roc_auc_score(all_labels, all_probs)
    print(f"AUC : {auc:.4f}")

    # --------------------------------------------------------
    # Matrice de confusion
    # --------------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\nMatrice de confusion :")
    print(f"  TN = {tn}  |  FP = {fp}")
    print(f"  FN = {fn}  |  TP = {tp}")
    print(f"\nSensibilité (Recall) = {tp / (tp + fn):.4f}")
    print(f"Spécificité           = {tn / (tn + fp):.4f}")

    # Figure matrice de confusion
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(ax=ax1, colorbar=False, cmap="Blues")
    ax1.set_title("Matrice de confusion - Test set")
    plt.tight_layout()
    fig1.savefig(os.path.join(FIGURES_PATH, "matrice_confusion.png"))
    print(f"\nMatrice de confusion sauvegardée.")

    # --------------------------------------------------------
    # Courbe ROC
    # --------------------------------------------------------
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    ax2.set_xlabel("Taux de faux positifs (FPR)")
    ax2.set_ylabel("Taux de vrais positifs (TPR)")
    ax2.set_title("Courbe ROC - Test set")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(FIGURES_PATH, "courbe_roc.png"))
    print(f"Courbe ROC sauvegardée.")

    print("\nÉvaluation terminée ! Figures dans outputs/figures/")


if __name__ == "__main__":
    evaluate()
