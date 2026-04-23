import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.model import CNNBaseline

# ------------------------------------------------------------
# Configuration (paramètres du PDF)
# ------------------------------------------------------------
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 1e-3
DROPOUT_RATE  = 0.5
PATIENCE      = 5       # Early stopping : arrêt si pas d'amélioration pendant 5 époques

CHECKPOINT_PATH = os.path.join("outputs", "checkpoints", "best_model.pt")
FIGURES_PATH    = os.path.join("outputs", "figures")


def train():
    # --------------------------------------------------------
    # Device : GPU si disponible, sinon CPU
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # --------------------------------------------------------
    # Données
    # --------------------------------------------------------
    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)
    print(f"Batches train : {len(train_loader)} | Batches val : {len(val_loader)}")

    # --------------------------------------------------------
    # Modèle, perte, optimiseur
    # --------------------------------------------------------
    model     = CNNBaseline(dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --------------------------------------------------------
    # Historique pour les courbes
    # --------------------------------------------------------
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []
    }

    best_val_loss  = float("inf")
    patience_count = 0

    # --------------------------------------------------------
    # Boucle d'entraînement
    # --------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):

        # --- Phase TRAIN ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = (outputs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        avg_train_loss = train_loss / train_total
        avg_train_acc  = train_correct / train_total

        # --- Phase VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs    = model(images)
                loss       = criterion(outputs, labels)
                val_loss  += loss.item() * images.size(0)
                preds      = (outputs >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        avg_val_loss = val_loss / val_total
        avg_val_acc  = val_correct / val_total

        # --- Sauvegarde historique ---
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)

        print(f"Époque {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        # --- Early Stopping + sauvegarde meilleur modèle ---
        if avg_val_loss < best_val_loss:
            best_val_loss  = avg_val_loss
            patience_count = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  --> Meilleur modèle sauvegardé (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            print(f"  --> Pas d'amélioration ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print("Early stopping déclenché.")
                break

    # --------------------------------------------------------
    # Courbes loss et accuracy
    # --------------------------------------------------------
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss par époque")
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs_range, history["train_acc"], label="Train Accuracy")
    ax2.plot(epochs_range, history["val_acc"],   label="Val Accuracy")
    ax2.set_title("Accuracy par époque")
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, "courbes_entrainement.png"))
    print(f"\nCourbes sauvegardées dans {FIGURES_PATH}")
    plt.show()

    print("\nEntraînement terminé !")
    print(f"Meilleur modèle : {CHECKPOINT_PATH}")

    return history


if __name__ == "__main__":
    train()
