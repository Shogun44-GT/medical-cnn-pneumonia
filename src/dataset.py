import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Chemins vers le dataset
DATA_DIR = r"C:\Users\kingj\OneDrive\Documents\chest_xray\chest_xray"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# Taille d'image imposée par le PDF
IMG_SIZE = 224

# ------------------------------------------------------------
# Transformations
# Train : resize + augmentation légère + normalisation
# Val/Test : resize + normalisation uniquement (pas d'augmentation)
# ------------------------------------------------------------

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),           # rotation légère ±10°
    transforms.RandomHorizontalFlip(),       # flip horizontal
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # variation luminosité/contraste
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_datasets():
    """Charge les 3 splits avec leurs transformations respectives."""
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=val_test_transforms)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=val_test_transforms)
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size=32):
    """Retourne les DataLoaders pour train, val et test."""
    train_dataset, val_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def get_class_names():
    """Retourne les noms des classes : ['NORMAL', 'PNEUMONIA']"""
    dataset = datasets.ImageFolder(TRAIN_DIR)
    return dataset.classes


if __name__ == "__main__":
    # Test rapide pour vérifier que tout fonctionne
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    classes = get_class_names()

    print(f"Classes : {classes}")
    print(f"Batches train : {len(train_loader)}")
    print(f"Batches val   : {len(val_loader)}")
    print(f"Batches test  : {len(test_loader)}")

    # Vérification d'un batch
    images, labels = next(iter(train_loader))
    print(f"Shape d'un batch : {images.shape}")
    print(f"Labels : {labels[:8]}")
