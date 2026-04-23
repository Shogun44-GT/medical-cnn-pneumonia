import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    Architecture CNN baseline définie dans le PDF :
    Bloc 1 : Conv(32)  + ReLU + MaxPool
    Bloc 2 : Conv(64)  + ReLU + MaxPool
    Bloc 3 : Conv(128) + ReLU + MaxPool
    Flatten
    Dense(128) + Dropout
    Dense(1)   + Sigmoid
    """

    def __init__(self, dropout_rate=0.5):
        super(CNNBaseline, self).__init__()

        # Bloc 1 : Conv(32) + ReLU + MaxPool
        self.bloc1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 224x224 -> 112x112
        )

        # Bloc 2 : Conv(64) + ReLU + MaxPool
        self.bloc2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 112x112 -> 56x56
        )

        # Bloc 3 : Conv(128) + ReLU + MaxPool
        self.bloc3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 56x56 -> 28x28
        )

        # Flatten + Dense(128) + Dropout + Dense(1) + Sigmoid
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Test rapide : on passe un batch fictif dans le modèle
    model = CNNBaseline()
    print(model)
    print()

    # Compte le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre total de paramètres : {total_params:,}")

    # Test avec un batch fictif 224x224
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Shape entrée  : {x.shape}")
    print(f"Shape sortie  : {out.shape}")
    print(f"Valeurs sortie (probabilités) : {out.detach().squeeze()}")
