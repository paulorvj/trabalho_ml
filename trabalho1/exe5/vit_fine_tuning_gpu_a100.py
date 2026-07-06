# ============================================================
# Fine Tuning de Vision Transformer (ViT-B/16)
# Base: CIFAR-10
# Autor: Paulo Vieira
# ============================================================

import copy
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import vit_b_16

SEED = 42
BATCH_SIZE = 256
EPOCHS_STAGE1 = 8
EPOCHS_STAGE2 = 20
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5
PATIENCE = 8
NUM_CLASSES = 10
NUM_WORKERS = 16
IMAGE_SIZE = 224

CIFAR_ROOT = Path(".")
MODEL_PATH = "vit_cifar10_best.pth"
VIT_WEIGHTS = r"pesos/vit_b_16-c867db91.pth"

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# norm do imgnet
IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)

# data augmentation
transform_train = transforms.Compose([
    transforms.ToPILImage(),

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])

transform_eval = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])

class CIFAR10Dataset(Dataset):

    def __init__(
        self,
        imagens,
        rotulos,
        transform=None
    ):

        self.imagens = imagens
        self.rotulos = rotulos
        self.transform = transform

    def __len__(self):

        return len(self.imagens)

    def __getitem__(self, indice):

        imagem = self.imagens[indice]
        rotulo = int(self.rotulos[indice])

        if self.transform is not None:
            imagem = self.transform(imagem)

        return imagem, rotulo

def main():
    print("1 - MAIN")

    print(f"Dispositivo: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        torch.backends.cudnn.benchmark = True

    trainset = CIFAR10(
        root=".",
        train=True,
        download=False
    )

    testset = CIFAR10(
        root=".",
        train=False,
        download=False
    )

    x_train = trainset.data
    y_train = np.array(trainset.targets)

    x_test = testset.data
    y_test = np.array(testset.targets)

    # treino, validação
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=5000,
        stratify=y_train,
        random_state=SEED
    )

    print()

    print(f"Treino    : {x_train.shape}")
    print(f"Validação : {x_val.shape}")
    print(f"Teste     : {x_test.shape}")

    train_dataset = CIFAR10Dataset(
        x_train,
        y_train,
        transform_train
    )

    val_dataset = CIFAR10Dataset(
        x_val,
        y_val,
        transform_eval
    )

    test_dataset = CIFAR10Dataset(
        x_test,
        y_test,
        transform_eval
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True
    )

    class ViTFineTuning(nn.Module):

        def __init__(self):
            super().__init__()

            print()
            print("Carregando ViT-B/16 pré-treinado no ImageNet...")

            if not Path(VIT_WEIGHTS).exists():
                raise FileNotFoundError(
                    f"Arquivo não encontrado: {VIT_WEIGHTS}"
                )

            # Cria o modelo sem baixar pesos
            self.model = vit_b_16(weights=None)

            print("Carregando pesos locais...")

            state_dict = torch.load(
                VIT_WEIGHTS,
                map_location="cpu",
                weights_only=True
            )

            self.model.load_state_dict(state_dict)

            # Troca da cabeça classificadora
            in_features = self.model.heads.head.in_features

            self.model.heads.head = nn.Linear(
                in_features,
                NUM_CLASSES
            )
            self.freeze_encoder()

            print()
            print("Modelo carregado com sucesso.")
            print(f"Embedding: {in_features}")
            print(f"Número de classes: {NUM_CLASSES}")

        # Congela todo o encoder
        def freeze_encoder(self):

            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.heads.parameters():
                param.requires_grad = True

        # Fine Tuning
        # Libera somente os 3 últimos blocos Transformer
        def unfreeze_last_blocks(self):

            encoder = self.model.encoder.layers
            for block in encoder[-3:]:
                for param in block.parameters():
                    param.requires_grad = True

            # cabeça continua treinável
            for param in self.model.heads.parameters():
                param.requires_grad = True

        def total_parameters(self):
            return sum(
                p.numel()
                for p in self.model.parameters()
            )

        def trainable_parameters(self):
            return sum(
                p.numel()
                for p in self.model.parameters()
                if p.requires_grad
            )

        def forward(self, x):
            return self.model(x)
    
    model = ViTFineTuning()
    model = model.to(DEVICE)

    print()
    print(f"Parâmetros totais      : {model.total_parameters():,}")
    print(f"Treináveis Fase 1      : {model.trainable_parameters():,}")

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda")

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LR_STAGE1,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    def train_epoch(model, loader, criterion, optimizer, scaler):

        model.train()

        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            running_correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * running_correct / total

        return epoch_loss, epoch_acc


    @torch.no_grad()
    def evaluate(model, loader, criterion):
        model.eval()

        running_loss = 0.0
        running_correct = 0
        total = 0
        predictions = []
        targets = []

        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            running_correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * running_correct / total

        return (
            epoch_loss,
            epoch_acc,
            np.array(predictions),
            np.array(targets)
        )

    history = {
        "train_loss": [],
        "val_loss": [],

        "train_acc": [],
        "val_acc": []
    }

    best_acc = 0.0
    best_state = None
    counter = 0

    start_time = time.time()

    print()
    print("=" * 65)
    print("FASE 1 — TREINAMENTO DA CABEÇA")
    print("=" * 65)

    for epoch in range(EPOCHS_STAGE1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Fase 1 | "
            f"Época {epoch+1:02d}/{EPOCHS_STAGE1} | "
            f"loss treino: {train_loss:.4f} | "
            f"acc treino: {train_acc:.2f}% | "
            f"loss val: {val_loss:.4f} | "
            f"acc val: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                best_state,
                MODEL_PATH
            )

            counter = 0
            print(
                f"  Melhor modelo salvo: {MODEL_PATH}"
            )

        else:
            counter += 1

    print()
    print("=" * 65)
    print("FASE 2 — FINE TUNING")
    print("=" * 65)

    model.unfreeze_last_blocks()

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LR_STAGE2,
        weight_decay=1e-4
    )

    print()

    print(
        f"Treináveis Fase 2 : "
        f"{model.trainable_parameters():,}"
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    counter = 0

    for epoch in range(EPOCHS_STAGE2):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Fase 2 | "
            f"Época {epoch+1:02d}/{EPOCHS_STAGE2} | "
            f"loss treino: {train_loss:.4f} | "
            f"acc treino: {train_acc:.2f}% | "
            f"loss val: {val_loss:.4f} | "
            f"acc val: {val_acc:.2f}% | "
            f"lr: {lr:.7f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                best_state,
                MODEL_PATH
            )

            counter = 0

            print(
                f"  Melhor modelo salvo: {MODEL_PATH}"
            )

        else:
            counter += 1

        if counter >= PATIENCE:
            print()
            print("Early stopping.")
            break

    print()
    print("Carregando melhor modelo...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    test_loss, test_acc, predictions, targets = evaluate(
        model,
        test_loader,
        criterion
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 65)
    print("RESULTADO FINAL — ViT-B/16 COM FINE TUNING")
    print(f"Melhor precisão de validação: {best_acc:.2f}%")
    print(f"Loss no teste               : {test_loss:.4f}")
    print(f"Precisão no teste           : {test_acc:.2f}%")
    print(f"Tempo total                 : {elapsed/60:.2f} minutos")
    print("=" * 65)

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    report = classification_report(
        targets,
        predictions,
        target_names=classes,
        digits=4
    )

    print()
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print()
    print(report)

    with open(
        "vit_classification_report.txt",
        "w",
        encoding="utf8"
    ) as f:
        f.write(report)

    cm = confusion_matrix(
        targets,
        predictions
    )

    plt.figure(figsize=(8,8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de Confusão - ViT-B/16")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.colorbar()

    ticks = np.arange(len(classes))

    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    plt.tight_layout()
    plt.savefig(
        "vit_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # acc
    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_acc"],
        label="Treino"
    )

    plt.plot(
        history["val_acc"],
        label="Validação"
    )

    plt.xlabel("Época")
    plt.ylabel("Precisão (%)")
    plt.title("Accuracy - ViT-B/16")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "vit_accuracy.png",
        dpi=300
    )

    plt.close()

    # loss
    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_loss"],
        label="Treino"
    )

    plt.plot(
        history["val_loss"],
        label="Validação"
    )

    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Loss - ViT-B/16")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "vit_loss.png",
        dpi=300
    )

    plt.close()

    print()
    print("Arquivos gerados:")
    print("- vit_cifar10_best.pth")
    print("- vit_classification_report.txt")
    print("- vit_accuracy.png")
    print("- vit_loss.png")
    print("- vit_confusion_matrix.png")


if __name__ == '__main__':
    main()
