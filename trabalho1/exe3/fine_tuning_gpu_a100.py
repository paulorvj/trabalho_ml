import os
import time
import random
import pickle
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import VGG16_Weights


SEED = 42

CIFAR_DIR = "/home/paulo.rojunior/ml/cifar-10-batches-py"

VGG16_WEIGHTS_PATH = (
    "/home/paulo.rojunior/ml/vgg16-397923af.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 256
NUM_WORKERS = 16

EPOCHS_HEAD = 8
EPOCHS_FINE_TUNING = 20

LR_HEAD = 2e-3
LR_FINE_TUNING = 2e-5

MODEL_PATH = "vgg16_cifar10_fine_tuning_best_a100.pth"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# carregar cifar-10
def carregar_batch(caminho_arquivo):
    with open(caminho_arquivo, "rb") as arquivo:
        batch = pickle.load(arquivo, encoding="bytes")

    imagens = batch[b"data"]
    rotulos = np.array(batch[b"labels"])

    imagens = imagens.reshape(-1, 3, 32, 32)
    imagens = imagens.transpose(0, 2, 3, 1)

    return imagens, rotulos


def carregar_cifar10(cifar_dir):
    imagens_treino = []
    rotulos_treino = []

    for i in range(1, 6):
        caminho = os.path.join(cifar_dir, f"data_batch_{i}")

        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

        x_batch, y_batch = carregar_batch(caminho)
        imagens_treino.append(x_batch)
        rotulos_treino.append(y_batch)

    x_train = np.concatenate(imagens_treino, axis=0)
    y_train = np.concatenate(rotulos_treino, axis=0)

    caminho_teste = os.path.join(cifar_dir, "test_batch")

    if not os.path.exists(caminho_teste):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_teste}")

    x_test, y_test = carregar_batch(caminho_teste)

    return (x_train, y_train), (x_test, y_test)

class CIFAR10VGGDataset(Dataset):
    def __init__(self, imagens, rotulos, augmentation=False):
        self.imagens = imagens
        self.rotulos = rotulos
        self.augmentation = augmentation

        self.media = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32
        ).view(3, 1, 1)

        self.desvio = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32
        ).view(3, 1, 1)

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, indice):
        imagem = self.imagens[indice].astype(np.float32) / 255.0
        rotulo = int(self.rotulos[indice])

        if self.augmentation:
            if random.random() < 0.5:
                imagem = np.fliplr(imagem).copy()

            if random.random() < 0.5:
                desloc_x = random.randint(-2, 2)
                desloc_y = random.randint(-2, 2)

                imagem = np.roll(imagem, shift=desloc_x, axis=1)
                imagem = np.roll(imagem, shift=desloc_y, axis=0)

        imagem = np.transpose(imagem, (2, 0, 1))
        imagem = torch.tensor(imagem, dtype=torch.float32)

        imagem = imagem.unsqueeze(0)

        imagem = torch.nn.functional.interpolate(
            imagem,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        imagem = imagem.squeeze(0)

        imagem = (imagem - self.media) / self.desvio

        return imagem, torch.tensor(rotulo, dtype=torch.long)


# fine tuning VGG16
class VGG16FineTuning(nn.Module):

    def __init__(self):
        super().__init__()

        # Baixa automaticamente os pesos oficiais na primeira execução
        # e reutiliza o cache nas próximas.
        pesos = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=pesos)

        self.features = vgg.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Nova cabeça para CIFAR-10.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def congelar_base(self):
        for parametro in self.features.parameters():
            parametro.requires_grad = False

    def descongelar_bloco_final(self):
        for parametro in self.features.parameters():
            parametro.requires_grad = False

        for camada in self.features[24:]:
            for parametro in camada.parameters():
                parametro.requires_grad = True

        for parametro in self.classifier.parameters():
            parametro.requires_grad = True

def executar_epoca_treino(model, loader, criterion, optimizer, scaler):
    model.train()

    perda_total = 0.0
    corretos = 0
    total = 0

    for imagens, rotulos in loader:
        imagens = imagens.to(DEVICE, non_blocking=True)
        rotulos = rotulos.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):
            saidas = model(imagens)
            perda = criterion(saidas, rotulos)

        scaler.scale(perda).backward()
        scaler.step(optimizer)
        scaler.update()

        perda_total += perda.item() * imagens.size(0)

        previsoes = torch.argmax(saidas, dim=1)
        corretos += (previsoes == rotulos).sum().item()
        total += rotulos.size(0)

    return perda_total / total, corretos / total

@torch.no_grad()
def avaliar(model, loader, criterion):
    model.eval()

    perda_total = 0.0
    corretos = 0
    total = 0

    todos_rotulos = []
    todas_previsoes = []

    for imagens, rotulos in loader:
        imagens = imagens.to(DEVICE, non_blocking=True)
        rotulos = rotulos.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):
            saidas = model(imagens)
            perda = criterion(saidas, rotulos)

        perda_total += perda.item() * imagens.size(0)

        previsoes = torch.argmax(saidas, dim=1)

        corretos += (previsoes == rotulos).sum().item()
        total += rotulos.size(0)

        todos_rotulos.extend(rotulos.cpu().numpy())
        todas_previsoes.extend(previsoes.cpu().numpy())

    return (
        perda_total / total,
        corretos / total,
        np.array(todos_rotulos),
        np.array(todas_previsoes)
    )

def salvar_matriz_confusao(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.colorbar()

    plt.xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(10), CLASS_NAMES)

    limiar = cm.max() / 2

    for i in range(10):
        for j in range(10):
            cor_texto = "white" if cm[i, j] < limiar else "black"

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=cor_texto
            )

    plt.xlabel("Classe predita")
    plt.ylabel("Classe real")
    plt.title("Matriz de confusão — VGG16 com fine tuning")
    plt.tight_layout()
    plt.savefig("vgg16_fine_tuning_confusion_matrix.png", dpi=300)
    plt.show()

def main():
    os.environ["PYTHONHASHSEED"] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = True

    # Aceleração por Tensor Cores da A100.
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Dispositivo: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    (x_train, y_train), (x_test, y_test) = carregar_cifar10(CIFAR_DIR)

    indices = np.random.permutation(len(x_train))
    quantidade_validacao = int(0.10 * len(x_train))

    indices_val = indices[:quantidade_validacao]
    indices_train = indices[quantidade_validacao:]

    x_val = x_train[indices_val]
    y_val = y_train[indices_val]

    x_train = x_train[indices_train]
    y_train = y_train[indices_train]

    print(f"Treino    : {x_train.shape}")
    print(f"Validação : {x_val.shape}")
    print(f"Teste     : {x_test.shape}")

    train_dataset = CIFAR10VGGDataset(
        x_train, y_train, augmentation=False
    )

    val_dataset = CIFAR10VGGDataset(
        x_val, y_val, augmentation=False
    )

    test_dataset = CIFAR10VGGDataset(
        x_test, y_test, augmentation=False
    )

    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4
    )

    print("\nCarregando VGG16 pré-treinada no ImageNet...")

    model = VGG16FineTuning().to(DEVICE)

    model.congelar_base()

    total_parametros = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parâmetros totais: {total_parametros:,}")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    melhor_val_acc = 0.0
    historico_treino_acc = []
    historico_val_acc = []
    historico_treino_loss = []
    historico_val_loss = []

    inicio = time.time()

    print("\n" + "=" * 65)
    print("FASE 1 — TREINAMENTO DA CABEÇA CLASSIFICADORA")
    print("=" * 65)

    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=LR_HEAD,
        weight_decay=1e-4
    )

    for epoca in range(1, EPOCHS_HEAD + 1):
        treino_loss, treino_acc = executar_epoca_treino(
            model, train_loader, criterion, optimizer, scaler
        )

        val_loss, val_acc, _, _ = avaliar(
            model, val_loader, criterion
        )

        historico_treino_loss.append(treino_loss)
        historico_val_loss.append(val_loss)
        historico_treino_acc.append(treino_acc)
        historico_val_acc.append(val_acc)

        print(
            f"Fase 1 | Época {epoca:02d}/{EPOCHS_HEAD} | "
            f"loss treino: {treino_loss:.4f} | "
            f"acc treino: {treino_acc * 100:.2f}% | "
            f"loss val: {val_loss:.4f} | "
            f"acc val: {val_acc * 100:.2f}%"
        )

        if val_acc > melhor_val_acc:
            melhor_val_acc = val_acc

            torch.save(
                {
                    "phase": 1,
                    "epoch": epoca,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc
                },
                MODEL_PATH
            )

            print(f"  Melhor modelo salvo: {MODEL_PATH}")

    print("\n" + "=" * 65)
    print("FASE 2 — FINE TUNING DO BLOCO 5 DA VGG16")
    print("=" * 65)

    model.descongelar_bloco_final()

    parametros_treinaveis = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = optim.Adam(
        parametros_treinaveis,
        lr=LR_FINE_TUNING,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )

    for epoca in range(1, EPOCHS_FINE_TUNING + 1):
        treino_loss, treino_acc = executar_epoca_treino(
            model, train_loader, criterion, optimizer, scaler
        )

        val_loss, val_acc, _, _ = avaliar(
            model, val_loader, criterion
        )

        scheduler.step(val_loss)

        historico_treino_loss.append(treino_loss)
        historico_val_loss.append(val_loss)
        historico_treino_acc.append(treino_acc)
        historico_val_acc.append(val_acc)

        lr_atual = optimizer.param_groups[0]["lr"]

        print(
            f"Fase 2 | Época {epoca:02d}/{EPOCHS_FINE_TUNING} | "
            f"loss treino: {treino_loss:.4f} | "
            f"acc treino: {treino_acc * 100:.2f}% | "
            f"loss val: {val_loss:.4f} | "
            f"acc val: {val_acc * 100:.2f}% | "
            f"lr: {lr_atual:.7f}"
        )

        if val_acc > melhor_val_acc:
            melhor_val_acc = val_acc

            torch.save(
                {
                    "phase": 2,
                    "epoch": epoca,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc
                },
                MODEL_PATH
            )

            print(f"  Melhor modelo salvo: {MODEL_PATH}")

    tempo_total = time.time() - inicio

    # avaliação final
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = avaliar(
        model, test_loader, criterion
    )

    print("\n" + "=" * 65)
    print("RESULTADO FINAL — VGG16 COM FINE TUNING")
    print(f"Melhor precisão de validação: {melhor_val_acc * 100:.2f}%")
    print(f"Loss no teste               : {test_loss:.4f}")
    print(f"Precisão no teste           : {test_acc * 100:.2f}%")
    print(f"Tempo total                 : {tempo_total / 60:.2f} minutos")
    print("=" * 65)

    print("\nRELATÓRIO DE CLASSIFICAÇÃO\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4
        )
    )

    salvar_matriz_confusao(y_true, y_pred)

    epocas = range(1, len(historico_treino_acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epocas, historico_treino_acc, label="Treino")
    plt.plot(epocas, historico_val_acc, label="Validação")
    plt.axvline(
        EPOCHS_HEAD,
        linestyle="--",
        label="Início do fine tuning"
    )
    plt.xlabel("Época")
    plt.ylabel("Precisão")
    plt.title("Precisão — VGG16 com fine tuning")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("vgg16_fine_tuning_accuracy.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epocas, historico_treino_loss, label="Treino")
    plt.plot(epocas, historico_val_loss, label="Validação")
    plt.axvline(
        EPOCHS_HEAD,
        linestyle="--",
        label="Início do fine tuning"
    )
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Loss — VGG16 com fine tuning")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("vgg16_fine_tuning_loss.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
