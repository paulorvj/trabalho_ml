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
from sklearn.metrics import classification_report, confusion_matrix
from torchinfo import summary


SEED = 42

CIFAR_DIR = r"D:\temp\ml\trabalho1\cifar-10-batches-py"

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_WORKERS = 4

MODEL_PATH = "cnn_cifar10_flatten_best.pth"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def carregar_batch(caminho_arquivo):
    with open(caminho_arquivo, "rb") as arquivo:
        batch = pickle.load(arquivo, encoding="bytes")

    imagens = batch[b"data"]
    rotulos = np.array(batch[b"labels"])

    # (N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3)
    imagens = imagens.reshape(-1, 3, 32, 32)
    imagens = imagens.transpose(0, 2, 3, 1)

    return imagens, rotulos

# carregamento local CIFAR-10
def carregar_cifar10(cifar_dir):
    imagens_treino = []
    rotulos_treino = []

    for i in range(1, 6):
        caminho = os.path.join(cifar_dir, f"data_batch_{i}")

        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"Arquivo não encontrado: {caminho}\n"
                "Verifique o valor de CIFAR_DIR."
            )

        x_batch, y_batch = carregar_batch(caminho)
        imagens_treino.append(x_batch)
        rotulos_treino.append(y_batch)

    x_train = np.concatenate(imagens_treino, axis=0)
    y_train = np.concatenate(rotulos_treino, axis=0)

    caminho_teste = os.path.join(cifar_dir, "test_batch")

    if not os.path.exists(caminho_teste):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_teste}\n"
            "Verifique o valor de CIFAR_DIR."
        )

    x_test, y_test = carregar_batch(caminho_teste)

    return (x_train, y_train), (x_test, y_test)

class CIFAR10Dataset(Dataset):
    def __init__(self, imagens, rotulos, augmentation=False):
        self.imagens = imagens
        self.rotulos = rotulos
        self.augmentation = augmentation

        self.media = np.array(
            [0.4914, 0.4822, 0.4465],
            dtype=np.float32
        )

        self.desvio = np.array(
            [0.2470, 0.2435, 0.2616],
            dtype=np.float32
        )

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, indice):
        imagem = self.imagens[indice].astype(np.float32) / 255.0
        rotulo = int(self.rotulos[indice])

        # O aumento de dados não é usado, no código do treinamento a opção está desligada
        if self.augmentation:
            # Inversão horizontal
            if random.random() < 0.5:
                imagem = np.fliplr(imagem).copy()

            # deslocamentos horizontais e verticais
            if random.random() < 0.5:
                desloc_x = random.randint(-2, 2)
                desloc_y = random.randint(-2, 2)

                imagem = np.roll(imagem, shift=desloc_x, axis=1)
                imagem = np.roll(imagem, shift=desloc_y, axis=0)

        # HWC -> CHW
        imagem = np.transpose(imagem, (2, 0, 1))

        # Normalização por canal
        imagem = (
            imagem - self.media[:, None, None]
        ) / self.desvio[:, None, None]

        return (
            torch.tensor(imagem, dtype=torch.float32),
            torch.tensor(rotulo, dtype=torch.long)
        )

# criação da CNN do zero
class CNNZero(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout(0.20),

            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout(0.30),

            # Bloco 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout(0.40),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# treinamento
def executar_epoca_treino(model, loader, criterion, optimizer):
    model.train()

    perda_total = 0.0
    corretos = 0
    total = 0

    for imagens, rotulos in loader:
        imagens = imagens.to(DEVICE, non_blocking=True)
        rotulos = rotulos.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        saidas = model(imagens)
        perda = criterion(saidas, rotulos)

        perda.backward()
        optimizer.step()

        perda_total += perda.item() * imagens.size(0)

        previsoes = torch.argmax(saidas, dim=1)
        corretos += (previsoes == rotulos).sum().item()
        total += rotulos.size(0)

    return perda_total / total, corretos / total

# avaliação
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

def main():
    # configurando seed para reprodutibilidade
    os.environ["PYTHONHASHSEED"] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = True

    print(f"Dispositivo: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    (x_train, y_train), (x_test, y_test) = carregar_cifar10(CIFAR_DIR)

    print(f"Treino: {x_train.shape} | {y_train.shape}")
    print(f"Teste : {x_test.shape} | {y_test.shape}")

    # divisão treino/validação
    indices = np.random.permutation(len(x_train))
    quantidade_validacao = int(0.10 * len(x_train))

    indices_val = indices[:quantidade_validacao]
    indices_train = indices[quantidade_validacao:]

    x_val = x_train[indices_val]
    y_val = y_train[indices_val]

    x_train = x_train[indices_train]
    y_train = y_train[indices_train]

    print(f"Treino após divisão: {x_train.shape}")
    print(f"Validação          : {x_val.shape}")

    train_dataset = CIFAR10Dataset(
        x_train,
        y_train,
        augmentation=False
    )

    val_dataset = CIFAR10Dataset(
        x_val,
        y_val,
        augmentation=False
    )

    test_dataset = CIFAR10Dataset(
        x_test,
        y_test,
        augmentation=False
    )

    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )

    # modelo, loss, otimizador e scheduler
    model = CNNZero().to(DEVICE)

    total_parametros = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Parâmetros treináveis: {total_parametros:,}")

    print("\nRESUMO DA ARQUITETURA\n")

    # (BATCH_SIZE, CANAIS, ALTURA, LARGURA)
    '''
    summary(
        model,
        input_size=(BATCH_SIZE, 3, 32, 32),
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds"
        ),
        depth=4,
        device=DEVICE.type
    )
    '''
    
    summary(
        model,
        input_size=(1, 3, 32, 32),
        col_names=("input_size", "output_size", "num_params"),
        depth=4,
        device=DEVICE.type
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6
    )

    # treinamento
    historico_treino_loss = []
    historico_val_loss = []

    historico_treino_acc = []
    historico_val_acc = []

    melhor_val_acc = 0.0
    epocas_sem_melhoria = 0
    PACIENCIA_EARLY_STOPPING = 10

    inicio = time.time()

    for epoca in range(1, EPOCHS + 1):
        treino_loss, treino_acc = executar_epoca_treino(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, _, _ = avaliar(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        historico_treino_loss.append(treino_loss)
        historico_val_loss.append(val_loss)

        historico_treino_acc.append(treino_acc)
        historico_val_acc.append(val_acc)

        lr_atual = optimizer.param_groups[0]["lr"]

        print(
            f"Época {epoca:02d}/{EPOCHS} | "
            f"loss treino: {treino_loss:.4f} | "
            f"acc treino: {treino_acc * 100:.2f}% | "
            f"loss val: {val_loss:.4f} | "
            f"acc val: {val_acc * 100:.2f}% | "
            f"lr: {lr_atual:.6f}"
        )

        if val_acc > melhor_val_acc:
            melhor_val_acc = val_acc
            epocas_sem_melhoria = 0

            torch.save(
                {
                    "epoch": epoca,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "class_names": CLASS_NAMES
                },
                MODEL_PATH
            )

            print(f"  Melhor modelo salvo: {MODEL_PATH}")

        else:
            epocas_sem_melhoria += 1

        if epocas_sem_melhoria >= PACIENCIA_EARLY_STOPPING:
            print("\nEarly stopping acionado.")
            break

    tempo_total = time.time() - inicio

    print(f"\nTempo total de treinamento: {tempo_total / 60:.2f} minutos")

    # avaliação final
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = avaliar(
        model,
        test_loader,
        criterion
    )

    print("\n" + "=" * 65)
    print(
        f"Melhor precisão de validação: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )
    print(f"Loss no teste               : {test_loss:.4f}")
    print(f"Precisão no teste           : {test_acc * 100:.2f}%")
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

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.colorbar()

    plt.xticks(
        range(10),
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

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
    plt.title("Matriz de confusão — CNN com Flatten")
    plt.tight_layout()
    plt.savefig("cnn_cifar10_flatten_confusion_matrix.png", dpi=300)
    #plt.show()

    # Curva de precisão
    epocas = range(1, len(historico_treino_acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epocas, historico_treino_acc, label="Treino")
    plt.plot(epocas, historico_val_acc, label="Validação")

    plt.xlabel("Época")
    plt.ylabel("Precisão")
    plt.title("Precisão durante o treinamento — CNN com Flatten")

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("cnn_cifar10_flatten_accuracy.png", dpi=300)
    #plt.show()

    # Curva de loss
    plt.figure(figsize=(10, 5))
    plt.plot(epocas, historico_treino_loss, label="Treino")
    plt.plot(epocas, historico_val_loss, label="Validação")

    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Loss durante o treinamento — CNN com Flatten")

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("cnn_cifar10_flatten_loss.png", dpi=300)
    #plt.show()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()