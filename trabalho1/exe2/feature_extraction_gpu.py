import os
import time
import random
import pickle
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

SEED = 42

CIFAR_DIR = r"D:\temp\ml\trabalho1\cifar-10-batches-py"
MOBILENET_WEIGHTS_PATH = (
    r"D:\temp\ml\trabalho1\pesos\mobilenet_v2-b0353104.pth"
)

BATCH_SIZE = 128
NUM_WORKERS = 4

IMAGE_SIZE = 224

FEATURES_TRAIN_PATH = "cifar10_features_train.npy"
LABELS_TRAIN_PATH = "cifar10_labels_train.npy"

FEATURES_TEST_PATH = "cifar10_features_test.npy"
LABELS_TEST_PATH = "cifar10_labels_test.npy"

CLASSIFIER_PATH = "mobilenetv2_logistic_regression.joblib"
CONFUSION_MATRIX_PATH = "mobilenetv2_logistic_regression_confusion_matrix.png"
REPORT_PATH = "mobilenetv2_logistic_regression_report.txt"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# carregamento do CIFAR-10
def carregar_batch(caminho_arquivo):
    with open(caminho_arquivo, "rb") as arquivo:
        batch = pickle.load(arquivo, encoding="bytes")

    imagens = batch[b"data"]
    rotulos = np.array(batch[b"labels"])

    # (N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3)
    imagens = imagens.reshape(-1, 3, 32, 32)
    imagens = imagens.transpose(0, 2, 3, 1)

    return imagens, rotulos


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

# Dataset
class CIFAR10FeatureDataset(Dataset):
    def __init__(self, imagens, rotulos):
        self.imagens = imagens
        self.rotulos = rotulos

        # Média e desvio padrão usados no ImageNet.
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

        # HWC -> CHW
        imagem = np.transpose(imagem, (2, 0, 1))

        imagem = torch.tensor(imagem, dtype=torch.float32)

        # Redimensionamento de 32x32 para IMAGE_SIZE x IMAGE_SIZE.
        imagem = imagem.unsqueeze(0)

        imagem = torch.nn.functional.interpolate(
            imagem,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        imagem = imagem.squeeze(0)

        # Normalização exigida pelos pesos ImageNet.
        imagem = (imagem - self.media) / self.desvio

        return imagem, rotulo

# extrator
class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        if not os.path.exists(MOBILENET_WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Pesos não encontrados:\n{MOBILENET_WEIGHTS_PATH}"
            )

        # Cria a arquitetura sem tentar baixar pesos.
        mobilenet = models.mobilenet_v2(weights=None)

        # Carrega os pesos locais.
        state_dict = torch.load(
            MOBILENET_WEIGHTS_PATH,
            map_location="cpu",
            weights_only=True
        )

        mobilenet.load_state_dict(state_dict)

        # Mantém somente a parte convolucional da MobileNetV2.
        self.features = mobilenet.features

        # Gera um vetor de 1280 características por imagem.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Congela todos os pesos do extrator.
        for parametro in self.features.parameters():
            parametro.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

# extração das características
@torch.no_grad()
def extrair_caracteristicas(model, loader, nome_conjunto):
    model.eval()

    caracteristicas = []
    rotulos = []

    print(f"\nExtraindo características: {nome_conjunto}")

    for indice_lote, (imagens, y) in enumerate(loader, start=1):
        imagens = imagens.to(DEVICE, non_blocking=True)

        vetores = model(imagens)

        caracteristicas.append(vetores.cpu().numpy())
        rotulos.append(y.numpy())

        if indice_lote % 50 == 0 or indice_lote == len(loader):
            print(
                f"  Lote {indice_lote}/{len(loader)} "
                f"processado"
            )

    X = np.concatenate(caracteristicas, axis=0)
    y = np.concatenate(rotulos, axis=0)

    print(f"Características extraídas: {X.shape}")

    return X, y

# matriz de confusão
def salvar_matriz_confusao(y_true, y_pred):
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
    plt.title(
        "Matriz de confusão — MobileNetV2 + Regressão Logística"
    )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    #plt.show()

def main():
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

    print(f"Tamanho de entrada da MobileNetV2: {IMAGE_SIZE}x{IMAGE_SIZE}")

    (x_train, y_train), (x_test, y_test) = carregar_cifar10(CIFAR_DIR)

    print(f"Treino: {x_train.shape} | {y_train.shape}")
    print(f"Teste : {x_test.shape} | {y_test.shape}")

    train_dataset = CIFAR10FeatureDataset(x_train, y_train)
    test_dataset = CIFAR10FeatureDataset(x_test, y_test)

    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
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

    print("\nCarregando MobileNetV2 pré-treinada no ImageNet...")

    extrator = MobileNetV2FeatureExtractor().to(DEVICE)

    total_parametros = sum(
        p.numel()
        for p in extrator.parameters()
    )

    parametros_treinaveis = sum(
        p.numel()
        for p in extrator.parameters()
        if p.requires_grad
    )

    print(f"Parâmetros totais      : {total_parametros:,}")
    print(f"Parâmetros treináveis  : {parametros_treinaveis:,}")
    print("A MobileNetV2 está congelada e será usada apenas como extrator.")

    inicio_extracao = time.time()

    X_train, y_train_features = extrair_caracteristicas(
        extrator,
        train_loader,
        "treinamento"
    )

    X_test, y_test_features = extrair_caracteristicas(
        extrator,
        test_loader,
        "teste"
    )

    tempo_extracao = time.time() - inicio_extracao

    np.save(FEATURES_TRAIN_PATH, X_train)
    np.save(LABELS_TRAIN_PATH, y_train_features)

    np.save(FEATURES_TEST_PATH, X_test)
    np.save(LABELS_TEST_PATH, y_test_features)

    print(f"\nCaracterísticas salvas em arquivos .npy")
    print(f"Tempo de extração: {tempo_extracao / 60:.2f} minutos")

    print("\nTreinando Regressão Logística...")

    inicio_classificador = time.time()

    classificador = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        C=1.0,
        n_jobs=-1,
        random_state=SEED
    )

    classificador.fit(X_train, y_train_features)

    tempo_classificador = time.time() - inicio_classificador

    joblib.dump(classificador, CLASSIFIER_PATH)

    y_pred = classificador.predict(X_test)

    precisao_teste = np.mean(y_pred == y_test_features)

    relatorio = classification_report(
        y_test_features,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    print("\n" + "=" * 65)
    print("MOBILENETV2 CONGELADA + REGRESSÃO LOGÍSTICA")
    print(f"Precisão no teste: {precisao_teste * 100:.2f}%")
    print(f"Tempo de extração: {tempo_extracao / 60:.2f} minutos")
    print(
        f"Tempo do classificador: "
        f"{tempo_classificador / 60:.2f} minutos"
    )
    print("=" * 65)

    print("\nRELATÓRIO DE CLASSIFICAÇÃO\n")
    print(relatorio)

    with open(REPORT_PATH, "w", encoding="utf-8") as arquivo:
        arquivo.write(
            "MOBILENETV2 CONGELADA + REGRESSÃO LOGÍSTICA\n"
        )
        arquivo.write("=" * 65 + "\n")
        arquivo.write(
            f"Precisão no teste: {precisao_teste * 100:.2f}%\n"
        )
        arquivo.write(
            f"Tempo de extração: {tempo_extracao / 60:.2f} minutos\n"
        )
        arquivo.write(
            f"Tempo do classificador: "
            f"{tempo_classificador / 60:.2f} minutos\n\n"
        )
        arquivo.write(relatorio)

    salvar_matriz_confusao(y_test_features, y_pred)

    print(f"\nArquivos gerados:")
    print(f"- {FEATURES_TRAIN_PATH}")
    print(f"- {FEATURES_TEST_PATH}")
    print(f"- {CLASSIFIER_PATH}")
    print(f"- {REPORT_PATH}")
    print(f"- {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()