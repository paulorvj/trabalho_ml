from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PASTA_PROJETO = Path(r"D:\temp\ml\trabalho2")
PASTA_DADOS = PASTA_PROJETO / "dados_processados"
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"

ARQUIVO_TREINO = PASTA_DADOS / "binario_treino.csv"
ARQUIVO_TESTE = PASTA_DADOS / "binario_teste.csv"

SEED = 42

MAX_VOCAB = 20_000
MAX_LEN = 80
EMBED_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.30

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3

ROTULO_PARA_ID = {
    "negativo": 0,
    "positivo": 1,
}

ID_PARA_ROTULO = {
    0: "negativo",
    1: "positivo",
}

def definir_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def obter_dispositivo():
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        dispositivo = torch.device("cpu")
        print("GPU não encontrada. O treinamento será feito em CPU.")
    return dispositivo

def tokenizar(texto):
    """
    Tokenização simples: converte para minúsculas e separa por espaços.
    """
    return str(texto).lower().split()

def construir_vocabulario(textos, max_vocab):
    frequencias = {}

    for texto in textos:
        for token in tokenizar(texto):
            frequencias[token] = frequencias.get(token, 0) + 1

    tokens_ordenados = sorted(
        frequencias.items(),
        key=lambda item: item[1],
        reverse=True
    )

    vocabulario = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token, _ in tokens_ordenados[:max_vocab - 2]:
        vocabulario[token] = len(vocabulario)

    return vocabulario

def texto_para_ids(texto, vocabulario, max_len):
    ids = [
        vocabulario.get(token, vocabulario["<UNK>"])
        for token in tokenizar(texto)
    ]

    ids = ids[:max_len]

    if len(ids) < max_len:
        ids += [vocabulario["<PAD>"]] * (max_len - len(ids))

    return ids

class DatasetTextoBinario(Dataset):
    def __init__(self, textos, rotulos, vocabulario, max_len):
        self.textos = textos
        self.rotulos = rotulos
        self.vocabulario = vocabulario
        self.max_len = max_len

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, indice):
        texto_ids = texto_para_ids(
            self.textos[indice],
            self.vocabulario,
            self.max_len
        )

        rotulo = ROTULO_PARA_ID[self.rotulos[indice]]

        return (
            torch.tensor(texto_ids, dtype=torch.long),
            torch.tensor(rotulo, dtype=torch.long)
        )

class ClassificadorLSTM(nn.Module):
    def __init__(
        self,
        tamanho_vocabulario,
        embed_dim,
        hidden_dim,
        dropout
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=tamanho_vocabulario,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.classificador = nn.Linear(
            hidden_dim * 2,
            2
        )

    def forward(self, x):
        x = self.embedding(x)

        _, (h_n, _) = self.lstm(x)

        # Últimos estados ocultos: forward e backward
        h_forward = h_n[-2]
        h_backward = h_n[-1]

        h = torch.cat((h_forward, h_backward), dim=1)

        h = self.dropout(h)

        return self.classificador(h)

def treinar_epoca(modelo, dataloader, otimizador, criterio, dispositivo):
    modelo.train()

    perdas = []

    for entradas, rotulos in tqdm(dataloader, desc="Treinando", leave=False):
        entradas = entradas.to(dispositivo)
        rotulos = rotulos.to(dispositivo)

        otimizador.zero_grad()

        logits = modelo(entradas)
        perda = criterio(logits, rotulos)

        perda.backward()
        otimizador.step()

        perdas.append(perda.item())

    return float(np.mean(perdas))

@torch.no_grad()
def prever(modelo, dataloader, dispositivo):
    modelo.eval()

    y_real = []
    y_pred = []

    for entradas, rotulos in dataloader:
        entradas = entradas.to(dispositivo)

        logits = modelo(entradas)
        previsoes = torch.argmax(logits, dim=1).cpu().numpy()

        y_pred.extend(previsoes.tolist())
        y_real.extend(rotulos.numpy().tolist())

    return np.array(y_real), np.array(y_pred)

def main():
    inicio_total = time.perf_counter()
    definir_seed(SEED)
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    dispositivo = obter_dispositivo()

    print("\nCarregando dados...")
    treino = pd.read_csv(ARQUIVO_TREINO, encoding="utf-8-sig")
    teste = pd.read_csv(ARQUIVO_TESTE, encoding="utf-8-sig")

    print(f"Treino: {len(treino)} exemplos")
    print(f"Teste : {len(teste)} exemplos")

    vocabulario = construir_vocabulario(
        treino["texto"].tolist(),
        MAX_VOCAB
    )

    print(f"Tamanho do vocabulário: {len(vocabulario)}")

    dataset_treino = DatasetTextoBinario(
        treino["texto"].tolist(),
        treino["rotulo"].tolist(),
        vocabulario,
        MAX_LEN
    )

    dataset_teste = DatasetTextoBinario(
        teste["texto"].tolist(),
        teste["rotulo"].tolist(),
        vocabulario,
        MAX_LEN
    )

    dataloader_treino = DataLoader(
        dataset_treino,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    dataloader_teste = DataLoader(
        dataset_teste,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    modelo = ClassificadorLSTM(
        tamanho_vocabulario=len(vocabulario),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(dispositivo)

    criterio = nn.CrossEntropyLoss()

    otimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=LEARNING_RATE
    )
    print("\nIniciando treinamento...\n")

    historico = []

    inicio_treinamento = time.perf_counter()
    for epoca in range(1, EPOCHS + 1):
        perda = treinar_epoca(
            modelo,
            dataloader_treino,
            otimizador,
            criterio,
            dispositivo
        )

        y_real_treino, y_pred_treino = prever(
            modelo,
            dataloader_treino,
            dispositivo
        )

        precisao_treino = accuracy_score(y_real_treino, y_pred_treino)
        f1_treino = f1_score(
            y_real_treino,
            y_pred_treino,
            pos_label=1
        )

        print(
            f"Época {epoca:02d}/{EPOCHS} | "
            f"Loss: {perda:.4f} | "
            f"Precisão treino: {precisao_treino:.4f} | "
            f"F1 treino: {f1_treino:.4f}"
        )

        historico.append({
            "epoca": epoca,
            "loss": perda,
            "accuracy_treino": precisao_treino,
            "f1_treino": f1_treino
        })

    tempo_treinamento = time.perf_counter() - inicio_treinamento

    print(
        f"\nTempo de treinamento: {tempo_treinamento:.2f} segundos "
        f"({tempo_treinamento / 60:.2f} minutos)"
    )

    print("\nAvaliando no conjunto de teste...")

    y_real, y_pred = prever(modelo, dataloader_teste, dispositivo)

    precisao_teste = accuracy_score(y_real, y_pred)
    f1_teste = f1_score(
        y_real,
        y_pred,
        pos_label=1
    )

    matriz = confusion_matrix(
        y_real,
        y_pred,
        labels=[0, 1]
    )

    relatorio = classification_report(
        y_real,
        y_pred,
        labels=[0, 1],
        target_names=["negativo", "positivo"],
        digits=4
    )

    print("\n" + "=" * 60)
    print("RESULTADOS — LSTM BINÁRIA")
    print("=" * 60)
    print(f"Precisão teste: {precisao_teste:.4f}")
    print(f"F1 teste       : {f1_teste:.4f}")
    print("\nMatriz de confusão:")
    print(matriz)
    print("\nRelatório de classificação:")
    print(relatorio)

    # Salva modelo e vocabulário
    torch.save(
        {
            "model_state_dict": modelo.state_dict(),
            "vocabulario": vocabulario,
            "configuracoes": {
                "max_len": MAX_LEN,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT
            }
        },
        PASTA_RESULTADOS / "lstm_binario_modelo.pt"
    )

    # Salva histórico
    pd.DataFrame(historico).to_csv(
        PASTA_RESULTADOS / "lstm_binario_historico.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Salva métricas e matriz
    resultado = {
        "modelo": "LSTM bidirecional",
        "tarefa": "binaria",
        "accuracy": float(precisao_teste),
        "f1_positivo": float(f1_teste),
        "classes": ["negativo", "positivo"],
        "matriz_confusao": matriz.tolist()
    }

    with open(
        PASTA_RESULTADOS / "lstm_binario_resultados.json",
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=4)

    print("\nArquivos salvos em:")
    print(PASTA_RESULTADOS)
    tempo_total = time.perf_counter() - inicio_total

    print(
        f"\nTempo total de execução: {tempo_total:.2f} segundos "
        f"({tempo_total / 60:.2f} minutos)"
    )

if __name__ == "__main__":
    main()