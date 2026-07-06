from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

PASTA_PROJETO = Path(r"D:\temp\ml\trabalho2")
PASTA_DADOS = PASTA_PROJETO / "dados_processados"
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"

ARQUIVO_TREINO = PASTA_DADOS / "multiclasse_treino.csv"
ARQUIVO_TESTE = PASTA_DADOS / "multiclasse_teste.csv"

MODELO_BASE = "neuralmind/bert-base-portuguese-cased"

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5

# Ordem fixa para IDs, relatório e matriz de confusão.
CLASSES = [
    "alegria",
    "desgosto",
    "medo",
    "neutro",
    "raiva",
    "surpresa",
    "tristeza",
]

ROTULO_PARA_ID = {
    rotulo: indice
    for indice, rotulo in enumerate(CLASSES)
}

ID_PARA_ROTULO = {
    indice: rotulo
    for indice, rotulo in enumerate(CLASSES)
}

def definir_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def verificar_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU CUDA não foi encontrada. "
            "Verifique a instalação do PyTorch com CUDA."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "VRAM total: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

class DatasetBERTMulticlasse(Dataset):
    def __init__(self, textos, rotulos, tokenizer):
        self.textos = textos
        self.rotulos = rotulos
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, indice):
        codificacao = self.tokenizer(
            str(self.textos[indice]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        item = {
            chave: valor.squeeze(0)
            for chave, valor in codificacao.items()
        }

        item["labels"] = torch.tensor(
            ROTULO_PARA_ID[self.rotulos[indice]],
            dtype=torch.long
        )

        return item

def calcular_metricas(eval_pred):
    logits, y_real = eval_pred
    y_pred = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "f1_weighted": f1_score(
            y_real,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_real,
            y_pred,
            average="macro",
            zero_division=0
        ),
    }

def main():
    inicio_total = time.perf_counter()

    definir_seed(SEED)
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    verificar_gpu()

    print("\nCarregando dados...")
    treino = pd.read_csv(ARQUIVO_TREINO, encoding="utf-8-sig")
    teste = pd.read_csv(ARQUIVO_TESTE, encoding="utf-8-sig")

    print(f"Treino: {len(treino)} exemplos")
    print(f"Teste : {len(teste)} exemplos")

    print("\nDistribuição no treino:")
    print(treino["rotulo"].value_counts())

    print("\nCarregando tokenizer e BERTimbau...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE,
        num_labels=len(CLASSES),
        id2label=ID_PARA_ROTULO,
        label2id=ROTULO_PARA_ID,
    )

    dataset_treino = DatasetBERTMulticlasse(
        treino["texto"].tolist(),
        treino["rotulo"].tolist(),
        tokenizer
    )

    dataset_teste = DatasetBERTMulticlasse(
        teste["texto"].tolist(),
        teste["rotulo"].tolist(),
        tokenizer
    )

    argumentos = TrainingArguments(
        output_dir=str(PASTA_RESULTADOS / "bert_multiclasse_checkpoints"),

        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        fp16=True,
        logging_strategy="epoch",
        save_strategy="no",
        eval_strategy="no",

        report_to="none",
        seed=SEED,
        data_seed=SEED,

        dataloader_num_workers=0,
    )

    treinador = Trainer(
        model=modelo,
        args=argumentos,
        train_dataset=dataset_treino,
        compute_metrics=calcular_metricas,
    )

    print("\nIniciando treinamento...")
    inicio_treinamento = time.perf_counter()

    treinador.train()

    tempo_treinamento = time.perf_counter() - inicio_treinamento

    print(
        f"\nTempo de treinamento: {tempo_treinamento:.2f} segundos "
        f"({tempo_treinamento / 60:.2f} minutos)"
    )

    print("\nAvaliando no conjunto de teste...")
    previsao = treinador.predict(dataset_teste)

    y_real = previsao.label_ids
    y_pred = np.argmax(previsao.predictions, axis=1)

    precisao_teste = accuracy_score(y_real, y_pred)

    f1_ponderado_teste = f1_score(
        y_real,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_macro_teste = f1_score(
        y_real,
        y_pred,
        average="macro",
        zero_division=0
    )

    matriz = confusion_matrix(
        y_real,
        y_pred,
        labels=list(range(len(CLASSES)))
    )

    relatorio = classification_report(
        y_real,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("RESULTADOS — BERT MULTICLASSE")
    print("=" * 60)
    print(f"Precisão teste      : {precisao_teste:.4f}")
    print(f"F1 ponderado teste  : {f1_ponderado_teste:.4f}")
    print(f"F1 macro teste      : {f1_macro_teste:.4f}")

    print("\nOrdem da matriz de confusão:")
    print(CLASSES)

    print("\nMatriz de confusão:")
    print(matriz)

    print("\nRelatório de classificação:")
    print(relatorio)

    modelo.save_pretrained(PASTA_RESULTADOS / "bert_multiclasse_modelo")
    tokenizer.save_pretrained(PASTA_RESULTADOS / "bert_multiclasse_modelo")

    resultado = {
        "modelo": "BERTimbau fine-tuning",
        "modelo_base": MODELO_BASE,
        "tarefa": "multiclasse",
        "accuracy": float(precisao_teste),
        "f1_weighted": float(f1_ponderado_teste),
        "f1_macro": float(f1_macro_teste),
        "classes": CLASSES,
        "matriz_confusao": matriz.tolist(),
        "tempo_treinamento_segundos": float(tempo_treinamento),
    }

    with open(
        PASTA_RESULTADOS / "bert_multiclasse_resultados.json",
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=4)

    tempo_total = time.perf_counter() - inicio_total

    print("\nArquivos salvos em:")
    print(PASTA_RESULTADOS)

    print(
        f"\nTempo total de execução: {tempo_total:.2f} segundos "
        f"({tempo_total / 60:.2f} minutos)"
    )

if __name__ == "__main__":
    main()