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

ARQUIVO_TREINO = PASTA_DADOS / "binario_treino.csv"
ARQUIVO_TESTE = PASTA_DADOS / "binario_teste.csv"

MODELO_BASE = "neuralmind/bert-base-portuguese-cased"

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5

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

def verificar_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU CUDA não foi encontrada. "
            "Verifique a instalação do PyTorch com CUDA."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

class DatasetBERT(Dataset):
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
        "f1_positivo": f1_score(y_real, y_pred, pos_label=1),
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

    print("\nCarregando tokenizer e BERTimbau...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE,
        num_labels=2,
        id2label=ID_PARA_ROTULO,
        label2id=ROTULO_PARA_ID,
    )

    dataset_treino = DatasetBERT(
        treino["texto"].tolist(),
        treino["rotulo"].tolist(),
        tokenizer
    )

    dataset_teste = DatasetBERT(
        teste["texto"].tolist(),
        teste["rotulo"].tolist(),
        tokenizer
    )

    argumentos = TrainingArguments(
        output_dir=str(PASTA_RESULTADOS / "bert_binario_checkpoints"),
        
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
    f1_teste = f1_score(y_real, y_pred, pos_label=1)

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
    print("RESULTADOS — BERT BINÁRIA")
    print("=" * 60)
    print(f"Precisão teste: {precisao_teste:.4f}")
    print(f"F1 teste       : {f1_teste:.4f}")
    print("\nMatriz de confusão:")
    print(matriz)
    print("\nRelatório de classificação:")
    print(relatorio)

    modelo.save_pretrained(PASTA_RESULTADOS / "bert_binario_modelo")
    tokenizer.save_pretrained(PASTA_RESULTADOS / "bert_binario_modelo")

    resultado = {
        "modelo": "BERTimbau fine-tuning",
        "modelo_base": MODELO_BASE,
        "tarefa": "binaria",
        "accuracy": float(precisao_teste),
        "f1_positivo": float(f1_teste),
        "classes": ["negativo", "positivo"],
        "matriz_confusao": matriz.tolist(),
        "tempo_treinamento_segundos": float(tempo_treinamento),
    }

    with open(
        PASTA_RESULTADOS / "bert_binario_resultados.json",
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