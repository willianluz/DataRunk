"""Carga, limpeza, validação e split dos dados de churn (Telco)."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

COLUNAS_OBRIGATORIAS = [
    "customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "Churn",
]


def carregar_config(caminho: str = "configs/config.yaml") -> dict:
    """Lê o arquivo de configuração."""
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def carregar_dados(caminho: str) -> pd.DataFrame:
    """Lê o CSV bruto."""
    if not Path(caminho).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho)


def validar_dados(df: pd.DataFrame) -> None:
    """Falha cedo se algo essencial estiver errado."""
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes: {faltando}")
    if df["customerID"].duplicated().any():
        raise ValueError("Há customerID duplicado")
    if df.empty:
        raise ValueError("DataFrame vazio")


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza básica. Não altera o DataFrame original."""
    out = df.copy()

    # TotalCharges vem como texto; vazios são clientes novos (tenure = 0)
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out.loc[out["TotalCharges"].isna() & (out["tenure"] == 0), "TotalCharges"] = 0.0

    # Alvo vira 0/1
    out["Churn"] = (out["Churn"] == "Yes").astype(int)
    return out


def dividir_treino_teste(
    df: pd.DataFrame, target: str, test_size: float, seed: int
):
    """Split estratificado: mantém a proporção de churn nos dois conjuntos."""
    X = df.drop(columns=[target, "customerID"])
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

def salvar_dados(df: pd.DataFrame, caminho: str) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)