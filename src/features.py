"""Criação de variáveis (funções puras) e pré-processamento."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUNAS_SERVICO = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_ticket_medio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ticket_medio_mensal"] = np.where(
        out["tenure"] > 0, out["TotalCharges"] / out["tenure"], 0.0
    )
    return out


def add_diferenca_cobranca(df: pd.DataFrame) -> pd.DataFrame:
    """Exige ticket_medio_mensal já criado."""
    out = df.copy()
    out["diferenca_cobranca"] = out["MonthlyCharges"] - out["ticket_medio_mensal"]
    return out


def add_qtd_servicos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["qtd_servicos"] = (out[COLUNAS_SERVICO] == "Yes").sum(axis=1)
    return out


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cliente_novo"] = (out["tenure"] <= 6).astype(int)
    out["contrato_mensal"] = (out["Contract"] == "Month-to-month").astype(int)
    out["pagamento_automatico"] = (
        out["PaymentMethod"].str.contains("automatic", case=False, na=False)
    ).astype(int)
    return out


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as features na ordem correta."""
    return (
        df.pipe(add_ticket_medio)
        .pipe(add_diferenca_cobranca)
        .pipe(add_qtd_servicos)
        .pipe(add_flags)
    )


def montar_preprocessador(X: pd.DataFrame) -> ColumnTransformer:
    """Escala numéricas e codifica categóricas. Só define; o fit acontece no Pipeline."""
    numericas = X.select_dtypes(include="number").columns.tolist()
    categoricas = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ])
