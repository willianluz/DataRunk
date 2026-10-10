import pandas as pd
import pytest

from src.features import (
    add_ticket_medio, add_diferenca_cobranca, add_qtd_servicos,
    add_flags, criar_features, COLUNAS_SERVICO,
)


def df_base():
    d = {
        "tenure": [0, 10, 3],
        "MonthlyCharges": [50.0, 100.0, 70.0],
        "TotalCharges": [0.0, 800.0, 210.0],
        "Contract": ["Month-to-month", "Two year", "Month-to-month"],
        "PaymentMethod": ["Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"],
    }
    for c in COLUNAS_SERVICO:
        d[c] = ["No", "Yes", "No"]
    return pd.DataFrame(d)


def test_ticket_medio_calcula_e_trata_tenure_zero():
    out = add_ticket_medio(df_base())
    assert out["ticket_medio_mensal"].tolist() == [0.0, 80.0, 70.0]


def test_diferenca_cobranca():
    out = add_diferenca_cobranca(add_ticket_medio(df_base()))
    assert out["diferenca_cobranca"].tolist() == [50.0, 20.0, 0.0]


def test_qtd_servicos_conta_yes():
    out = add_qtd_servicos(df_base())
    assert out["qtd_servicos"].tolist() == [0, len(COLUNAS_SERVICO), 0]


def test_flags():
    out = add_flags(df_base())
    assert out["cliente_novo"].tolist() == [1, 0, 1]
    assert out["contrato_mensal"].tolist() == [1, 0, 1]
    assert out["pagamento_automatico"].tolist() == [0, 1, 1]


def test_funcoes_nao_alteram_o_original():
    df = df_base()
    colunas = df.columns.tolist()
    criar_features(df)
    assert df.columns.tolist() == colunas


def test_criar_features_gera_as_6_colunas_sem_nulos():
    out = criar_features(df_base())
    novas = ["ticket_medio_mensal", "diferenca_cobranca", "qtd_servicos",
             "cliente_novo", "contrato_mensal", "pagamento_automatico"]
    assert all(c in out.columns for c in novas)
    assert out[novas].isna().sum().sum() == 0