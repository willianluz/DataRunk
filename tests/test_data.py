import pandas as pd
import pytest

from src.data import validar_dados, limpar_dados, dividir_treino_teste


def df_exemplo():
    return pd.DataFrame({
        "customerID": ["a", "b", "c", "d"],
        "tenure": [0, 10, 20, 30],
        "MonthlyCharges": [50.0, 60.0, 70.0, 80.0],
        "TotalCharges": [" ", "600", "1400", "2400"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "Churn": ["No", "Yes", "No", "Yes"],
    })


def test_limpar_converte_total_charges_para_numero():
    out = limpar_dados(df_exemplo())
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])


def test_limpar_preenche_cliente_novo_com_zero():
    out = limpar_dados(df_exemplo())
    assert out.loc[0, "TotalCharges"] == 0.0


def test_limpar_transforma_churn_em_0_e_1():
    out = limpar_dados(df_exemplo())
    assert out["Churn"].tolist() == [0, 1, 0, 1]


def test_limpar_nao_altera_o_original():
    df = df_exemplo()
    limpar_dados(df)
    assert df["Churn"].tolist() == ["No", "Yes", "No", "Yes"]


def test_validar_falha_com_coluna_ausente():
    df = df_exemplo().drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Colunas ausentes"):
        validar_dados(df)


def test_validar_falha_com_id_duplicado():
    df = df_exemplo()
    df.loc[1, "customerID"] = "a"
    with pytest.raises(ValueError, match="duplicado"):
        validar_dados(df)


def test_split_estratificado_mantem_proporcao_e_remove_id():
    df = pd.DataFrame({
        "customerID": [str(i) for i in range(100)],
        "x": range(100),
        "Churn": [1] * 20 + [0] * 80,
    })
    X_train, X_test, y_train, y_test = dividir_treino_teste(df, "Churn", 0.2, 42)
    assert "customerID" not in X_train.columns
    assert "Churn" not in X_train.columns
    assert len(X_test) == 20
    assert y_train.mean() == pytest.approx(0.2, abs=0.01)
    assert y_test.mean() == pytest.approx(0.2, abs=0.01)