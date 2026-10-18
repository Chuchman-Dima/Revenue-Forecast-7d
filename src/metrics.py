"""Метрики для прогнозу виручки."""
import numpy as np
import pandas as pd


def gini(y_true, y_pred):
    """Якість ранжування: 1 — ідеально, 0 — випадково. Показує, чи піднімає модель платників угору."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    def raw_gini(y, order_by):
        order = np.argsort(-order_by, kind="mergesort")
        cumulative = np.cumsum(y[order]) / y.sum()
        return cumulative.sum() / len(y) - (len(y) + 1) / (2 * len(y))

    return raw_gini(y_true, y_pred) / raw_gini(y_true, y_true)


def top_share(y_true, y_pred, fraction=0.05):
    """Яка частка реальної виручки припадає на топ-5% користувачів за прогнозом."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    k = max(1, int(len(y_true) * fraction))
    top_idx = np.argsort(-y_pred)[:k]
    return y_true[top_idx].sum() / y_true.sum()


def evaluate(y_true, y_pred, name="model"):
    """MAE, RMSE, зсув сумарної виручки, Gini та Top-5% share."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return pd.Series({
        "MAE": np.abs(y_true - y_pred).mean(),
        "RMSE": np.sqrt(((y_true - y_pred) ** 2).mean()),
        "Bias, %": 100 * (y_pred.sum() - y_true.sum()) / y_true.sum(),
        "Gini": gini(y_true, y_pred),
        "Top-5% share": top_share(y_true, y_pred),
    }, name=name)


def cohort_report(df, y_pred, by, min_users=500):
    """Порівнює реальну та прогнозовану сумарну виручку по групах (когортах).

    Повертає таблицю по когортах і WAPE — загальну відносну помилку, %.
    """
    data = df[["revenue_7d"]].copy()
    data["pred"] = np.asarray(y_pred, dtype=float)
    for col in ([by] if isinstance(by, str) else by):
        data[col] = df[col].values

    table = data.groupby(by, observed=True).agg(
        users=("revenue_7d", "size"), actual=("revenue_7d", "sum"), pred=("pred", "sum"))
    table = table[table["users"] >= min_users]
    table["error_%"] = 100 * (table["pred"] - table["actual"]) / table["actual"]
    wape = 100 * (table["pred"] - table["actual"]).abs().sum() / table["actual"].sum()
    return table, wape
