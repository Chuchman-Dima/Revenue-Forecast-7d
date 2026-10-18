"""Підготовка даних: завантаження, очищення, ознаки, розбиття на train / valid / test.

Використовуємо лише те, що відоме після першого дня життя користувача:
параметри реєстрації + дії за перші 24 години (4 інтервали по 6 годин).
"""
from pathlib import Path

import pandas as pd

TARGET = "revenue_7d"

# Ці колонки — мітки, а не числа, тому їх треба трактувати як категорії
CAT_COLS = [
    "traffic_type_id", "traffic_group_id", "gender", "country",
    "device_browser", "device_brand", "device_model", "device_platform", "device_os",
]
# Колонки з великою кількістю унікальних значень — рідкісні значення об'єднаємо в "rare"
HIGH_CARD_COLS = ["country", "device_brand", "device_model"]


def load_data(data_dir):
    data_dir = Path(data_dir)
    params = pd.read_csv(data_dir / "task_2_users_params.csv")
    actions = pd.read_csv(data_dir / "task_2_users_actions.csv")
    return params, actions


def clean_data(params, actions):
    """Перетворює дати та видаляє дублікати."""
    params = params.copy()
    actions = actions.copy()

    params["timestamp_reg"] = pd.to_datetime(params["timestamp_reg"], format="mixed")
    actions["timestamp_interval_start"] = pd.to_datetime(actions["timestamp_interval_start"], format="mixed")
    actions["timestamp_interval_end"] = pd.to_datetime(actions["timestamp_interval_end"], format="mixed")

    actions = actions.drop_duplicates()
    params = params.sort_values("timestamp_reg").drop_duplicates(subset="id_user", keep="first")
    return params.reset_index(drop=True), actions.reset_index(drop=True)


def make_action_features(actions):
    """Зводимо 4 інтервали кожного користувача в один рядок."""
    actions = actions.sort_values(["id_user", "timestamp_interval_start"]).copy()
    actions["interval_number"] = actions.groupby("id_user").cumcount()  # 0, 1, 2, 3
    actions["has_payment"] = actions["cnt_payments"] > 0
    actions["has_visit"] = actions["cnt_visit_other_users"] > 0

    grouped = actions.groupby("id_user")
    feats = grouped.agg(
        d1_sum_payments=("sum_payments", "sum"),
        d1_max_payment=("sum_payments", "max"),
        d1_cnt_payments=("cnt_payments", "sum"),
        d1_credits=("sum_credits_spend", "sum"),
        d1_returns=("cnt_returns", "sum"),
        d1_visits=("cnt_visit_other_users", "sum"),
        d1_max_visits=("cnt_visit_other_users", "max"),
        d1_active_intervals=("has_visit", "sum"),
        d1_pay_intervals=("has_payment", "sum"),
    )

    # У якому інтервалі була перша оплата (4 = оплат не було)
    first_payment = actions[actions["has_payment"]].groupby("id_user")["interval_number"].min()
    feats["d1_first_pay_interval"] = first_payment.reindex(feats.index).fillna(4)

    # Скільки візитів було в перші 6 годин
    first_interval = actions[actions["interval_number"] == 0].set_index("id_user")
    feats["d1_visits_first_interval"] = first_interval["cnt_visit_other_users"].reindex(feats.index).fillna(0)

    # Похідні ознаки (ділимо на NaN замість 0, щоб не отримати inf)
    payments_no_zero = feats["d1_cnt_payments"].replace(0, float("nan"))
    feats["d1_credits_per_payment"] = feats["d1_credits"] / payments_no_zero
    feats["d1_avg_payment"] = feats["d1_sum_payments"] / payments_no_zero
    feats["d1_is_payer"] = (feats["d1_cnt_payments"] > 0).astype(int)
    return feats.reset_index()


def build_dataset(params, actions):
    """Об'єднує params та ознаки з actions, додає час реєстрації як ознаки."""
    action_feats = make_action_features(actions)
    df = params.merge(action_feats, on="id_user", how="left")

    # Якщо в actions немає жодного рядка — активності не було, тож ставимо 0
    df["has_actions"] = df["d1_sum_payments"].notna().astype(int)
    keep_nan = ["d1_credits_per_payment", "d1_avg_payment", "d1_first_pay_interval"]
    fill_zero = [c for c in action_feats.columns if c.startswith("d1_") and c not in keep_nan]
    df[fill_zero] = df[fill_zero].fillna(0)
    df["d1_first_pay_interval"] = df["d1_first_pay_interval"].fillna(4)

    df["reg_hour"] = df["timestamp_reg"].dt.hour
    df["reg_dayofweek"] = df["timestamp_reg"].dt.dayofweek
    df["reg_is_weekend"] = (df["reg_dayofweek"] >= 5).astype(int)

    # Категорії: ID каналів → текст, пропуски → "unknown"
    for col in ["traffic_type_id", "traffic_group_id"]:
        df[col] = df[col].astype("Int64").astype(str)
    text_cols = [c for c in CAT_COLS if c not in ["traffic_type_id", "traffic_group_id"]]
    df[text_cols] = df[text_cols].astype("object").fillna("unknown").astype(str)

    # Ознаки, які в усіх користувачів однакові (наприклад cnt_returns), нічого не дають
    constant_cols = [c for c in df.columns if c.startswith("d1_") and df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant_cols)

    df = df.sort_values("timestamp_reg").reset_index(drop=True)
    return df, constant_cols


def split_by_time(df, train_share=0.6, valid_share=0.2):
    """Розбиття за часом: найраніші користувачі — train, найпізніші — test. Без shuffle."""
    n = len(df)
    train_end = int(n * train_share)
    valid_end = int(n * (train_share + valid_share))
    return df.iloc[:train_end].copy(), df.iloc[train_end:valid_end].copy(), df.iloc[valid_end:].copy()


def prepare_categories(train, valid, test, min_count=200):
    """Рідкісні значення (менше min_count у train) → "rare"; всі набори отримують однаковий список категорій."""
    sets = [train.copy(), valid.copy(), test.copy()]

    for col in HIGH_CARD_COLS:
        counts = sets[0][col].value_counts()
        frequent = set(counts[counts >= min_count].index)
        for d in sets:
            d[col] = d[col].where(d[col].isin(frequent), "rare")

    for col in CAT_COLS:
        categories = sorted(sets[0][col].unique().tolist()) + ["__unseen__"]
        for d in sets:
            d[col] = pd.Categorical(d[col], categories=categories)
            d[col] = d[col].fillna("__unseen__")
    return sets


def get_feature_columns(df):
    not_features = ["id_user", "timestamp_reg", TARGET]
    return [c for c in df.columns if c not in not_features]


def prepare_data(data_dir, min_count=200):
    """Весь пайплайн одним викликом: CSV → train, valid, test, список ознак."""
    params, actions = load_data(data_dir)
    params, actions = clean_data(params, actions)
    df, _ = build_dataset(params, actions)
    train, valid, test = split_by_time(df)
    train, valid, test = prepare_categories(train, valid, test, min_count=min_count)
    return train, valid, test, get_feature_columns(train)
