# Revenue Forecast 7D

A comprehensive data science project for predicting 7-day user revenue combining SQL-based session analytics with machine learning forecasting.

## 📊 Project Overview

This project demonstrates end-to-end data processing and predictive modeling:
- **SQL Analytics**: Efficient session duration calculations using BigQuery
- **ML Pipeline**: Two-stage Hurdle model with CatBoost for revenue prediction
- **Data Engineering**: Feature extraction from user behavior logs
- **Model Comparison**: Benchmarking multiple algorithms (CatBoost, XGBoost, LightGBM, Random Forest)

## 📁 Repository Structure

```
├── src/
│   ├── sql/
│   │   ├── query.sql        # Session duration calculation query
│   │   └── README.md        # SQL design decisions & optimizations
│   └── ipynb/
│       ├── model.ipynb      # Main EDA, feature engineering & model training
│       └── comparison_of_models.ipynb  # Algorithm benchmarking
├── data/
│   ├── final_data.csv       # Generated after feature engineering
│   └── README.md            # Data dictionary
├── requirements.txt
└── README.md
```

## 🔧 Tech Stack

- **SQL**: BigQuery with window functions for efficient analytics
- **Python**: Pandas, Scikit-learn, CatBoost, XGBoost, LightGBM
- **Validation**: TimeSeriesSplit to prevent data leakage
- **Encoding**: One-Hot Encoding (low cardinality) + Frequency Encoding (high cardinality)

## 💡 Key Insights & Decisions

### SQL Session Analytics
- **Window Functions**: Used `LEAD()` to avoid costly JOINs and compute session close times efficiently
- **Data Validation**: Filters for malformed sessions (e.g., consecutive `open` events without `close`)
- **BigQuery Optimization**: Pre-filters data with `INTERVAL 11 DAY` to reduce table scans
- **Assumption**: Sessions crossing midnight are counted in full toward their start date

### Machine Learning Architecture

The core challenge: ~98% of users generate zero revenue (extreme class imbalance).

**Solution: Two-Stage Hurdle Model**
1. **Classification Stage**: Predicts probability of user making any purchase
2. **Regression Stage**: Trained only on buyers; predicts purchase amount
3. **Final Prediction**: `P(buyer) × Predicted_Amount`

### Model Performance

Tested multiple algorithms on feature-engineered dataset:

| Algorithm | MAE | Notes |
|-----------|-----|-------|
| **CatBoost** | ~0.94 | ✅ Best balance of classification & regression accuracy |
| LightGBM | ~1.15 | Close second, faster training |
| XGBoost | ~1.15 | Similar to LightGBM |
| Random Forest | Higher | Baseline reference |

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Chuchman-Dima/Revenue-Forecast-7d.git
cd Revenue-Forecast-7d
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data CSV files in `data/`:
   - `task_2_users_params.csv` — User parameters + revenue target
   - `task_2_users_actions.csv` — User actions from day 1

   (See [data/README.md](data/README.md) for full details)

### Running the Pipeline

Execute notebooks in order:

```bash
jupyter notebook src/ipynb/model.ipynb
```
This notebook handles:
- Exploratory Data Analysis (EDA)
- Feature engineering
- Hurdle model training
- Generates `data/final_data.csv`

```bash
jupyter notebook src/ipynb/comparison_of_models.ipynb
```
This notebook:
- Loads pre-engineered `final_data.csv`
- Benchmarks multiple algorithms
- Compares performance metrics

## 📈 Future Improvements

For production deployment:

1. **Log-space Math**: Apply proper inverse transformation with `np.expm1()` when converting predictions from log scale
2. **Dynamic Threshold**: Optimize classification threshold based on business KPIs rather than default 0.5
3. **Feature Monitoring**: Track feature distribution drift in production
4. **Model Versioning**: Implement MLflow or similar for experiment tracking
5. **Cross-validation**: Expand testing beyond TimeSeriesSplit to multiple validation strategies

## 📝 Design Notes

- **Cardinality Handling**: High-cardinality features (`country`, `device_model`) use frequency encoding with strict train-set mapping to prevent data leakage
- **Temporal Validation**: TimeSeriesSplit preserves temporal ordering of data
- **Missing Data Strategy**: See individual notebooks for handling approach

## 📄 License

Personal project

---

**Author**: Chuchman-Dima  
**Last Updated**: 2026
