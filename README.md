# 🔮 Enterprise Churn Intelligence & Strategic Analytics Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning, Data Analytics Engineering, and Data Governance solution designed to predict customer attrition, safeguard Monthly Recurring Revenue (MRR), and establish an enterprise data management strategy for subscription-based telecommunications services.

---

## 📌 Executive Summary & Business Impact

Customer attrition poses a direct threat to recurring revenue models. Across a portfolio of **7,043 customer accounts**, historical data identified an active **26.54% churn rate**, putting **$139,130.85 in Monthly Recurring Revenue (MRR)** ($1.66M+ annualized exposure) at risk.

### Strategic Outcomes Delivered
* **Predictive ML Modeling:** Trained a class-balanced **Logistic Regression Pipeline** achieving an **0.8419 ROC-AUC Score** and a **78.88% Churn Recall**, capturing nearly 4 out of 5 churners prior to cancellation.
* **Interactive UI Scoring:** Deployed a real-time Streamlit web app allowing customer success teams to simulate risk profiles during live retention calls.
* **Executive Analytics & Visuals:** Generated automated visual charts detailing contract risk profiles, early-tenure drop-off density, and service-payment channel risk heatmaps.
* **Data Governance & Quality:** Built an automated data quality audit engine validating schema compliance, boundary rules, and portfolio KPIs.

---

## 🛠️ Implementation Architecture

| Lifecycle Phase | Script / Component | Outputs & Key Technical Deliverables |
| :--- | :--- | :--- |
| **Phase 1: Ingestion & Cleaning** | `src/week1_cleaning.py` | Type conversion, missing value handling, generated `cleaned_telco_churn.csv`. |
| **Phase 2: Exploratory Analytics** | `src/week2_eda.py` | Discovered month-to-month contracts contribute over **88%** of total churned revenue. |
| **Phase 3: Machine Learning & UI** | `src/week3_modeling.py`<br>`src/week3_app.py` | Trained Logistic Regression (`ROC-AUC: 0.8419`, `Recall: 78.88%`). Launched Streamlit scoring UI. |
| **Phase 4: Visual Analytics** | `src/week4_visuals.py` | Exported publication-ready charts (`mrr_risk_by_contract.png`, `tenure_churn_density.png`). |
| **Phase 5: Strategy & Governance** | `src/week5_data_strategy.py` | Automated schema validation (`data_quality_audit.json`) and strategic KPI reporting. |

---

## 🤖 Machine Learning Performance Benchmark

Models were trained to handle class imbalance (73% non-churn vs. 27% churn). Because failing to flag a churning customer (False Negative) results in total customer lifetime value loss, **Recall** was designated as the primary evaluation metric.

| Model Architecture | Accuracy | ROC-AUC | Churn Recall | Churn Precision | F1-Score | Production Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Class-Weighted)** | **74.10%** | **0.8419** | **78.88%** | **50.77%** | **0.6178** | **Selected Production Pipeline** |
| **Random Forest Classifier** | 78.28% | 0.8250 | 47.86% | 61.72% | 0.5392 | Baseline Benchmark |

---

## 💡 Key Strategic Recommendations

* 1. Targeted Contract Upgrades: Offer targeted $5/month billing credits to month-to-month subscribers before month 6 to convert them into 1-Year locked contracts.

* 2. Automated Payment Migration: Incentivize customers on manual Electronic Check payments with a one-time credit to transition to automatic ACH bank transfers or credit card billing.

* 3. Fiber Optic Support Bundling: Automatically bundle 90 days of complimentary Tech Support with new Fiber Optic installations to reduce high early-tenure drop-off.

---

## 📂 Project Architecture

```text
Enterprise Churn Intelligence/
├── data/
│   ├── processed/
│   │   └── cleaned_telco_churn.csv            # Standardized & preprocessed dataset
│   └── raw/
│       └── raw_telco_churn.csv                # Raw ingested source data
├── reports/
│   ├── figures/
│   │   ├── mrr_risk_by_contract.png           # Revenue risk by contract type
│   │   ├── service_risk_heatmap.png           # Service & payment method risk matrix
│   │   └── tenure_churn_density.png           # Lifespan attrition risk distribution
│   ├── churn_model.pkl                        # Serialized Scikit-Learn Pipeline
│   ├── data_quality_audit.json                # Automated governance audit results
│   ├── eda_charts.png                         # Exploratory data analysis visuals
│   ├── strategic_kpi_summary.json             # Executive revenue & portfolio KPIs
│   ├── WEEK 1 Report.docx                     # Data ingestion & cleaning documentation
│   ├── Week 2 Report.docx                     # Exploratory data analysis report
│   ├── Week 3 Report.docx                     # Machine learning & modeling report
│   ├── Week 4.docx                            # Executive visuals & narrative report
│   └── Week 5 Strategic Report.docx           # Enterprise data strategy & governance report
├── src/
│   ├── week1_cleaning.py                      # Automated data cleaning & type casting
│   ├── week2_eda.py                           # Statistical analysis & EDA chart generator
│   ├── week3_modeling.py                      # ML training, evaluation, & serialization
│   ├── week3_app.py                           # Interactive Streamlit application
│   ├── week4_visuals.py                       # High-res executive figure pipeline
│   └── week5_data_strategy.py                 # Automated quality audit & KPI engine
├── .gitignore                                 # Git exclusion settings
└── README.md                                  # Production repository documentation
