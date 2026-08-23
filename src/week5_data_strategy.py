"""
Week 5: Enterprise Data Strategy & Quality Governance Engine
Author: Senior Data Analytics Engineer
Description: Programmatically enforces data quality rules, calculates strategic business 
             metrics, and generates governance reports for enterprise data pipelines.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np

# Configure enterprise logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class DataStrategyEngine:
    def __init__(self, raw_data_path: str, output_dir: str):
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        self.df = None
        self.quality_report = {}
        self.strategic_kpis = {}
        
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> bool:
        """Loads and prepares data for strategic validation."""
        try:
            logging.info(f"Ingesting dataset from {self.raw_data_path}...")
            self.df = pd.read_csv(self.raw_data_path)
            logging.info(f"Successfully loaded {len(self.df)} records.")
            return True
        except Exception as e:
            logging.error(f"Data ingestion failed: {str(e)}")
            return False

    def validate_data_governance(self):
        """Executes automated Data Quality & Governance checks."""
        logging.info("Executing Data Governance Rules & Quality Checks...")
        
        total_records = len(self.df)
        
        # 1. Null Value Audit
        null_counts = {str(k): int(v) for k, v in self.df.isnull().sum().to_dict().items()}
        
        # 2. Schema Consistency Check
        expected_dtypes = {
            'tenure': ['int64', 'float64'],
            'MonthlyCharges': ['float64'],
            'Contract': ['object'],
            'Churn': ['object']
        }
        
        schema_violations = {}
        for col, expected in expected_dtypes.items():
            actual = str(self.df[col].dtype)
            if actual not in expected:
                schema_violations[col] = f"Expected {expected}, got {actual}"

        # 3. Value Range & Boundary Validation
        invalid_charges = int((self.df['MonthlyCharges'] < 0).sum())
        invalid_tenure = int((self.df['tenure'] < 0).sum())

        self.quality_report = {
            "total_records_processed": total_records,
            "null_value_summary": null_counts,
            "schema_violations": schema_violations,
            "data_anomaly_count": {
                "negative_charges": invalid_charges,
                "negative_tenure": invalid_tenure
            },
            "data_completeness_score": f"{((total_records - sum(null_counts.values())) / total_records) * 100:.2f}%"
        }
        
        # Save JSON Quality Audit artifact
        audit_file = os.path.join(self.output_dir, 'data_quality_audit.json')
        with open(audit_file, 'w') as f:
            json.dump(self.quality_report, f, indent=4)
        
        logging.info(f"Data Quality Audit saved to {audit_file}")

    def calculate_strategic_business_metrics(self):
        """Calculates executive-level KPIs for enterprise decision-making."""
        logging.info("Calculating Strategic KPIs & Revenue Exposure...")
        
        total_customers = len(self.df)
        churned_customers = len(self.df[self.df['Churn'] == 'Yes'])
        churn_rate = (churned_customers / total_customers) * 100
        
        total_mrr = self.df['MonthlyCharges'].sum()
        at_risk_mrr = self.df[self.df['Churn'] == 'Yes']['MonthlyCharges'].sum()
        
        # High-Risk Segment Identification
        m2m_churn = self.df[(self.df['Contract'] == 'Month-to-month') & (self.df['Churn'] == 'Yes')]
        m2m_revenue_loss = m2m_churn['MonthlyCharges'].sum()

        self.strategic_kpis = {
            "total_active_subscribers": total_customers,
            "baseline_churn_rate_pct": round(churn_rate, 2),
            "total_monthly_recurring_revenue": round(float(total_mrr), 2),
            "at_risk_monthly_recurring_revenue": round(float(at_risk_mrr), 2),
            "m2m_contract_revenue_exposure": round(float(m2m_revenue_loss), 2),
            "annualized_churn_revenue_loss": round(float(at_risk_mrr * 12), 2)
        }
        
        # Save JSON KPI artifact
        kpi_file = os.path.join(self.output_dir, 'strategic_kpi_summary.json')
        with open(kpi_file, 'w') as f:
            json.dump(self.strategic_kpis, f, indent=4)
            
        logging.info(f"Strategic KPI Summary saved to {kpi_file}")

    def run(self):
        """Executes full Data Strategy pipeline."""
        if self.load_data():
            self.validate_data_governance()
            self.calculate_strategic_business_metrics()
            logging.info("Week 5 Data Strategy Code Pipeline Execution Complete.")

if __name__ == '__main__':
    DATA_PATH = 'data/processed/cleaned_telco_churn.csv'
    OUTPUT_PATH = 'reports'
    
    engine = DataStrategyEngine(raw_data_path=DATA_PATH, output_dir=OUTPUT_PATH)
    engine.run()