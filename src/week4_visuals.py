"""
Week 4: Executive Data Visualization & Reporting Engine
Author: Senior Data Analytics Engineer
Description: Generates publication-ready visualizations and statistical charts
             for executive stakeholders to visualize churn risk and MRR exposure.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging framework
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class ExecutiveVisualizer:
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        self._apply_custom_theme()

    def _apply_custom_theme(self):
        """Sets enterprise-grade Matplotlib & Seaborn styling."""
        sns.set_theme(style="white", palette="muted")
        plt.rcParams.update({
            'font.sans-serif': 'Segoe UI',
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'axes.labelsize': 11,
            'axes.labelweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.autolayout': True
        })

    def load_and_validate_data(self) -> bool:
        """Loads dataset and performs structural checks."""
        try:
            logging.info(f"Loading data from {self.data_path}...")
            self.df = pd.read_csv(self.data_path)
            required_cols = ['Contract', 'MonthlyCharges', 'Churn', 'tenure', 'PaymentMethod', 'InternetService']
            
            missing = [col for col in required_cols if col not in self.df.columns]
            if missing:
                logging.error(f"Missing required columns for visualization: {missing}")
                return False
                
            logging.info("Dataset successfully validated for executive reporting.")
            return True
        except Exception as e:
            logging.error(f"Failed to load dataset: {str(e)}")
            return False

    def generate_mrr_impact_chart(self):
        """Chart 1: Total Monthly Recurring Revenue Risk by Contract Type."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mrr_summary = self.df.groupby(['Contract', 'Churn'])['MonthlyCharges'].sum().reset_index()
        
        colors = {'No': '#2b5c8f', 'Yes': '#d9534f'}
        barplot = sns.barplot(
            data=mrr_summary, x='Contract', y='MonthlyCharges', hue='Churn',
            palette=colors, ax=ax, edgecolor='none'
        )
        
        # Annotate exact dollar amounts on bars
        for p in barplot.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'${height:,.0f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            fontsize=9, color='#333333', xytext=(0, 3),
                            textcoords='offset points', weight='bold')

        ax.set_title('Total Monthly Revenue Exposure by Contract Strategy', pad=20)
        ax.set_xlabel('Customer Contract Type')
        ax.set_ylabel('Total Monthly Recurring Revenue ($)')
        ax.yaxis.set_major_formatter('${x:,.0f}')
        
        # Format legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=['Retained Account', 'Churned Account'], title='Status', frameon=False)
        
        output_path = os.path.join(self.output_dir, 'mrr_risk_by_contract.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {output_path}")

    def generate_tenure_density_plot(self):
        """Chart 2: Kernel Density Estimation of Tenure vs Churn Status."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.kdeplot(
            data=self.df[self.df['Churn'] == 'No']['tenure'], 
            label='Retained Customer Cohort', color='#2b5c8f', 
            fill=True, alpha=0.4, linewidth=2, ax=ax
        )
        sns.kdeplot(
            data=self.df[self.df['Churn'] == 'Yes']['tenure'], 
            label='Churned Customer Cohort', color='#d9534f', 
            fill=True, alpha=0.4, linewidth=2, ax=ax
        )

        ax.axvline(x=12, color='#777777', linestyle='--', linewidth=1.5, label='12-Month Critical Risk Threshold')
        
        ax.set_title('Customer Retention vs. Attrition Risk Curve Across Lifespan', pad=20)
        ax.set_xlabel('Tenure Length (Months)')
        ax.set_ylabel('Customer Density')
        ax.set_xlim(0, 72)
        ax.legend(frameon=False, loc='upper right')
        
        output_path = os.path.join(self.output_dir, 'tenure_churn_density.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {output_path}")

    def generate_service_risk_matrix(self):
        """Chart 3: Heatmap Matrix showing Churn Rate % by Service Combinations."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        self.df['Churn_Numeric'] = self.df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        pivot_table = self.df.pivot_table(
            index='InternetService', 
            columns='PaymentMethod', 
            values='Churn_Numeric', 
            aggfunc='mean'
        ) * 100

        sns.heatmap(
            pivot_table, annot=True, fmt=".1f", cmap="Reds", 
            cbar_kws={'label': 'Churn Probability (%)'}, ax=ax,
            linewidths=1, linecolor='white'
        )

        ax.set_title('Service Infrastructure vs. Payment Channel Risk Heatmap (%)', pad=20)
        ax.set_xlabel('Payment Method Type')
        ax.set_ylabel('Internet Service Type')
        plt.xticks(rotation=15, ha='right')

        output_path = os.path.join(self.output_dir, 'service_risk_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {output_path}")

    def run_pipeline(self):
        """Executes full visualization suite."""
        if self.load_and_validate_data():
            self.generate_mrr_impact_chart()
            self.generate_tenure_density_plot()
            self.generate_service_risk_matrix()
            logging.info("All executive figures generated cleanly.")

if __name__ == '__main__':
    DATA_FILE = 'data/processed/cleaned_telco_churn.csv'
    FIGURES_DIR = 'reports/figures'
    
    visualizer = ExecutiveVisualizer(data_path=DATA_FILE, output_dir=FIGURES_DIR)
    visualizer.run_pipeline()