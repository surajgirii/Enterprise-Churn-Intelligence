import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Cleaned Data
df = pd.read_csv('data/processed/cleaned_telco_churn.csv')

# 2. Print Summary Statistics
print("=== SUMMARY STATISTICS ===")
print("Total Customers:", len(df))
print("Overall Churn Rate:", round((df['Churn'] == 'Yes').mean() * 100, 2), "%")
print("Total Monthly Revenue: $", round(df['MonthlyCharges'].sum(), 2))
print("Monthly Revenue Lost to Churn: $", round(df[df['Churn'] == 'Yes']['MonthlyCharges'].sum(), 2))

# 3. Print Breakdown Tables
print("\n=== CHURN BY CONTRACT ===")
print(round(pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100, 2))

print("\n=== CHURN BY INTERNET SERVICE ===")
print(round(pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100, 2))

# 4. Generate & Save Charts
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Contract', hue='Churn', palette='Set1')
plt.title('Churn Count by Contract Type')

plt.subplot(1, 2, 2)
sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, palette='Set1')
plt.title('Customer Tenure Distribution')

plt.tight_layout()
plt.savefig('reports/eda_charts.png')
print("\nCharts successfully saved as 'reports/eda_charts.png'.")