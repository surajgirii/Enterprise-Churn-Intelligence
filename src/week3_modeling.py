import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# 1. Load Cleaned Dataset
df = pd.read_csv('data/processed/cleaned_telco_churn.csv')

# 2. Prepare Features (X) and Target (y)
df['Churn_Binary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

X = df.drop(columns=['customerID', 'Churn', 'Churn_Binary', 'TenureCohort'])
y = df['Churn_Binary']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 3. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# 4. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Model 1: Logistic Regression (Balanced)
pipe_lr = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
pipe_lr.fit(X_train, y_train)

# 6. Model 2: Random Forest (Balanced)
pipe_rf = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
])
pipe_rf.fit(X_train, y_train)

# 7. Print Model Performance Metrics
y_pred_lr = pipe_lr.predict(X_test)
y_prob_lr = pipe_lr.predict_proba(X_test)[:, 1]

print("=== LOGISTIC REGRESSION METRICS ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr) * 100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_lr), 4))
print(classification_report(y_test, y_pred_lr, digits=4))

# 8. Save Trained Model Pipeline Artifact
joblib.dump(pipe_lr, 'reports/churn_model.pkl')
print("\nTrained model saved to 'reports/churn_model.pkl'.")