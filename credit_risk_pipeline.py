"""
Abstract:
This Python script implements a complete credit risk prediction pipeline using the Home Credit Default Risk dataset. The script performs the following key steps:

1. **Data Preprocessing**: Handles missing values using median imputation for numerical features and most frequent imputation for categorical features.
2. **Feature Engineering**: Applies one-hot encoding for categorical attributes and ordinal encoding for sequential attributes.
3. **Model Training**: Trains and evaluates three models - Logistic Regression, Random Forest, and XGBoost.
4. **Hyperparameter Tuning**: Uses GridSearchCV to optimize the Random Forest model.
5. **Pipeline Implementation**: Combines all preprocessing and modeling steps into a scikit-learn Pipeline for streamlined execution.
6. **Results Visualization**: Saves evaluation metrics, generates confusion matrices, and visualizes prediction results.

The goal of this script is to predict whether an applicant will default on a loan, supporting risk assessment in financial institutions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
data = pd.read_csv("https://storage.googleapis.com/kaggle-competitions-data/kaggle/6362/application_train.csv.zip", compression='zip')

# Define categorical and ordinal columns
categorical_cols = ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START']
ordinal_cols = ['NAME_EDUCATION_TYPE', 'REGION_RATING_CLIENT_W_CITY']
numerical_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH']

data['DAYS_BIRTH'] = abs(data['DAYS_BIRTH']) // 365  # Convert days to years

target = 'TARGET'

# Handle missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
ord_imputer = SimpleImputer(strategy='most_frequent')

# Encoding
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
ordinal_encoder = OrdinalEncoder()

# Standardization
scaler = StandardScaler()

# Define column transformer
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', num_imputer), ('scaler', scaler)]), numerical_cols),
    ('cat', Pipeline([('imputer', cat_imputer), ('onehot', one_hot_encoder)]), categorical_cols),
    ('ord', Pipeline([('imputer', ord_imputer), ('encoder', ordinal_encoder)]), ordinal_cols)  
])


# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=[target]), data[target], test_size=0.2, random_state=42, stratify=data[target]
)

# Evaluate models and save results
def evaluate_model(model, name):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Save classification report
    report = classification_report(y_test, y_pred)
    with open(f'{name}_classification_report.txt', 'w') as f:
        f.write(report)
    
    # Save model
    joblib.dump(pipeline, f'{name}_model.pkl')
    
    # Print evaluation
    print(f"{name} Performance:")
    print(report)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("-" * 50)
    
    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.show()
    
    # Save and visualize prediction results
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results_df.to_csv(f'{name}_predictions.csv', index=False)
    
    plt.figure(figsize=(8, 5))
    sns.histplot(results_df, x='Predicted', hue='Actual', multiple='stack', bins=3, palette='coolwarm')
    plt.xlabel('Predicted Default Probability')
    plt.ylabel('Count')
    plt.title(f'Prediction Distribution - {name}')
    plt.savefig(f'{name}_prediction_distribution.png')
    plt.show()

for name, model in models.items():
    evaluate_model(model, name)

# Hyperparameter tuning for Random Forest
param_grid = {'classifier__n_estimators': [50, 100, 200]}
grid_search = GridSearchCV(Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
]), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best Random Forest Parameters:", grid_search.best_params_)

# Save best model
joblib.dump(grid_search.best_estimator_, 'Best_RandomForest_Model.pkl')

