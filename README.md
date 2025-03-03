# Credit Risk Modeling Pipeline

## Overview

This repository contains a Python pipeline for credit risk prediction using machine learning. The project is based on the Home Credit Default Risk dataset and focuses on assessing the likelihood of a loan applicant defaulting. The pipeline includes data preprocessing, feature engineering, model training, evaluation, and result visualization.

## Features

- **Data Preprocessing**: Handles missing values, normalizes numerical data, and encodes categorical and ordinal attributes.
- **Feature Engineering**: Selects relevant features for credit risk modeling.
- **Machine Learning Models**: Implements Logistic Regression, Random Forest, and XGBoost.
- **Hyperparameter Tuning**: Uses GridSearchCV to optimize the best model.
- **Visualization**: Generates confusion matrices and prediction distribution plots.
- **Model Persistence**: Saves trained models and predictions for further analysis.

## Dataset

- **Source**: [Kaggle - Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- **Target Variable**: `TARGET` (1 = Default, 0 = No Default)

### Key Features

#### Numerical Features:
- `AMT_INCOME_TOTAL`: Total income of the applicant.
- `AMT_CREDIT`: Total credit amount of the loan.
- `DAYS_BIRTH`: Age of the applicant in days (negative values).

#### Categorical Features:
- `NAME_INCOME_TYPE`: Type of income source (e.g., working, pensioner).
- `OCCUPATION_TYPE`: Type of occupation.

#### Ordinal Features:
- `NAME_EDUCATION_TYPE`: Level of education (e.g., secondary, higher).
- `REGION_RATING_CLIENT_W_CITY`: Rating of the region where the client lives.

## Installation

### Prerequisites

Ensure you have Python installed (>=3.7) and install the required dependencies:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

## Usage

- **Clone the repository**:

git clone https://github.com/your-username/credit-risk-modeling.git
cd credit-risk-modeling

- **Run the script**:

python credit_risk_pipeline.py

- **Outputs**:

1. Classification Reports (*_classification_report.txt)

2. Confusion Matrices (*_confusion_matrix.png)

3. Prediction Results (*_predictions.csv)

4. Trained Models (*_model.pkl)

## Results

- **Accuracy & Performance Metrics**: The model performance is evaluated using accuracy, precision, recall, and F1-score.

- **Prediction Visualization**: Histogram plots show the distribution of predicted defaults vs. actual outcomes.

## Future Improvements

- Implement feature selection for better model performance.

- Explore additional models (e.g., Neural Networks, Gradient Boosting).

- Integrate real-world credit risk data.

## License

This project is licensed under the MIT License.

## Author: 
Xikun Jiang
For inquiries, contact: xikunjiang@163.com
