# Predicting Airbnb Listing Prices with MLflow and AWS S3

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting Airbnb listing prices in New York City. Developed as part of StayWise's data science initiative, the project addresses the business challenge of helping hosts set competitive nightly rates based on property characteristics, location, and market data.

**Company**: StayWise - A global vacation rental platform
**Team**: Data Science Team
**Author**: Rajan
**Institution**: Lambton College

- - -

## Problem Statement

Listing prices on StayWise vary significantly, even among similar properties. The business team needs a machine learning model that predicts optimal nightly prices for new listings based on:

* **Location**: Neighbourhood, coordinates, geographic clusters
* **Property Characteristics**: Room type, amenities, size
* **Host Information**: Experience level, listing count, activity
* **Review Metrics**: Number of reviews, engagement scores, ratings

The dataset from AWS S3 contains noisy data with missing values, outliers, and categorical fields requiring extensive preprocessing. The solution must be reproducible, trackable, and production-ready.

- - -

## Objectives

1. **Data Pipeline**: Retrieve and clean noisy data from AWS S3
2. **Preprocessing**: Handle missing values, outliers, and categorical encoding
3. **Feature Engineering**: Create new features capturing pricing patterns
4. **Model Development**: Train and compare 8 different regression models
5. **Experiment Tracking**: Log all experiments using MLflow for reproducibility
6. **Model Selection**: Identify best-performing model for production deployment
7. **Model Registry**: Register top model in MLflow Model Registry

- - -

## Setup and Installation

### Prerequisites

* Python 3.12 or higher
* AWS Account with S3 bucket access
* MLflow (for experiment tracking)

### Installation Steps

1. **Clone the repository**:

``` bash
git clone <repository-url>
cd Predicting-Airbnb-Listing-Prices-with-MLflow-and-AWS-S3
```

2. **Install dependencies**:

``` bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. **Configure AWS S3**:

``` env
S3_BUCKET_NAME=your-bucket-name
```

* Create a `.env` file in the project root:

AWS\_REGION=us-east-1
AWS\_ACCESS\_KEY\_ID=your-access-key
AWS\_SECRET\_ACCESS\_KEY=your-secret-key
\`\`\`

\* Upload the dataset to S3: `s3://your-bucket-name/AB NYC 2019 Assessment 2.csv`
4. **Launch Jupyter Notebook**:

``` bash
jupyter notebook airbnb_complete_project.ipynb
```

- - -

## 📁 Repository Structure

```
│
├── airbnb_complete_project.ipynb    # Main notebook with complete pipeline
├── main.py                          # Python script version (optional)
├── README.md                        # This file
├── pyproject.toml                   # Project dependencies
├── uv.lock                          # Locked dependency versions
│
├── .env                             # Environment variables (not in git)
├── .gitignore                       # Git ignore rules
│
├── mlruns/                          # MLflow experiment tracking data
│   └── [experiment-runs]/
│
├── models/                          # Saved model files
│   └── best_model_xgboost.pkl
│
├── visualizations/                  # Generated plots and charts
│   ├── price_distribution.png
│   ├── categorical_analysis.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   └── residuals.png
│
├── model_comparison_results.csv     # Model performance comparison
│
└── AB NYC 2019 Assessment 2.csv     # Raw dataset (if stored locally)
```

- - -

## Workflow Description

### 1\. Data Loading

* **Source**: AWS S3 bucket (`s3://ml-ops-lambton-bucket/AB NYC 2019 Assessment 2.csv`)
* **Method**: Direct S3 access using `boto3` client
* **Dataset**: 48,895 listings with 16 features

### 2\. Exploratory Data Analysis \(EDA\)

* Dataset overview and structure analysis
* Missing values identification (\~20% missing review data)
* Target variable (price) distribution analysis
* Categorical variable distributions
* Statistical summaries and visualizations

### 3\. Data Preprocessing

* **Missing Values**:
    * Filled missing reviews with 0
    * Filled missing dates with 'Unknown'
* **Outlier Removal**:
    * Removed prices above 95th percentile ($355)
    * Removed minimum\_nights > 365 days
    * Final dataset: 46,430 listings (5% removed)

### 4\. Feature Engineering

Created 7 new features:

* **location\_cluster**: K-Means clustering of coordinates (10 clusters)
* **is\_luxury**: Binary indicator from listing name keywords
* **has\_reviews**: Binary indicator for listings with reviews
* **review\_score**: Engagement metric (reviews × reviews\_per\_month)
* **host\_activity**: Categorical (Single/Small/Medium/Large)
* **availability\_category**: Categorical (Low/Medium/High/Very High)
* **price\_category**: Categorical (Budget/Mid-range/Premium/Luxury)

### 5\. Feature Preparation

* **Numeric Features**: 10 features (latitude, longitude, reviews, etc.)
* **Categorical Features**: 6 features (neighbourhood, room\_type, etc.)
* **Preprocessing Pipeline**:
    * Numeric: Median imputation → Standard scaling
    * Categorical: Constant imputation → One-hot encoding
* **Train-Test Split**: 80/20 (37,144 training, 9,286 testing)

### 6\. Model Training

Trained 8 regression models with MLflow tracking:

1. **Linear Regression** \- Baseline linear model
2. **Ridge Regression** \- L2 regularization
3. **Lasso Regression** \- L1 regularization with feature selection
4. **ElasticNet** \- Combined L1/L2 regularization
5. **Decision Tree** \- Non\-linear single tree
6. **Random Forest** \- Ensemble of trees
7. **Gradient Boosting** \- Sequential tree boosting
8. **XGBoost** \- Optimized gradient boosting

**Hyperparameter Tuning**: GridSearchCV with 3-fold cross-validation

### 7\. Model Evaluation

* **Metrics Tracked**: RMSE, MAE, R² (both train and test)
* **Model Comparison**: Ranked by Test R² score
* **Best Model Selection**: XGBoost (R² = 0.5707, RMSE = $46.90)

### 8\. Model Analysis

* Feature importance analysis
* Prediction vs actual scatter plots
* Residual analysis
* Model performance visualizations

- - -

## MLflow Experiment Tracking

### Accessing MLflow UI

1. **Start MLflow UI**:

``` bash
mlflow ui
```

2. **Open in browser**:

```
http://localhost:5000
```

### What's Tracked in MLflow

* **Parameters**: Model type, hyperparameters, random state
* **Metrics**: Train/test RMSE, MAE, R² scores
* **Artifacts**: Trained models, preprocessing pipelines
* **Metadata**: Training samples, test samples, experiment name

### Experiment Details

* **Experiment Name**: `airbnb_price_prediction`
* **Total Runs**: 8 (one per model)
* **Tracking URI**: Local file system (`mlruns/` directory)

### Viewing Results

In MLflow UI, you can:

* Compare all 8 models side-by-side
* Filter and sort by metrics
* View detailed parameters for each run
* Download models and artifacts
* Register best model to Model Registry

**Note**: Screenshots of MLflow UI showing experiment runs, metrics comparison, and model registry should be added to the `visualizations/` directory or included in project documentation.

- - -

## Model Performance Results

### Model Comparison Summary

| Rank | Model | Test RMSE | Test R² | Performance Tier |
| ---- | ----- | --------- | ------- | ---------------- |
| 🥇 1 | **XGBoost** | $46.90 | **0.5707** | Best Overall |
| 🥈 2 | Random Forest | $47.11 | 0.5668 | Excellent |
| 🥉 3 | Gradient Boosting | $47.36 | 0.5622 | Excellent |
| 4 | Decision Tree | $49.50 | 0.5218 | Good |
| 5 | Ridge Regression | $49.83 | 0.5152 | Baseline |
| 6 | Linear Regression | $49.84 | 0.5151 | Baseline |
| 7 | ElasticNet | $55.69 | 0.3947 | Moderate |
| 8 | Lasso Regression | $62.76 | 0.2311 | Poor |

### Best Model: XGBoost

**Performance Metrics**:

* **Test R²**: 0.5707 (explains 57% of price variation)
* **Test RMSE**: $46.90 (average prediction error)
* **Train R²**: 0.6412
* **Generalization Gap**: 7% (acceptable overfitting)

**Hyperparameters**:

* `n_estimators`: 200
* `learning_rate`: 0.1
* `max_depth`: 7

**Business Interpretation**:

* For a typical $150 listing, predictions are within \~$47 (31% error)
* Model captures most predictable price factors
* Remaining 43% variation due to factors not in dataset (amenities, photos, seasonal trends)

- - -

## Key Insights and Observations

### 1. **Tree-Based Models Dominate**

* All top 3 models are tree-based (XGBoost, Random Forest, Gradient Boosting)
* Non-linear models capture complex feature interactions better than linear models
* Ensemble methods outperform single models

### 2. **Feature Engineering Impact**

* Created features (location\_cluster, is\_luxury, review\_score) contribute significantly
* Location and room type are strongest price predictors
* Review metrics provide valuable pricing signals

### 3. **Model Performance Patterns**

* **Linear Models** (Linear, Ridge): \~51% R² - Baseline performance
* **Regularized Models** (Lasso, ElasticNet): Underperformed due to over-regularization
* **Tree Models**: 52-57% R² - Significant improvement over linear
* **Ensemble Models**: Best performance through model combination

### 4. **Data Quality Insights**

* 20% of listings missing review data (new listings)
* Price distribution is right-skewed (mean > median)
* Manhattan and Brooklyn dominate (85% of listings)
* Entire homes/apts and private rooms are most common

### 5. **Overfitting Analysis**

* Random Forest shows largest train/test gap (84% vs 57%)
* XGBoost has best balance (64% vs 57%)
* All models show acceptable generalization

### 6. **Business Value**

* Model can help hosts set competitive initial prices
* 57% explanatory power is strong for real estate pricing
* Average $47 error is acceptable for $100-300 price range
* Model serves as starting point, not absolute truth

- - -

## Next Steps and Recommendations

### Immediate Actions

1. **Register Best Model**: Add XGBoost to MLflow Model Registry
2. **Deploy to Production**: Create API endpoint for real-time predictions
3. **Monitor Performance**: Track model accuracy over time

### Model Improvements

1. **Additional Features**:
    * Property amenities (pool, parking, WiFi)
    * Listing photos quality score
    * Host response time and superhost status
    * Seasonal demand patterns
    * Competitor pricing data
2. **Advanced Techniques**:
    * Deep learning models (neural networks)
    * Time series analysis for seasonal trends
    * Ensemble of best models (stacking)
    * Hyperparameter optimization (Optuna, Hyperopt)
3. **Data Collection**:
    * Collect more detailed property features
    * Track booking rates vs. prices
    * Monitor competitor pricing
    * Gather host feedback on predictions

### Production Deployment

1. **Model Serving**: Deploy via MLflow Model Serving or containerized API
2. **A/B Testing**: Compare model predictions vs. current pricing
3. **Feedback Loop**: Collect host adjustments to improve model
4. **Retraining Schedule**: Monthly retraining with fresh data

- - -

## Dependencies

Key Python packages used:

* `pandas` (2.3.3+) - Data manipulation
* `numpy` (2.3.5+) - Numerical computing
* `scikit-learn` (1.7.2+) - Machine learning algorithms
* `xgboost` (3.1.2+) - Gradient boosting
* `mlflow` (3.6.0+) - Experiment tracking
* `boto3` (1.41.1+) - AWS S3 integration
* `matplotlib` (3.10.7+) - Visualization
* `seaborn` (0.13.2+) - Statistical visualization
* `python-dotenv` (1.2.1+) - Environment variables

See `pyproject.toml` for complete dependency list.

- - -

## Visualizations

The project generates several visualizations saved in the `visualizations/` directory:

1. **price\_distribution.png** \- Price distribution analysis
2. **categorical\_analysis.png** \- Neighbourhood and room type distributions
3. **model\_comparison.png** \- Side\-by\-side model performance comparison
4. **feature\_importance.png** \- Top 20 most important features \(if available\)
5. **actual\_vs\_predicted.png** \- Scatter plot of predictions vs\. actual prices
6. **residuals.png** \- Residual analysis for model diagnostics

- - -

##  Learning Outcomes

This project demonstrates:

* End-to-end ML pipeline from data loading to model deployment
* AWS S3 integration for production data sources
* MLflow for experiment tracking and model management
* Feature engineering and preprocessing best practices
* Model comparison and selection methodology
* Business-focused model interpretation
* Production-ready code structure

- - -

<br>
<br>
