# Customer Churn Predictor

## Overview
This project implements a machine learning model to predict customer churn in a banking context. The model analyzes various customer characteristics and transaction patterns to identify customers who are likely to leave the bank's services.

## Dataset
The project uses the Telco Customer Churn dataset from Kaggle, which includes:
- Customer demographics (age, gender, location)
- Account information (balance, type of account, tenure)
- Transaction history
- Product usage
- Credit status
- Salary information

## Features
- Data preprocessing and feature engineering
- Transaction history analysis
- Customer behavior pattern recognition
- Machine learning model for churn prediction
- Performance metrics and evaluation

## Project Structure
```
4. Customer Churn Predictor/
├── datas/                  # Dataset directory
├── params/                 # Model parameters
├── building model.ipynb    # Main notebook with model implementation
├── pyproject.toml         # Project dependencies
└── poetry.lock           # Locked dependencies
```

## Setup
1. Install Poetry (dependency management tool)
2. Install dependencies:
```bash
poetry install
```

## Usage
1. Open `building model.ipynb` in your preferred Jupyter environment
2. Run the cells sequentially to:
   - Load and preprocess the data
   - Engineer features
   - Train the model
   - Evaluate predictions

## Dependencies
- pandas
- numpy
- scikit-learn
- Jupyter

## License
This project is open source and available under the MIT License.
