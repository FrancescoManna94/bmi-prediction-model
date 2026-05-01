# Body Fat Percentage Prediction using Regularized Regression

## Overview
This repository hosts a data science project focused on developing a predictive model to estimate **Body Fat Percentage (BFP)**. While BFP is a vital health metric, its gold-standard measurement via underwater weighing (Densitometry) is often expensive and inaccessible. 

The goal of this study is to demonstrate how **Linear Regression**, optimized through **Lasso** and **Ridge** regularization, can provide an accurate and practical estimation tool using only non-invasive anthropometric measurements.

## Project Objectives
*   **Mitigate Multicollinearity:** Use regularization techniques to handle highly correlated anatomical data (e.g., body circumferences).
*   **Model Optimization:** Compare **Lasso ($L_1$)** and **Ridge ($L_2$)** regression to find the best balance between complexity and predictive power.
*   **Practical Application:** Create a model based on easily acquirable metrics like age, weight, and body measurements.

## Dataset Description
The model utilizes a dataset consisting of **15 variables**. The target variable is the Body Fat percentage determined by **Siri's Equation**.

### Features:
*   **Demographics:** Age, Weight, Height.
*   **Physical Metrics:** Body Density (via densitometry).
*   **Anatomical Circumferences:** Abdomen, Chest, Thigh, Neck, Hips, Knee, Ankle, Bicep, Wrist, and Forearm.

## Methodology
The workflow follows standard Data Science practices:
1.  **Exploratory Data Analysis (EDA):** Visualizing correlations between body parts and fat percentage.
2.  **Preprocessing:** Feature scaling and handling of potential outliers.
3.  **Model Training:** Implementation of Linear, Ridge, and Lasso Regression.
4.  **Evaluation:** Performance assessment using **$R^2$**, **MAE**, and **RMSE** metrics.

## Key Results
The study highlights how **Lasso Regression** effectively performs feature selection, identifying the most significant body circumferences for prediction, while **Ridge Regression** ensures stability by penalizing extreme coefficient values. This approach provides a reliable tool for health professionals to estimate body composition with high precision.

## Technologies Used
*   **Python** (Pandas, NumPy)
*   **Scikit-Learn** (Linear Models, Preprocessing)
*   **Matplotlib / Seaborn** (Data Visualization)
