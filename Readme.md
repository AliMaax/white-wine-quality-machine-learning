EE 439 Machine Learning: Phase 2 Project
Topic: UCI White Wine Quality Prediction

## Team Breakdown & Contributions
- Muhammad Hashim Butt (2023-EE-27): Handled the dataset partitioning and the Stratified 5-Fold cross-validation loop. Also wrote the Main Model (Decision Tree and Random Forest) completely from scratch using numpy, implementing bootstrap sampling and Gini impurity calculations.
- Ali Maaz (2023-EE-21): Managed the preprocessing pipeline (logarithmic transformations and scaling). Coded the Baseline Model (Linear Regression) from scratch using the Moore-Penrose pseudo-inverse, and built the Flexible Comparison Model using scikit-learn's SVM.

## Project Overview
This folder contains our complete modeling pipeline for the White Wine Quality dataset. We built three distinct models to handle the class imbalances and feature overlaps we found during Phase 0:
1. Baseline: Linear Regression (Built from scratch)
2. Main: Random Forest Ensemble (Built from scratch)
3. Flexible: SVM with an RBF Kernel (Using sklearn)

## Tech Stack & Requirements
We wrote and tested this in VS Code on Macos Tahoe 26.3.1 (Muhammad Hashim Butt ,2023 -EE-27) & Windows 11 (Ali Maaz ,2023-EE-21) using Python 3.14 (any version 3.10+ will work). 

You only need three standard libraries to run this:
- numpy
- pandas
- scikit-learn 
*(Note for grading: As per the rules, we only used scikit-learn for data splitting, scaling, metrics, and the SVM. The Linear Regression and Random Forest algorithms are purely custom numpy math).*

## How to Run the Code
1. Extract the Project_code.zip file into a single folder.
2. Make sure the dataset (`winequality-white.csv`) is sitting in the exact same folder as the `phase2.py` script.
3. Open your terminal or command prompt in that folder.
4. Run the script by typing: `python phase2.py`
5. The script will automatically load the data, run the cross-validation for all three models, and print out the final RMSE, Accuracy, and Macro F1-Scores.