💳 Credit Card Fraud Detection using Support Vector Classifier (SVC)
This project focuses on building a machine learning model to detect fraudulent credit card transactions. Due to the highly imbalanced nature of the dataset (many more non-fraudulent transactions than fraudulent ones), evaluating the model performance is crucial.

The model is developed using the Support Vector Classifier (SVC) from the scikit-learn library.

💻 Setup and Dependencies
This project requires Python and the following libraries:

pandas

numpy

scikit-learn

seaborn

You can install them using pip:
pip install pandas numpy scikit-learn seaborn
Dataset
The notebook uses the creditcard.csv dataset, which contains transactions made by credit cards over two days in September 2013 by European cardholders.

Key Data Characteristics:
Total Transactions: 284,807

Features: Time, Amount, and 28 principal components (V1 through V28) obtained via PCA.

Target Variable (Class):

0: Non-fraudulent transaction (284,315 instances)

1: Fraudulent transaction (492 instances)

As shown by the value_counts(), the data is severely imbalanced, which is a typical challenge in fraud detection.

Methodology
1. Data Pre-processing
Feature Scaling: The features (including Time and Amount) were scaled using StandardScaler to normalize the data, which is essential for distance-based algorithms like SVC.

Splitting Data: The dataset was split into training and testing sets with a test_size=0.2 (20% for testing).

2. Modeling
Model: A Support Vector Classifier (SVC) was initialized and trained on the scaled training data (X_train, y_train).

Training Score: 0.999675

Test Score (Accuracy): 0.999385

3. Evaluation
While the accuracy is very high (close to 1.00), this is often misleading for imbalanced datasets. The key is to look at metrics that measure performance on the minority class (fraudulent transactions).

Confusion Matrix
The confusion matrix on the test set (y_test) is as follows:

,Predicted Fraud (1),Predicted Normal (0)
Actual Fraud (1),55 (True Positives),32 (False Negatives)
Actual Normal (0),3 (False Positives),56872 (True Negatives)
Classification Report

precision    recall  f1-score   support

           0       1.00      1.00      1.00     56875
           1       0.95      0.63      0.76        87

    accuracy                           1.00     56962
   macro avg       0.97      0.82      0.88     56962
weighted avg       1.00      1.00      1.00     56962

Results Interpretation
High Accuracy (1.00): This is primarily driven by the large number of correctly identified Normal transactions (True Negatives: 56,872).

Precision for Fraud (0.95): When the model predicts a transaction is fraudulent, it's correct 95% of the time. This means few legitimate transactions are incorrectly flagged (low False Positives: 3).

Recall for Fraud (0.63): The model only captured 63% (55 out of 87) of the actual fraudulent transactions. This results in 32 fraudulent transactions being missed (False Negatives), which is a major concern in a fraud detection system.

Conclusion and Next Steps
The SVC model performs exceptionally well in avoiding false alarms (high precision), but its Recall for the fraud class is moderate (0.63). For a real-world fraud detection system, Recall is often the most critical metric, as minimizing missed fraud is paramount.

Potential next steps for improvement:

Imbalance Handling: Implement techniques like SMOTE (Synthetic Minority Over-sampling Technique) or use class weights to address the data imbalance.

Hyperparameter Tuning: Use Grid Search or Randomized Search to optimize the parameters of the SVC model (e.g., C, gamma, kernel).

Alternative Models: Compare performance with other models better suited for imbalanced data, such as Random Forest or LightGBM/XGBoost.
