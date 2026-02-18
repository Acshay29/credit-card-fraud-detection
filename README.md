# Real-Time Anomaly Detection in Financial Transactions

## One-Line Summary
An unsupervised machine learning project focused on detecting fraudulent credit card transactions under extreme class imbalance using Isolation Forest and LOF.

## Project Overview
Credit card fraud is rare — but when it happens, it’s expensive.
In real-world transaction data, fraudulent activity makes up less than 0.2% of total transactions. That makes fraud detection a highly imbalanced and challenging problem.
In this project, I built an unsupervised anomaly detection system to identify potentially fraudulent transactions without relying on labeled training data.
Instead of focusing on accuracy (which is misleading in imbalanced datasets), I focused on what actually matters in fraud systems:
- Precision
- Recall
- Business tradeoffs

## Dataset Overview
Total transactions: 284,807
Fraudulent transactions: 492
Fraud rate: 0.17%
The dataset contains:
28 anonymized PCA features (V1–V28)
Transaction Time
Transaction Amount
Target class (0 = Normal, 1 = Fraud)
The extreme imbalance makes this a perfect candidate for anomaly detection methods.

## Data Preparation
Before modeling, I:
- Explored class imbalance
- Visualized transaction distribution
- Scaled Amount and Time using RobustScaler
I chose RobustScaler because fraud detection datasets often contain extreme values, and RobustScaler handles outliers better than StandardScaler.

## Models Implemented
I tested two unsupervised anomaly detection algorithms:
1. Isolation Forest
A tree-based model that isolates anomalies by randomly partitioning features.
2. Local Outlier Factor (LOF)
A density-based method that detects anomalies based on local data density.

## Data Visualization

### Class Distribution

![Class Distribution](images/class_distribution.png)

The dataset shows extreme class imbalance, with fraudulent transactions representing less than 0.2% of total transactions.

### Confusion Matrix (Isolation Forest - 0.002)

![Confusion Matrix](images/confusion_matrix.png)

The confusion matrix highlights the tradeoff between recall and precision after tuning the contamination parameter.

## Isolation Forest Results
Version 1 (contamination = 0.001)
- Precision: 0.36
- Recall: 0.21
- F1-score: 0.27
This configuration was strict — it produced fewer false alarms but missed many fraud cases.
Tuned Version (contamination = 0.002)
- Precision: 0.26
- Recall: 0.30
- F1-score: 0.28
By slightly increasing contamination, recall improved while maintaining acceptable precision.
This configuration achieved a better precision–recall balance, improving fraud detection while controlling false positives.
| Model | Precision | Recall | F1-score |
|-------|----------|--------|----------|
| Isolation Forest (0.001) | 0.36 | 0.21 | 0.27 |
| Isolation Forest (0.002) | 0.26 | 0.30 | 0.28 |
| LOF | 0.00 | 0.00 | 0.00 |


## Local Outlier Factor (LOF) Results
- Precision: 0.00
- Recall: 0.00
LOF failed to detect fraudulent transactions in this high-dimensional and extremely imbalanced dataset.
This highlighted an important practical lesson:
"Not all anomaly detection algorithms scale well for large, high-dimensional financial datasets."
Isolation Forest proved more robust and scalable.

## Threshold Tuning (Advanced Step)
Instead of relying only on contamination, I manually analyzed anomaly scores using:
- decision_function()
By selecting the lowest 1% anomaly scores as fraud:
- Recall increased to 57%
- Precision dropped to 10%
This experiment clearly demonstrated the precision–recall tradeoff.
In real financial systems:
- Missing fraud (false negatives) is costly
- False alarms (false positives) are inconvenient but manageable
Threshold selection depends on business risk tolerance.

## Final Recommendation
For a balanced and scalable fraud detection system:
Isolation Forest with contamination = 0.002
It provides:
- Reasonable recall
- Acceptable precision
- Better scalability than LOF
- Stable performance on large datasets
Threshold tuning can further increase recall depending on business requirements.

## Key Takeaways
- Accuracy is misleading in highly imbalanced datasets.
- Precision–Recall tradeoff is central to fraud detection.
- Isolation Forest scales better than LOF for large financial data.
- Model performance must be evaluated from a business perspective — not just metrics.

## Tech Stack
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook

## Project Structure
Fraud_Project/
│
├── fraud_detection.ipynb
├── README.md
├── requirements.txt

## Next Steps
If extended further, this project could include:
- Precision–Recall curve optimization
- Ensemble anomaly detection
- Real-time API deployment
- Integration into a streaming fraud monitoring system

## Final Thoughts
This project helped me understand how anomaly detection works in real-world financial systems, especially under extreme class imbalance.
It also reinforced that machine learning is not just about building models — it’s about making informed tradeoffs based on business needs.
