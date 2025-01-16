# Bank Transaction Fraud Detection

## Project Overview
This project demonstrates a clustering-based approach to detect fraudulent bank transactions using the DBSCAN algorithm.

## Dataset Description
The dataset contains simulated bank transactions with attributes such as:
- `TransactionAmount`: The amount of money involved in the transaction.
- `CustomerAge`: The age of the customer making the transaction.
- `AccountBalance`: The balance in the customer's account.

## Preprocessing Steps
1. **Feature Selection**: Key features (`TransactionAmount`, `CustomerAge`, `AccountBalance`) were selected for clustering.
2. **Feature Scaling**: Data was scaled using `RobustScaler` to handle outliers effectively.
3. **Dimensionality Reduction**: PCA reduced the dataset to 2 dimensions for clustering and visualization.

## Model
The DBSCAN algorithm was used with the following parameters:
- `eps = 0.5`
- `min_samples = 5`

## Results
- **Silhouette Score**: `0.6669`
- **Number of Potential Frauds Detected**: `5`

## Visualization
1. **DBSCAN Clustering Results**: Clusters and potential frauds are highlighted in the PCA-reduced feature space.
2. **Fraud Details**: Potential frauds are visualized based on `TransactionAmount` and `CustomerAge`.

## Files
1. `fraud_detection_dbscan.py`: Python script implementing the fraud detection pipeline.
