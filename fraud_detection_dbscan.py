# Code begins here
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import kagglehub

### Part 1: Download the Dataset
# Download latest version of the dataset
path = kagglehub.dataset_download("valakhorasani/bank-transaction-dataset-for-fraud-detection")
print("Path to dataset files:", path)

# Load the dataset into a DataFrame
data_file = f"{path}/bank_transactions_data_2.csv"  # Replace with the correct file name if different
df = pd.read_csv(data_file)

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

### Part 2: Preprocessing
# Step 1: Feature Selection
features = ['TransactionAmount', 'CustomerAge', 'AccountBalance']
X = df[features]

# Step 2: Scaling Features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Dimensionality Reduction with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

### Part 3: Clustering with DBSCAN
# Initialize DBSCAN with best parameters
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_pca)

# Step 4: Add Clustering Results to Dataset
df['DBSCAN_Cluster'] = labels
df['Potential_Fraud'] = labels == -1
num_frauds = df['Potential_Fraud'].sum()

### Part 4: Evaluate Clustering
unique_labels = np.unique(labels)
if len(unique_labels) > 1:
    silhouette = silhouette_score(X_pca, labels)
else:
    silhouette = None

# Print Results
print("\nClustering Results:")
print("Best DBSCAN Parameters:")
print("eps: 0.5, min_samples: 5")
print(f"Silhouette Score: {silhouette}")
print(f"Number of Potential Frauds Detected: {num_frauds}")

### Part 5: Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10', s=60, alpha=0.7)
plt.title('DBSCAN Clustering with Potential Frauds Highlighted')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title="Cluster", loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()

fraud_data = df[df['Potential_Fraud']]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=fraud_data, x='TransactionAmount', y='CustomerAge', color='red', label='Potential Frauds', s=100, marker='X')
plt.title('Potential Frauds Detected by DBSCAN')
plt.xlabel('Transaction Amount')
plt.ylabel('Customer Age')
plt.legend()
plt.show()
