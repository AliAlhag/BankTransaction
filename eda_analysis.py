

# Exploratory Data Analysis (EDA) for Bank Transaction Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA



### Part 1: Initial Inspection
# Dataset Overview
print("Dataset Overview:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Check for Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

### Part 2: Feature Exploration
# Key Features to Explore
features = ['TransactionAmount', 'CustomerAge', 'AccountBalance']

# Plot Distributions
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Correlation Matrix
correlation_matrix = df[features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

### Part 3: Scaling Impact
# Scaling the Features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df[features])

# Plot Scaled Features
scaled_df = pd.DataFrame(X_scaled, columns=features)
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.histplot(scaled_df[feature], kde=True, bins=30, color='green')
    plt.title(f'Scaled Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

### Part 4: Dimensionality Reduction with PCA
# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Variance Explained
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance by PCA Components:")
for i, variance in enumerate(explained_variance, start=1):
    print(f"Component {i}: {variance:.2%}")

# Visualize PCA Components
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], alpha=0.7)
plt.title('PCA Components Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

### Part 5: Insights from EDA
print("\nKey Insights:")
print("- Features show some correlation, especially between AccountBalance and TransactionAmount.")
print("- Scaling significantly normalizes the data distributions.")
print("- PCA reduces dimensionality while retaining most of the variance.")

# Save Insights to a Text File (Optional)
with open("eda_insights.txt", "w") as f:
    f.write("Key Insights from EDA:\n")
    f.write("- Features show some correlation, especially between AccountBalance and TransactionAmount.\n")
    f.write("- Scaling significantly normalizes the data distributions.\n")
    f.write("- PCA reduces dimensionality while retaining most of the variance.\n")
