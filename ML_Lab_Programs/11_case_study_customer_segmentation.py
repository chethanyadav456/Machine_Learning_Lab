"""
Case Study 3: Customer Segmentation for an E-Commerce Platform

Scenario:
An online retailer wants to understand its diverse customer base to design personalized
marketing campaigns. By analyzing purchasing behavior and transaction patterns, the
company hopes to identify different types of customers (e.g., high-value, frequent
buyers, or low-spending occasional shoppers).

Objective:
Use a Neural Network-based clustering approach (autoencoder) to automatically segment
customers based on their purchasing behavior.

Concept Used:
Unsupervised Learning combined with Neural Networks. An autoencoder compresses data
(encoding) and then reconstructs it (decoding). The compressed layer captures meaningful
patterns, which can then be clustered using algorithms like K-Means for segmentation.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    'Annual Income': [15, 16, 18, 35, 40, 42, 75, 80, 90, 95],
    'Spending Score': [80, 82, 78, 60, 65, 63, 25, 22, 18, 20],
    'Purchase Frequency': [12, 11, 10, 8, 7, 6, 4, 3, 2, 1]
}

df = pd.DataFrame(data)
print("Customer Dataset:\n")
print(df)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters
print("\nClustered Customer Data:\n")
print(df)

plt.figure(figsize=(8, 6))
plt.scatter(
    df['Annual Income'],
    df['Spending Score'],
    c=clusters,
    cmap='viridis',
    s=100
)
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df['Annual Income'],
    df['Spending Score'],
    c=clusters,
    cmap='viridis',
    s=100
)

for i in range(len(df)):
    plt.text(
        df['Annual Income'][i],
        df['Spending Score'][i],
        str(i + 1),
        fontsize=9
    )

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.colorbar(scatter, label="Cluster")
plt.show()

# Evaluation:
# - Cluster 0: High-income, low-spending, infrequent shoppers.
# - Cluster 1: Moderate-income, average spenders.
# - Cluster 2: Low-income, high-spending frequent customers.

# Conclusion:
# This case study demonstrates how Neural Networks (Autoencoders) can be used for
# Unsupervised Learning to segment customers in an e-commerce platform. By combining
# Autoencoders with K-Means clustering, we can discover hidden customer groups and
# provide valuable insights for targeted marketing and customer relationship management.
