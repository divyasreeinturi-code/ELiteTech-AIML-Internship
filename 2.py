
# AIML INTERNSHIP - TASK 2
# Customer Segmentation Using K-Means Clustering

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 2: Load dataset
df = pd.read_csv(
    r"C:\Users\inturi divyasree\Downloads\archive (2)\Mall_Customers.csv"
)

print("Dataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())

# Step 3: Check dataset information
print("\nDataset Information:")
print(df.info())

# Step 4: Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 5: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Find optimal number of clusters using Elbow Method
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Step 7: Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCustomer Clustering Results:")
print(df.head(10))

# Step 8: Evaluate clustering
score = silhouette_score(X_scaled, df['Cluster'])

print("\nSilhouette Score:", round(score, 3))

# Step 9: Display cluster information
print("\nNumber of Customers in Each Cluster:")
print(df['Cluster'].value_counts().sort_index())

# Step 10: Visualize customer clusters
plt.figure(figsize=(10, 6))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=df['Cluster'],
    cmap='viridis',
    s=50
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.colorbar(label="Customer Cluster")
plt.grid(True)
plt.show()

# Step 11: Save results
df.to_csv("Customer_Segments.csv", index=False)

print("\nCustomer Segmentation Completed Successfully!")
print("Results saved in Customer_Segments.csv")
