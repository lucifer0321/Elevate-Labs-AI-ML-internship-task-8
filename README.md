# Elevate-Labs-AI-ML-internship-task-8
Customer Segmentation using K-Means Clustering Implemented K-Means clustering on the Mall Customer dataset to segment customers based on annual income and spending score. Applied the Elbow Method to find the optimal number of clusters, visualized customer segments using scatter plots, and evaluated clustering using the Silhouette Score.


#description
We used the Mall Customer dataset to perform K-Means clustering for customer segmentation based on Annual Income and Spending Score. The data was scaled using StandardScaler to ensure fair distance calculations. We applied the Elbow Method to determine the optimal number of clusters and used KMeans from Scikit-learn to group customers. Finally, we visualized the clusters using a scatter plot and evaluated the model with the Silhouette Score to measure clustering quality.

#code
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Select numerical features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal K
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.grid(True)
plt.show()

# Fit final model (use K=5 as a common choice)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
            c=df['Cluster'], cmap='viridis', s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments - KMeans Clustering')
plt.grid(True)
plt.show()

# Evaluate Clustering
score = silhouette_score(X_scaled, clusters)
print(f'Silhouette Score: {score:.2f}')
