import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from skfuzzy import cmeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the country happiness data from CSV file
df = pd.read_csv('C:\\Users\\USER\\Downloads\\2017.csv')

# Drop any rows with missing values
df.dropna(inplace=True)

# Select features for clustering
# X = df[['Score', 'GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
X = df[['Happiness.Score', 'Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.', 'Freedom', 'Generosity', 'Trust..Government.Corruption.']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# def fuzzy_kmeans(X, max_clusters, m, max_iter=100):
#     errors = []
#     for n_clusters in range(2, max_clusters + 1):
#         cntr, u, _, _, _, _, _ = cmeans(X.T, n_clusters, m, error=0.005, maxiter=max_iter)
#         labels = np.argmax(u, axis=0)
#         error = np.sum((X - cntr[labels]) ** 2)  # Mean squared error
#         errors.append(error)
#     return errors

# # Choose the maximum number of clusters to consider
# max_clusters = 10
# errors = fuzzy_kmeans(X_scaled, max_clusters, m=2)

# # Visualize Elbow Method
# plt.plot(range(2, max_clusters + 1), errors)
# plt.title('Elbow Method for Optimal Number of Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Mean Squared Error')
# plt.show()

def fuzzy_kmeans(X, n_clusters, m, max_iter=100):
    # Fuzzy K-means clustering
    cntr, u, _, _, _, _, _ = cmeans(X.T, n_clusters, m, error=0.005, maxiter=max_iter)
    
    # Predict cluster membership
    labels = np.argmax(u, axis=0)
    
    return cntr, labels

def triangle_membership(data, centers, width):
    membership = np.zeros((len(data), len(centers)))
    for i in range(len(data)):
        for j in range(len(centers)):
            if abs(data[i] - centers[j]) < width:
                membership[i, j] = 1 - abs(data[i] - centers[j]) / width
    return membership

def visualize_clusters(X, centers, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 2], X_scaled[:, 3], c=labels, cmap='viridis', s=50, alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
    plt.title(title)
    plt.xlabel('Family')
    plt.ylabel('Health Life Expectancy')
    plt.colorbar(label='Cluster')
    plt.show()

# Apply fuzzy K-means clustering
n_clusters = 4
m = 2
centers, labels = fuzzy_kmeans(X_scaled, n_clusters, m)
# Visualize the clusters
visualize_clusters(X_scaled, centers, labels, 'Fuzzy K-means Clustering')

# Define membership functions
membership_functions = triangle_membership(X_scaled[:, 2], centers[:, 2], width=1)

# Plot membership functions
plt.figure(figsize=(8, 6))
for j in range(len(centers)):
    plt.plot(X_scaled[:, 0], membership_functions[:, j], label=f'Cluster {j}')
plt.title('Membership Functions for Feature 1')
plt.xlabel('Feature 1')
plt.ylabel('Membership')
plt.legend()
plt.show()
