from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# creating artificial 2D data
X = np.random.randn(300, 2) * 0.8

model = KMeans(n_clusters=3)
model.fit(X)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1) # 1 row, 2 col, first portion
plt.scatter(X[:, 0], X[:, 1]) # plotting: 1st feature as X and 2nd feature as Y
plt.title('Before Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)

plt.subplot(1, 2, 2) # 1 row, 2 col, second portion
plt.scatter(X[:, 0], X[:, 1], c=model.labels_) # color by clusters
plt.title('After Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)

plt.show()

