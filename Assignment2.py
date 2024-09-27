from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# 1. Retrieve and load the Olivetti faces dataset
olivetti_faces = fetch_olivetti_faces()

X = olivetti_faces.data  #  flattened 1D array format
y = olivetti_faces.target  # The target labels (person identifier)


# 2. Split the training set, a validation set, and a test set using stratified sampling to ensure that there are the same number of images per person in each set
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=51)
for train_idx, temp_idx in sss.split(X, y):
    X_train, X_temp = X[train_idx], X[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]


sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=51)
for val_idx, test_idx in sss_val_test.split(X_temp, y_temp):
    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]

# You now have stratified training, validation, and test sets
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")


# 3. Using k-fold cross validation, train a classifier to predict which person is represented in each picture, and evaluate it on the validation set
# Initialize the classifier
clf = SVC(kernel='linear', random_state=51)

# Perform 5-fold cross-validation
scores = cross_val_score(clf, X_train, y_train, cv=5)

# Fit on full training set and evaluate on validation set
clf.fit(X_train, y_train)
val_accuracy = clf.score(X_val, y_val)

print(f'Cross-validation accuracy scores: {scores}')
print(f'Average cross-validation accuracy: {scores.mean()}')
print(f'Validation accuracy: {val_accuracy}')


# 4. Use K-Means to reduce the dimensionality of the set
# Use the silhouette score approach to choose the number of clusters.
silhouette_scores = []
k_values = range(2, 150)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=51)
    kmeans.fit(X_train)
    
    # Compute silhouette score
    score = silhouette_score(X_train, kmeans.labels_)
    silhouette_scores.append(score)

best_k = k_values[np.argmax(silhouette_scores)]
print("Best number of clusters based on silhouette score:", best_k)


# 5. Use the set from step (4) to train a classifier as in step (3)
kmeans = KMeans(n_clusters=best_k, random_state=51)
X_train_reduced = kmeans.fit_transform(X_train)

# Train the classifier on reduced features
clf.fit(X_train_reduced, y_train)

# Transform validation set and evaluate
X_val_reduced = kmeans.transform(X_val)
val_score_reduced = clf.score(X_val_reduced, y_val)
print("Validation set score after K-Means reduction:", val_score_reduced)


# 5. Apply DBSCAN algorithm to the Olivetti Faces dataset for clustering.
# Preprocess the images and convert them into feature vectors, then use DBSCAN to group similar images together based on their density
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.4, min_samples=5, metric='cosine')
dbscan_labels = dbscan.fit_predict(X_scaled)

# Evaluate clustering result
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print("Estimated number of clusters (DBSCAN):", n_clusters)

# Check how many images belong to each cluster
unique, counts = np.unique(dbscan_labels, return_counts=True)
cluster_summary = {int(label): int(count) for label, count in zip(unique, counts)}
print("Cluster distribution:", cluster_summary)

# Plot some sample images from a cluster
def plot_images(cluster_label, dbscan_labels, images, n_images=10):
    idxs = np.where(dbscan_labels == cluster_label)[0]
    plt.figure(figsize=(10, 5))
    
    for i in range(min(n_images, len(idxs))):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(images[idxs[i]].reshape(64, 64), cmap='gray')
        plt.axis('off')
    
    plt.show()

# Plot images from one of the clusters
plot_images(1, dbscan_labels, olivetti_faces.images)