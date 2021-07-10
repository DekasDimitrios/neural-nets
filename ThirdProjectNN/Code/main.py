import struct as st
import time
import numpy as np
from scipy.spatial.distance import minkowski, cityblock, chebyshev
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as numpy

NUM_OF_CLASSES = 4


# Unpacks the binary files and sets up the arrays based on
# file format that can be found here: http://yann.lecun.com/exdb/mnist/ .
def Load_MNIST():
    # Simple linear struct unpack to get the bytes of the two files into NumPy arrays.

    training_images_file = open('../Resources/train-images-idx3-ubyte', 'rb')
    training_images_file.seek(0)
    magic_num = st.unpack('>4B', training_images_file.read(4))
    num_of_training_images = st.unpack('>I', training_images_file.read(4))[0]
    num_of_training_rows = st.unpack('>I', training_images_file.read(4))[0]
    num_of_training_cols = st.unpack('>I', training_images_file.read(4))[0]
    training_images_array = []

    training_labels_file = open('../Resources/train-labels-idx1-ubyte', 'rb')
    training_labels_file.seek(0)
    training_labels_magic_num = st.unpack('>4B', training_labels_file.read(4))
    num_of_labels = st.unpack('>I', training_labels_file.read(4))[0]
    training_labels_array = []

    for i in range(0, 60000):
        label = st.unpack('>' + 'B', training_labels_file.read(1))[0]
        image = []
        bits = num_of_training_rows * num_of_training_cols
        for j in range(0, bits):
            pixel = st.unpack('>' + 'B', training_images_file.read(1))[0]
            image.append(pixel)
        if label < 4:
            training_labels_array.append(label)
            training_images_array.append(image)

    testing_images_file = open('../Resources/t10k-images-idx3-ubyte', 'rb')
    testing_images_file.seek(0)
    testing_images_magic_num = st.unpack('>4B', testing_images_file.read(4))
    num_of_testing_images = st.unpack('>I', testing_images_file.read(4))[0]
    num_of_testing_rows = st.unpack('>I', testing_images_file.read(4))[0]
    num_of_testing_cols = st.unpack('>I', testing_images_file.read(4))[0]
    testing_images_array = []

    testing_labels_file = open('../Resources/t10k-labels-idx1-ubyte', 'rb')
    testing_labels_file.seek(0)
    testing_labels_magic_num = st.unpack('>4B', testing_labels_file.read(4))
    num_of_testing_labels = st.unpack('>I', testing_labels_file.read(4))[0]
    testing_labels_array = []

    for i in range(0, 10000):
        label = st.unpack('>' + 'B', testing_labels_file.read(1))[0]
        image = []
        bits = num_of_training_rows * num_of_training_cols
        for j in range(0, bits):
            pixel = st.unpack('>' + 'B', testing_images_file.read(1))[0]
            image.append(pixel)
        if label < 4:
            testing_labels_array.append(label)
            testing_images_array.append(image)

    return np.array(training_images_array), np.array(training_labels_array), np.array(testing_images_array), np.array(
        testing_labels_array)


# Plots points of Χ based on the classes represented by the array labels.
def plot(X, labels, title):
    color = ['red', 'green', 'cyan', 'yellow']
    colors = np.empty(len(X), dtype=object)
    for i in range(0, len(X)):
        colors[i] = color[labels[i]]
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.title(title)
    plt.show()


# Calculate classification accuracy and return the percentage
# of the correct classified samples based on labels param.
def clf_accuracy(classes, labels):
    correct_labels = 0
    for i in range(0, classes.size):
        if classes[i] == labels[i]:
            correct_labels += 1
    return (correct_labels / classes.size) * 100


# My implementation of PCA algorithm for reduction to V dimensions.
def PCA(X, y, dimensions):
    # Mean Normalize M.
    mean = np.mean(X, 0)
    normalized_X = X - mean
    normalized_y = y - mean

    # Calculate Covariance Matrix.
    covariance_matrix = np.cov(normalized_X.T)

    #  Find eigenValues and eigenVectors.
    eigenValues, eigenVectors = np.linalg.eig(covariance_matrix)

    # Sort eigenValues and eigenVectors.
    indexes = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[indexes]
    eigenVectors = eigenVectors[:, indexes]

    # Find Reduced Dimensions Matrix.
    eigenValues_reduced = eigenValues[:dimensions]
    U_reduced = eigenVectors[:, :dimensions]

    # Calculate transformed X.
    X_transformed = np.dot(normalized_X, U_reduced).real
    y_transformed = np.dot(normalized_y, U_reduced).real
    print("Using " + str(dimensions) + " dimensions, PCA is able to retain " + str(
        abs(100 * (np.sum(eigenValues_reduced) / np.sum(eigenValues)))) + " % of information")

    return X_transformed, y_transformed


# My implementation of the K Nearest Neighbor Classifier
def K_Nearest_Neighbors_Classifier(train_samples, train_labels, test_samples, K):
    predictions = np.full(test_samples.shape[0], -1)
    # For each test sample
    for i in range(test_samples.shape[0]):
        dists = []
        # For each train sample
        for j in range(train_samples.shape[0]):
            # Calculate the distance
            dist = np.linalg.norm(train_samples[j] - test_samples[i])
            # Insert the calculated distance in the empty list paired with the label of that distance
            dists.append((dist, train_labels[j]))

        # Sort the distances
        dists = sorted(dists)
        # Pick the first K distances
        dists = dists[:K]

        # Transform the list into a 2D np array so that
        # a pair (distance, label) is now separated into two columns
        dists = np.array(dists)
        # The second column of the above array is the labels of the kNearestNeighbors
        labels = dists[:, 1]

        # Count the frequency of each class in the kNearestNeighbors for this instance
        counts = np.zeros(NUM_OF_CLASSES)
        for m in range(K):
            counts[int(labels[m])] += 1

        # Make the prediction based on the most frequent class in the neighborhood.
        predictions[i] = np.argmax(counts)
    return predictions


# My implementation of the Nearest Class Centroid Classifier
def Nearest_Class_Centroid_Classifier(train_samples, train_labels, test_samples):
    st_time = time.time()
    predictions = np.full(test_samples.shape[0], -1)
    centroids = np.zeros((NUM_OF_CLASSES, train_samples.shape[1]))
    # Sum the value of each characteristic for each class
    for i in range(train_samples.shape[0]):
        for j in range(train_samples.shape[1]):
            centroids[train_labels[i]][j] += train_samples[i][j]

    # Divide with the number of samples in each class in order to calculate the class centroid
    for i in range(NUM_OF_CLASSES):
        samples_in_class = (train_labels == i).sum()
        centroids[i] /= samples_in_class
    train_time = time.time() - st_time

    for i in range(test_samples.shape[0]):
        # Calculate the distance of the sample from each centroid
        dists = np.zeros(NUM_OF_CLASSES)
        for j in range(NUM_OF_CLASSES):
            dists[j] = np.linalg.norm(centroids[j] - test_samples[i])

        # Make the prediction based on minimum distance from centroid
        predictions[i] = np.argmin(dists)

    return predictions, train_time


def K_means(X, K):
    km = KMeans(n_clusters=K).fit(X)
    return km.cluster_centers_


def rand_init(X, K):
    [num_of_samples, num_of_features] = X.shape
    centers = np.zeros((K, num_of_features))

    # np.random.randint returns ints from [low, high) !
    idx = np.random.randint(num_of_samples, size=K)
    centers = X[idx, :]

    return centers


class RBF:

    def __init__(self, X, y, test_X, test_y, k, use_K_Means=True):
        self.X = X
        self.y = y
        self.test_X = test_X
        self.test_y = test_y
        self.k = k
        self.use_K_Means = use_K_Means

    def to_one_hot(self, data):
        arr = np.zeros((len(data), NUM_OF_CLASSES))
        for i in range(len(data)):
            arr[i][int(data[i])] = 1
        return arr

    def gaussian(self, x, c, s):
        d = np.linalg.norm(x - c)
        return 1 / np.exp(-d / s ** 2)

    def rbf_list(self, X, centers, std_list):
        RBF_list = []
        for x in X:
            RBF_list.append([self.gaussian(x, c, s) for (c, s) in zip(centers, std_list)])
        return np.array(RBF_list)

    def fit(self):
        st_time = time.time()
        if self.use_K_Means:
            self.C = K_means(self.X, self.k)
        else:
            self.C = rand_init(self.X, self.k)

        maxDist = np.max([np.linalg.norm(c1 - c2) for c1 in self.C for c2 in self.C])
        self.sigmas = np.repeat(maxDist / np.sqrt(2 * self.k), self.k)

        rbf_X = self.rbf_list(self.X, self.C, self.sigmas)
        self.w = np.linalg.pinv(rbf_X.T @ rbf_X) @ rbf_X.T @ self.to_one_hot(self.y)

        el_time = time.time() - st_time

        self.predictions = rbf_X @ self.w
        self.predictions = np.array([np.argmax(x) for x in self.predictions])
        diff = self.predictions - self.y

        print('Training Accuracy in ' + str(el_time) + ' seconds, for k=' + str(self.k) + ' and Use_K_Means=' + str(self.use_K_Means) + ': ', len(np.where(diff == 0)[0]) / len(diff))

        RBF_list_test = self.rbf_list(self.test_X, self.C, self.sigmas)
        self.predictions = RBF_list_test @ self.w
        self.predictions = np.array([np.argmax(x) for x in self.predictions])
        diff = self.predictions - self.test_y

        print('Testing Accuracy for k=' + str(self.k) + ' and Use_K_Means=' + str(self.use_K_Means) + ': ', len(np.where(diff == 0)[0]) / len(diff))


if __name__ == '__main__':
    # Initialize the arrays containing the train/test sets and train/test labels.
    training, training_labels, testing, testing_labels = Load_MNIST()

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(training)
    norm_training = scaler.transform(training)
    norm_testing = scaler.transform(testing)

    n = 75
    training_PCA, testing_PCA = PCA(norm_training, norm_testing, n)

    RBFNet = RBF(training_PCA, training_labels, training_PCA, training_labels, k=4)
    RBFNet.fit()

    RBFNet = RBF(training_PCA, training_labels, training_PCA, training_labels, k=10)
    RBFNet.fit()

    RBFNet = RBF(training_PCA, training_labels, training_PCA, training_labels, k=75)
    RBFNet.fit()

    RBFNet = RBF(training_PCA, training_labels, training_PCA, training_labels, k=150)
    RBFNet.fit()

    RBFNet = RBF(training_PCA, training_labels, training_PCA, training_labels, k=300)
    RBFNet.fit()

    RBFNet = RBF(training_PCA, training_labels, training_PCA, training_labels, k=4, use_K_Means=False)
    RBFNet.fit()

    RBFNet = RBF(training_PCA, training_labels, training_PCA, training_labels, k=10, use_K_Means=False)
    RBFNet.fit()

    RBFNet = RBF(training_PCA, training_labels, training_PCA, training_labels, k=75, use_K_Means=False)
    RBFNet.fit()

    RBFNet = RBF(training_PCA, training_labels, training_PCA, training_labels, k=150, use_K_Means=False)
    RBFNet.fit()

    RBFNet = RBF(training_PCA, training_labels, training_PCA, training_labels, k=300, use_K_Means=False)
    RBFNet.fit()