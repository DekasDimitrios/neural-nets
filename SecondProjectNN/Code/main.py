import struct as st
import time
import numpy as np
from scipy.spatial.distance import minkowski, cityblock, chebyshev
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


NUM_OF_CLASSES = 2


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


if __name__ == '__main__':
    # Initialize the arrays containing the train/test sets and train/test labels.
    training, training_labels, testing, testing_labels = Load_MNIST()

    # Transform the 4 classes (0, 1, 2, 3) into 2 (Odds, Even).
    training_labels = training_labels % 2
    testing_labels = testing_labels % 2

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(training)
    norm_training = scaler.transform(training)
    norm_testing = scaler.transform(testing)

    n = 75
    training_PCA, testing_PCA = PCA(norm_training, norm_testing, n)
    # plot(training_PCA, training_labels, "PCA Method")

    preds, el_time = Nearest_Class_Centroid_Classifier(training_PCA, training_labels, testing_PCA)
    acc = clf_accuracy(preds, testing_labels)
    print("NC Whole AB " + str(el_time) + "  " + str(acc))

    # --------------------- Linear SVM ---------------------

    # Find best parameters via 10-Fold Cross Validation
    # my_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    # myCV = KFold(n_splits=10, random_state=None, shuffle=False)
    # grid = GridSearchCV(SVC(kernel='linear'), param_grid=my_param_grid, cv=myCV, verbose=3)
    # grid.fit(training_PCA, training_labels)
    #
    # print("Best parameter %s" % grid.best_params_)

    # Training SVM with CV results

    classifier = SVC(kernel='linear', C=0.1)
    st_time = time.time()
    classifier.fit(training_PCA, training_labels)
    el_time = time.time() - st_time
    print("LSVM Training Time: " + str(el_time))

    preds = classifier.predict(training_PCA)
    print("LSVM Training Accuracy: " + str(clf_accuracy(preds, training_labels)))

    preds = classifier.predict(testing_PCA)
    print("LSVM Testing Accuracy: " + str(clf_accuracy(preds, testing_labels)))

    # --------------------- Polynomial SVM ---------------------

    # Find best parameters via 10-Fold Cross Validation
    # my_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                  'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #                  'coef0': [0, 1],
    #                  'degree': [2, 3, 4, 5, 6]}
    # myCV = KFold(n_splits=10, random_state=None, shuffle=False)
    # grid = GridSearchCV(SVC(kernel='poly'), param_grid=my_param_grid, cv=myCV, verbose=3)
    # grid.fit(training_PCA, training_labels)
    #
    # print("Best parameters %s" % grid.best_params_)
    # print("Best score is %s" % grid.best_score_)

    # Training SVM with CV results

    classifier = SVC(kernel='poly', C=1, coef0=0, degree=3)
    st_time = time.time()
    classifier.fit(training_PCA, training_labels)
    el_time = time.time() - st_time
    print("Polynomial SVM Training Time: " + str(el_time))

    preds = classifier.predict(training_PCA)
    print("Polynomial SVM Training Accuracy: " + str(clf_accuracy(preds, training_labels)))

    preds = classifier.predict(testing_PCA)
    print("Polynomial SVM Testing Accuracy: " + str(clf_accuracy(preds, testing_labels)))

    # --------------------- Gaussian(RBF) SVM ---------------------

    # Find best parameters via 10-Fold Cross Validation
    # my_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                  'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    # myCV = KFold(n_splits=10, random_state=None, shuffle=False)
    # grid = GridSearchCV(SVC(kernel='rbf'), param_grid=my_param_grid, cv=myCV, verbose=3)
    # grid.fit(training_PCA, training_labels)
    #
    # print("Best parameters %s" % grid.best_params_)
    # print("Best score is %s" % grid.best_score_)
    #
    # Training SVM with CV results

    classifier = SVC(kernel='rbf', C=10)
    st_time = time.time()
    classifier.fit(training_PCA, training_labels)
    el_time = time.time() - st_time
    print("RBF SVM Training Time: " + str(el_time))

    preds = classifier.predict(training_PCA)
    print("RBF SVM Training Accuracy: " + str(clf_accuracy(preds, training_labels)))

    preds = classifier.predict(testing_PCA)
    print("RBF SVM Testing Accuracy: " + str(clf_accuracy(preds, testing_labels)))

    # --------------------- Sigmoid SVM ---------------------

    # Find best parameters via 10-Fold Cross Validation
    # my_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    #                  'coef0': [0, 1]}
    # myCV = KFold(n_splits=10, random_state=None, shuffle=False)
    # grid = GridSearchCV(SVC(kernel='sigmoid'), param_grid=my_param_grid, cv=myCV, verbose=3)
    # grid.fit(training_PCA, training_labels)
    #
    # print("Best parameters %s" % grid.best_params_)
    # print("Best score is %s" % grid.best_score_)

    # Training SVM with CV results

    classifier = SVC(kernel='sigmoid', C=100, gamma=0.001, coef0=0)
    st_time = time.time()
    classifier.fit(training_PCA, training_labels)
    el_time = time.time() - st_time
    print("Sigmoid SVM Training Time: " + str(el_time))

    preds = classifier.predict(training_PCA)
    print("Sigmoid SVM Training Accuracy: " + str(clf_accuracy(preds, training_labels)))

    preds = classifier.predict(testing_PCA)
    print("Sigmoid SVM Testing Accuracy: " + str(clf_accuracy(preds, testing_labels)))
