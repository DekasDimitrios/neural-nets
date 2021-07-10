import struct as st
import time
import numpy as np
from scipy.spatial.distance import minkowski, cityblock, chebyshev
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

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
        if label < NUM_OF_CLASSES:
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
        if label < NUM_OF_CLASSES:
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


# Transform X array using the average brightnesses of array's X rows and columns.
def bright_trans(X):
    # Reshape every samples' features into a 28x28 image.
    X = X.reshape(X.shape[0], 28, 28)
    X_transformed = np.zeros((X.shape[0], 2))
    # For every sample.
    for k in range(0, X.shape[0]):
        row_sum = np.zeros(X.shape[1])
        # Find Average Row Brightness.
        for i in range(0, X.shape[1]):
            for j in range(0, X.shape[2]):
                row_sum[i] += X[k][i][j]
            row_sum[i] /= X.shape[1]
        for i in range(0, X.shape[1]):
            X_transformed[k][0] += row_sum[i] / X.shape[1]
        # Find Average Column Brightness.
        col_sum = np.zeros(X.shape[2])
        for j in range(0, X.shape[2]):
            for i in range(0, X.shape[1]):
                col_sum[j] += X[k][i][j]
            col_sum[j] /= X.shape[2]
        for j in range(0, X.shape[2]):
            X_transformed[k][1] += col_sum[j] / X.shape[2]
    return X_transformed


# My implementation of PCA algorithm for reduction to V dimensions.
def PCA(X, dimensions):
    # Mean Normalize M.
    mean = np.mean(X, 0)
    normalized_X = X - mean

    # Calculate Covariance Matrix.
    covariance_matrix = np.cov(normalized_X.T)

    #  Find eigenValues and eigenVectors.
    eigenValues, eigenVectors = np.linalg.eig(covariance_matrix)

    # Sort eigenValues and eigenVectors.
    indexes = eigenValues.argsort()[::-1]
    eigenVectors = eigenVectors[:, indexes]

    # Find Reduced Dimensions Matrix.
    U_reduced = eigenVectors[:, :dimensions]

    # Calculate transformed X.
    X_transformed = np.dot(normalized_X, U_reduced).real

    return X_transformed


# My implementation of the K Nearest Neighbor Classifier
def K_Nearest_Neighbors_Classifier(train_samples, train_labels, test_samples, K):
    predictions = np.full(test_samples.shape[0], -1)
    # For each test sample
    for i in range(test_samples.shape[0]):
        dists = []
        # For each train sample
        for j in range(train_samples.shape[0]):
            # Calculate the distance
            dist = np.linalg.norm(train_samples[j], test_samples[i], 3)
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
        print(i)
    return predictions


# My implementation of the Nearest Class Centroid Classifier
def Nearest_Class_Centroid_Classifier(train_samples, train_labels, test_samples):
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

    for i in range(test_samples.shape[0]):
        # Calculate the distance of the sample from each centroid
        dists = np.zeros(NUM_OF_CLASSES)
        for j in range(NUM_OF_CLASSES):
            dists[j] = np.linalg.norm(centroids[j] - test_samples[i])

        # Make the prediction based on minimum distance from centroid
        predictions[i] = np.argmin(dists)

    return predictions


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    # Initialize the arrays containing the train/test sets and train/test labels.
    # training, training_labels, testing, testing_labels = Load_MNIST()

    # Use average brightness method to reduce 784 features of each sample to 2 and plot training_bright.
    # training_bright = bright_trans(training)
    # plot(training_bright, training_labels, "Average Brightness Method")
    # testing_bright = bright_trans(testing)

    # st_time = time.time()
    # preds = Nearest_Class_Centroid_Classifier(training, training_labels, testing)
    # el_time = time.time() - st_time
    # acc = clf_accuracy(preds, testing_labels)
    # print("NC Whole " + str(el_time) + "  " + str(acc))
    #
    # st_time = time.time()
    # preds = Nearest_Class_Centroid_Classifier(training_bright, training_labels, testing_bright)
    # el_time = time.time() - st_time
    # acc = clf_accuracy(preds, testing_labels)
    # print("NC Whole AB " + str(el_time) + "  " + str(acc))

    train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # Selecting first 4 classes by selecting the proper indexes
    indexes = (train.targets == 0) | (train.targets == 1) | (train.targets == 2) | (train.targets == 3)
    train.targets = train.targets[indexes]
    train.data = train.data[indexes]

    indexes = (test.targets == 0) | (test.targets == 1) | (test.targets == 2) | (test.targets == 3)
    test.targets = test.targets[indexes]
    test.data = test.data[indexes]

    training_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    testing_set = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    network = Network()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    EPOCHS = 50
    for epoch in range(EPOCHS):
        for data in training_set:
            # data is already in a batch form of samples and labels
            X, y = data
            # Set the network on "train" mode
            network.zero_grad()
            network.train()
            # Get the network results for the current batch samples
            output = network(X.view(-1, 28*28))
            # Get the loss of the results in contrast to the desired output
            loss = F.nll_loss(output, y)
            # Back-propagate the loss computed in the output layer back into the network
            loss.backward()
            # Adjust the network weights
            optimizer.step()
        loss_value = loss.item()
        print("Epoch Number " + str(epoch) + ": Training Loss " + str(loss_value))
        if loss_value < 0.0005:
            break

    correct = 0
    total = 0

    # Set the network on "test" mode
    with torch.no_grad():
        network.eval()
        for data in testing_set:
            X, y = data
            output = network(X.view(-1, 28*28))
            for index, i in enumerate(output):
                if torch.argmax(i) == y[index]:
                    correct += 1
                total += 1
    print("Accuracy: ",     (correct/total)*100)

    n = 0
    plt.imshow(X[n].view(28, 28))
    plt.show()
    print(torch.argmax(network(X[n].view(-1, 28*28))[0]))

    n = 1
    plt.imshow(X[n].view(28, 28))
    plt.show()
    print(torch.argmax(network(X[n].view(-1, 28*28))[0]))

    n = 2
    plt.imshow(X[n].view(28, 28))
    plt.show()
    print(torch.argmax(network(X[n].view(-1, 28*28))[0]))

    n = 3
    plt.imshow(X[n].view(28, 28))
    plt.show()
    print(torch.argmax(network(X[n].view(-1, 28*28))[0]))
