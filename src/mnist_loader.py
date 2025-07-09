from pathlib import Path
import struct
import array
import numpy as np

def read_image_data(filepath: Path) -> tuple:
    """
    This method loads the images (either training or testing) from the specified filepath and returns the image as a list of
    length 784 (28*28) numpy arrays as well as size of the dataset and the dimensions of each image in the dataset
    Args:
        filepath: Path to the binary file containing the image data
    
    Output:
        images: a list of 784x1 numpy arrays representing each image
        size: the size of the dataset
        rows: number of rows in each image
        cols: number of columns in each image
    """
    with filepath.open("rb") as file:
        _, size, rows, cols = struct.unpack(">IIII", file.read(16)) #The first 16 bytes contain information about the way in which the data is stored
        N = rows*cols
        data = array.array("B", file.read())
        images = []
        for i in range(size):
            image = np.array(data[N*i:N*(i+1)])/255 #Each pixel of the array contains an integer in the range 0-255 so we rescale the pixel values to the range 0...1
            images.append(np.reshape(image, (N, 1))) #We reshape the arrays to (N, 1) to make them behave nicely with matrix multiplication
        return images, size, rows, cols #We return the size of the datasize as well as the dimensions for testing purposes
    
def read_labels(filepath: Path) -> tuple:
    """
    Reads the label data from the given MNIST file and one-hot encodes it

    inputs:
        filepath: path to the data file
    outputs:
        labels: a list of one-hot encoded label vectors
        size: the size of the dataset
    """
    with filepath.open("rb") as file:
        _, size = struct.unpack(">II", file.read(8))
        data = array.array("B", file.read())
        labels = [np.zeros((10, 1)) for _ in range(size)]
        for i, label in enumerate(data):
            labels[i][label] = 1
        return labels, size

def make_batches(images: list, labels: list, batch_size: int) -> list:
    """
    Split the training data into batches where each batch contains a subset of the training data. 
    The size of each batch (except possibly the last one if the nmber of training examples is not 
    divisible by batch size) is the same

    inputs:
        train_images: a list of training images. Each training image is a numpy array
        train_labels: a list of corresponding labels
        batch size: the size of each batch
    """
    batches = []
    N = (len(images)-1)//batch_size+1 #Division that rounds up
    for i in range(N):
        batch_images, batch_labels = images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size]
        batches.append(list(zip(batch_images, batch_labels)))
    return batches
