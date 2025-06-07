import struct
import array
import numpy as np

class MNISTLoader:
    @classmethod
    def read_image_data(cls, filepath: str) -> tuple:
        """
        This method loads the images (either training or testing) from the specified filepath and returns the image as a list of
        length 784 (28*28) numpy arrays as well as size of the dataset and the dimensions of each image in the dataset
        Args:
            filepath: Path to the binary file containing the image data
        
        Output:

        """
        with open(filepath, "rb") as file:
            _, size, rows, cols = struct.unpack(">IIII", file.read(16)) #The first 16 bytes contain information about the way in which the data is stored
            N = rows*cols
            data = array.array("B", file.read())
            images = []
            for i in range(size):
                image = np.array(data[N*i:N*(i+1)])/256 #Each pixel of the array contains an integer in the range 0-255 so we rescale the pixel values to the range 0...1
                images.append(np.reshape(image, (N, 1))) #We reshape the arrays to (N, 1) to make them behave nicely with matrix multiplication
            return images, size, rows, cols #We return the size of the datasize as well as the dimensions for testing purposes
        
    @classmethod
    def read_labels(cls, filepath) -> array:
        with open(filepath, "rb") as file:
            _, size = struct.unpack(">II", file.read(8))
            return array.array("B", file.read()), size


if __name__=="__main__":
    #images_train = MNISTLoader.read_image_data("../data/train-images.idx3-ubyte")
    labels_train, _ = MNISTLoader.read_labels("../data/train-labels.idx1-ubyte")
    print(labels_train[0])