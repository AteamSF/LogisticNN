import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = np.sum(train_set_x_orig.shape[0]) #Number of training examples
m_test = np.sum(test_set_x_orig.shape[0]) #Number of test examples in the data set
num_px = np.sum(train_set_x_orig.shape[1]) #Height and width of the image

#Eliminate this part once confirmed.

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

#This reshapes the training and test examples. Images of size (num_px, num_px, 3) are flattened into single vectors of shape ((num_px ∗ num_px ∗ 3, 1))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  #reshaping training examples
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T #reshaping test examples

#Eliminate this part once confirmed.

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
