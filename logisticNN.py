import numpy as np
import matplotlib.pyplot as plt
import h5py #Comment this out if you don't have a dataset on h5. Instead, directly import local dataset of images.
import scipy
from PIL import Image 
from scipy import ndimage
from lr_utils import load_dataset

#This loads the entire dataset from h5.
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

#This part is just used to test if the previous portion of the code written is outputting correct.

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))

##
#The best way to approach ML is to standardize the entire training or test data set. Basically, standardizing everything.
#So, here we will standardize images by diving them by 255
##

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# Computing a sigmoid function which can be further used to calculate different formulas.

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

##
# Time to initialize all the parameters with zeros.
# This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
# dim -- size of the w vector we want.
#

def initialize_with_zeros(dim):

    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))


#  w -- initialized vector of shape (dim, 1)
#  b -- initialized scalar
    return w, b

dim = 2
w, b = initialize_with_zeros(dim)

print ("w = " + str(w))
print ("b = " + str(b))


### At this point all the parameters are initialized.
### This allows to perform forward and backward propagation steps to study the parameters

##
#Implemeting the cost function and the gradient
#   Arguments:
#    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
#    b -- bias, a scalar
#    X -- data of size (num_px * num_px * 3, number of examples)
#    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
##

def propagate(w, b, X, Y):

    m = X.shape[1]

    # Forward propagation (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X) + b)     # compute activation
    cost = (-1/m)*(np.sum(np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T)))  # compute cost (For the actual calculus formula of cost function check the read me)

    # Backward propagation to find the gradient
    dw = (1/m)*np.dot(X, (A-Y).T)
    db = (1/m)*np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])

grads, cost = propagate(w, b, X, Y)

print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))


##
# This function optimizes w and b by running a gradient descent algorithm
# Update the parameters using gradient descent rule for w and b.
# We can use propagate() to calc. cost and gradient for the current parameters.
##

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    costs = []

    for i in range(num_iterations):


        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        w = w-learning_rate*dw
        b = b-learning_rate*db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

##
# Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
##

def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T,X) + b)


    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0


    assert(Y_prediction.shape == (1, m))

# Y_prediction -- a numpy array vector containing all predictions (0/1) for the examples in X
    return Y_prediction

w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])

print ("predictions = " + str(predict(w, b, X)))
