#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
get_ipython().run_line_magic('matplotlib', 'inline')
import tarfile
import os
from urllib.request import urlretrieve
import timeit


# In[5]:


def load_data_mnist():
    """
    Load the MNIST dataset. Reads the training and testing files and create matrices.
    :Expected return:
    train_data:the matrix with the training data
    test_data: the matrix with the data that will be used for testing
    y_train: the matrix consisting of one 
                        hot vectors on each row(ground truth for training)
    y_test: the matrix consisting of one
                        hot vectors on each row(ground truth for testing)
    """
    
    #load the train files
    df = None
    
    y_train = []

    for i in range( 10 ):
        tmp = pd.read_csv( 'mnist/train%d.txt' % i, header=None, sep=" " )
        #build labels - one hot vector
        hot_vector = [ 1 if j == i else 0 for j in range(0,10) ]
        
        for j in range( tmp.shape[0] ):
            y_train.append( hot_vector )
        #concatenate dataframes by rows    
        if i == 0:
            df = tmp
        else:
            df = pd.concat( [df, tmp] )

    train_data = df.as_matrix()
    y_train = np.array( y_train )
    
    #load test files
    df = None
    
    y_test = []

    for i in range( 10 ):
        tmp = pd.read_csv( 'mnist/test%d.txt' % i, header=None, sep=" " )
        #build labels - one hot vector
        
        hot_vector = [ 1 if j == i else 0 for j in range(0,10) ]
        
        for j in range( tmp.shape[0] ):
            y_test.append( hot_vector )
        #concatenate dataframes by rows    
        if i == 0:
            df = tmp
        else:
            df = pd.concat( [df, tmp] )

    test_data = df.as_matrix()
    y_test = np.array( y_test )
    
    return train_data, test_data, y_train, y_test


# In[6]:


"""Load from /home/USER/data/cifar10 or elsewhere; download if missing."""

def load_data_cifar10(path=None):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing CIFAR-10. Default is
            /home/USER/data/cifar10 or C:\Users\USER\data\cifar10.
            Create if nonexistant. Download CIFAR-10 if missing.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """
    url = 'https://www.cs.toronto.edu/~kriz/'
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser('~'), 'data', 'cifar10')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download tarfile if missing
    if tar not in os.listdir(path):
        urlretrieve(''.join((url, tar)), os.path.join(path, tar))
        print("Downloaded %s to %s" % (tar, path))

    # Load data from tarfile
    with tarfile.open(os.path.join(path, tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffr = np.zeros(fsize * 6, dtype='uint8')

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other extraneous stuff
        members = [file for file in tar_object if file.name in files]

        # Sort those members by name
        # -- Ensures we load train data in the proper order
        # -- Ensures that test data is the last file in the list
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte of each chunk is the label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

    # Labels are the first byte of every chunk
    labels = buffr[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape(-1, 3072).astype('float32') / 255

    # Split into train and test
    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]

    def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot

    return train_images, _onehot(train_labels),         test_images, _onehot(test_labels)


# In[2]:


def activation(a,activation_number):
    
    if activation_number == 1:
        return np.maximum(a, 0) + np.log(1 + np.exp(-np.abs(a)))
    elif activation_number == 2:
        return np.cos(a)
    else:
        return np.tanh(a)


# In[3]:


def activationDerivative(a,activation_number):

    if activation_number == 1:
        return np.exp(np.minimum(0,a))/(1+np.exp(-np.abs(a)))
    elif activation_number == 2:
        return -(np.sin(a))
    else:
        return 1 - np.tanh(a)**2

        


# In[4]:


def softmax( x, ax=1 ):
    m = np.max( x, axis=ax, keepdims=True )#max per row
    p = np.exp( x - m )
    return ( p / np.sum(p,axis=ax,keepdims=True) )   


# In[7]:


#z is the output of the hidden layer after activation
def output_z(X,w1,activation_number):
    
    z = activation(X.dot(w1.T), activation_number) 

    #add bias
    z = np.hstack((np.ones((z.shape[0], 1)), z))
    return z


# In[8]:


def test(w1,w2, X, activation_number):
    z = output_z(X,w1,activation_number)
    b = z.dot(w2.T)
    y = softmax(b)
    # Hard classification decisions
    ttest = np.argmax(y, 1)
    return ttest


# In[9]:


def cost_grad_softmax(w1,w2, X, T, lamda, activ):
    
    #feedforward
    Z = output_z(X,w1,activ)
    S = Z.dot(w2.T)
    Y = softmax(S)
    
    max_error = np.max(Y, axis=1)

    # Compute the cost function to check convergence
    Ew = np.sum(T * S) - np.sum(max_error) -          np.sum(np.log(np.sum(np.exp(S - np.array([max_error, ] * S.shape[1]).T), 1))) -          (0.5 * lamda) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))

    # calculate gradient for w2
    gradEw2 = (T-Y).T.dot(Z) - lamda*w2
    
    #remove the bias 
    w2_withoutBias = np.copy(w2[:, 1:])

    # derivative of the activation function
    z = activationDerivative(X.dot(w1.T), activ)

    # Calculate gradient for w1
    gradEw1 = ((T-Y).dot(w2_withoutBias)*z).T.dot(X) - lamda*w1

    return Ew, gradEw1, gradEw2


# In[10]:


def gradient_check (w1, w2, T, X, lamda, activ):

    epsilon = 1e-6
    
    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(T[_list, :])
    
    Ew, gradw1, gradw2 = cost_grad_softmax(w1, w2, x_sample, t_sample, lamda, activ)
    
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad 
    
    numericalGrad = np.zeros(gradw2.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numericalGrad.shape[0]):
        for d in range(numericalGrad.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp = np.copy(w2)
            w_tmp[k, d] += epsilon
            
            #approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            e_plus, _, _ =cost_grad_softmax(w1,w_tmp, x_sample, t_sample, lamda, activ)
        
            #subtract epsilon to the w[k,d]
            w_tmp = np.copy(w2)
            w_tmp[k, d] -= epsilon
            e_minus, _, _ =cost_grad_softmax(w1,w_tmp, x_sample, t_sample, lamda, activ)
            
            #approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)

    print ("The difference for gradient of w2 is : ", np.max(np.abs(gradw2 - numericalGrad)))

    
    numericalGrad = np.zeros(gradw1.shape)

    for k in range(numericalGrad.shape[0]):
        for d in range(numericalGrad.shape[1]):
            #add epsilon to the w[k,d]
            w_tmp = np.copy(w1)
            w_tmp[k, d] += epsilon
            
            #approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            e_plus, _, _ = cost_grad_softmax(w_tmp, w2, x_sample, t_sample, lamda, activ)
            
            #subtract epsilon to the w[k,d]
            w_tmp = np.copy(w1)
            w_tmp[k, d] -= epsilon
            
            #approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            e_minus, _, _ = cost_grad_softmax(w_tmp, w2, x_sample, t_sample, lamda, activ)
            numericalGrad[k,d] = (e_plus - e_minus) / (2 * epsilon)

    print ("The difference for gradient of w1 is : ", np.max(np.abs(gradw1 - numericalGrad)))

    


# In[120]:


def train(T, X, lamda, w1, w2, options):
        
    # Maximum number of iteration 
    epochs = options[0]

    # minibatch size
    minibatch = options[1]

    # Learning rate
    eta = options[2]
    
    
    # Activation Function
    activ = options[3]
    
    #fix eta for cifar only
    cifarParam = options[4]
    
    costs = []
    Ewold = -np.inf
    sequence = 0
    for i in range(epochs):
        print("epoch", i)

        #shuffle data 
        combined =list(zip(X,T)) 
        np.random.shuffle(combined)
        X_train, T_train = zip(*combined)
        
        X_train =np.asarray(X_train)
        T_train = np.asarray(T_train)         
            
        for x in range(0, X.shape[0], minibatch):
            
            X_train_mini = X_train[i:i + minibatch]
            T_train_mini = T_train[i:i + minibatch]
        
            Ew, gradEw1, gradEw2 = cost_grad_softmax(w1, w2, X_train_mini, T_train_mini, lamda, activ)
            
            # save cost
            costs.append(Ew)
                      
            # Update parameters based on gradient ascend
            w1 = w1 + eta*gradEw1
            w2 = w2 + eta*gradEw2
    
        print('Iteration : %d, Cost function :%f' % (i, Ew))
        if Ewold>Ew and cifarParam == 1 :
            sequence = sequence + 1
            
            if  sequence%2==0:
                eta = eta*0.1   
                print("fix eta", str(eta))
        else :
            sequence = 0
        Ewold = Ew 
        
        
            
    return w1, w2, costs


# In[122]:


## K is for categories
#D is for input
#M is for hidden layer

#default parameters
epochs = 400
minibatch = 150
eta = 0.05
lamda = 0.01
M = 300
cifarParam = -1

i = -1

while(i!=1 and i!=2):
    i = int(input("Choose which dataset to use: \n 1 for MNIST \n 2 for CIFAR \n "))

if i == 1:
    x_train,x_test, y_train, y_test = load_data_mnist()
    
    q = int(input("If you want to use the default parameters choose 1, otherwise press any other button"))
    if q != 1:
        epochs = int(input('Define the number of epochs:\n>'))
        minibatch = int(input('Define the size of the minibatch:\n>'))
        eta = float(input('Define the eta:\n>'))
        lamda = float(input('Define the lamda:\n>'))
        M = int(input('Define the number of neurons of the hidden layer:\n>'))    
else:
    x_train, y_train, x_test, y_test = load_data_cifar10()
    cifarParam = 1
    q = int(input("If you want to use the default parameters choose 1, otherwise press any other button"))
    if q != 1:
        epochs = int(input('Define the number of epochs:\n>'))
        minibatch = int(input('Define the size of the minibatch:\n>'))
        eta = float(input('Define the eta:\n>'))
        lamda = float(input('Define the lamda:\n>'))
        M = int(input('Define the number of neurons of the hidden layer:\n>'))    

Ν, D = x_train.shape

print ("insert activation number:")
activation_number = int(input("enter 1,2 or 3"))
options = [epochs, minibatch, eta, activation_number, cifarParam]

#normalization
x_train = x_train.astype(float)/255
x_test = x_test.astype(float)/255

K = y_train.shape[1]

#initialize weights
#randomly
#w1 = np.random.randn(M,D+1)
#w2 = np.random.randn(K,M+1)

#ReLu initialization
#w1 = np.random.randn(M,D+1)*np.sqrt(6/D)
#w2 = np.random.randn(K,M+1)*np.sqrt(6/D)

#Xavier initialization
w1 = np.random.randn(M,D+1)*np.sqrt(6/((D+1) + K)) 
w2 = np.random.randn(K,M+1)*np.sqrt(6/((D+1) + K)) 

#add bias
w1[:, 1] = 1
w2[: ,1] = 1


# make X and Y :N+1
x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

#q is for gradient check
q = int(input("If you want to perform gradient check press 1. "))


# In[123]:


#gradient check

if q==1:
    print("computing gradient check")
    gradient_check(w1, w2, y_train, x_train, lamda, activation_number)

#training time
start = timeit.default_timer()

#training
w1_final, w2_final, costs = train(y_train, x_train, lamda, w1, w2, options)

stop = timeit.default_timer()

print("trained")

training_time = stop - start
print('Time: ', training_time)  


# In[124]:


print(x_test.shape)
pred = test(w1_final, w2_final, x_test, activation_number)
print(eta)

faults = np.where(np.not_equal(np.argmax(y_test,1),pred))[0]
print(faults/y_test.shape[0])

#accuracy
print("accuracy= "+str(np.mean( pred == np.argmax(y_test,1))) )

error_count = np.not_equal(np.argmax(y_test, 1), pred).sum()
print ("error is: "+str(error_count / y_test.shape[0] * 100))

#iterateXcost by learning rate
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(format(options[2], 'f')))
plt.show()

# We save the output to a file
file = open("new.txt", "a")
file.write("\n Epochs: " +str(options[0])+"\n Mini batch used: " +str(options[1])+"\n Learning rate: "+str(options[2]))
file.write("\n activasion function: "+str(activation_number) + "\n Neurons on hiddel layer: " + str(M)+"\n Lamda value: "+str(lamda))
file.write("\n Error: "+str(error_count / y_test.shape[0] * 100)+"\n Training time was: "+str(training_time)+"\n")
file.close()

# In[121]:


#TRY ALL POSSIBLE PARAMETERS FOR PDF REPORT 
question = int(input("press 1 if you want to manually creaate the results for the pdf"))

if question == 1:

    epochs = 50
    dataset = ["cifar10"]
    mb = [100,200]
    Mhidden = [100,200,300]
    ha = [1,2,3]
    cifarParam = -1
    for d in dataset:
        for m in mb:
            for h in ha:
                for M in Mhidden:
                    if d == "mnist":
                        x_train,x_test, y_train, y_test = load_data_mnist()
                        eta = 0.001
                        lamda = 0.01
                    else : 
                        cifarParam = 1
                        x_train, y_train, x_test, y_test = load_data_cifar10()
                        eta = 0.01
                        lamda = 0.01
                    Ν, D = x_train.shape
                    options = [epochs, m, eta, h, cifarParam]

                    #normalization
                    x_train = x_train.astype(float)/255
                    x_test = x_test.astype(float)/255

                    K = y_train.shape[1]

                    #Xavier initialization
                    w1 = np.random.randn(M,D+1)*np.sqrt(6/((D+1) + K)) 
                    w2 = np.random.randn(K,M+1)*np.sqrt(6/((D+1) + K)) 

                    #add bias
                    w1[:, 1] = 1
                    w2[: ,1] = 1


                    # make X and Y :N+1
                    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
                    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

                    #training time
                    start = timeit.default_timer()

                    #training
                    w1_final, w2_final, costs = train(y_train, x_train, lamda, w1, w2, options)

                    stop = timeit.default_timer()

                    print("trained")

                    training_time = stop - start
                    print('Time: ', training_time)  

                    pred = test(w1_final, w2_final, x_test, h)

                    faults = np.where(np.not_equal(np.argmax(y_test,1),pred))[0]
                    #error
                    error_count = np.not_equal(np.argmax(y_test, 1), pred).sum()

                    # We save the output to a file
                    file = open(d + ".txt", "a")
                    file.write("\n Epochs: " +str(options[0])+ "\n Learning rate: "+str(options[2]) +"\n Lamda value: "+str(lamda))
                    file.write("\n Mini batch used: " +str(options[1]) + "\n activasion function: "+str(h) + "\n Neurons on hiddel layer: " + str(M))
                    file.write("\n Error: "+str(error_count / y_test.shape[0] * 100)+"\n Training time was: "+str(training_time)+"\n")
                    file.close()
                    
else: exit()