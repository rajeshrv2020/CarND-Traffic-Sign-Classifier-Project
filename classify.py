# Load pickled data
import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile

print('All modules imported.')

# TODO: Fill this in based on where you saved the training and testing data
training_file   = './dataset/train.p'
validation_file = './dataset/valid.p'
testing_file    = './dataset/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test   = test['features'] , test['labels']
        

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(x_train)

# TODO: Number of validation examples
n_validation = len(x_valid)

# TODO: Number of testing examples.
n_test = len(x_test)

# TODO: What's the shape of an traffic sign image?
image_shape = np.array(x_train[0]).shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

sign_classes, class_indices, class_counts = np.unique(y_train, return_index = True, return_counts = True)
test_sign_classes, test_class_indices, test_class_counts = np.unique(y_test, return_index = True, return_counts = True)

print("Number of training examples =", n_train)
print("Number of testing examples  =", n_validation)
print("Number of testing examples  =", n_test)
print("Image data shape            =", image_shape)
print("Number of classes           =", n_classes)


### Preprocess the data here. It is required to normalize the data.  Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

def grayscale(img): 
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    img = img.reshape(img.shape + (1,))
    return img

def normalize_image(img) :
    return ((img-128)/128)


x_train = normalize_image(grayscale(x_train))
x_valid = normalize_image(grayscale(x_valid))
x_test = normalize_image(grayscale(x_test))


### Define your architecture here.
### Feel free to use as many code cells as needed.

## LeNet Standard architecture code

import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

def LeNet(x, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x32.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 32), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # pooling. Input = 32x32x32. Output = 16x16x32.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Layer 2: Convolutional. Output = 16x16x32.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b    
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # pooling. Input = 16x16x32. Output = 8x8x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # Layer 3: Convolutional. Output = 8x8x32.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(32))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b    
    # Activation.
    conv3 = tf.nn.relu(conv3)
    # Dropout
    conv3 = tf.nn.dropout(conv3, keep_prob)
    
    # Layer 4: Convolutional. Output = 8x8x32.
    conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(32))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b    
    # Activation.
    conv4 = tf.nn.relu(conv4)
    # Dropout
    conv4 = tf.nn.dropout(conv4, keep_prob)    
    
    # Flatten. Input = 8x8x32. Output = 2048.
    fc0   = flatten(conv4)
    
    # Layer 6: Fully Connected. Input = 2048. Output = 128.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(2048, 128), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(128))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b    
    # Activation.
    fc1    = tf.nn.relu(fc1)
    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 7: Fully Connected. Input = 128. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(128, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b    
    #Activation.
    fc2    = tf.nn.relu(fc2)
    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 8: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    
    return logits


# Train your model here.

def get_loss_accuracy ( x, y , keep_prob, training_operation, loss, accuracy, x_data, y_data , kp, batch_size ) :
    
    num_of_samples = len(x_data)
    
    sess = tf.get_default_session()
    for offset in range(0, num_of_samples, batch_size) :
        end_offset = offset + batch_size
        batch_x , batch_y = x_data[offset:end_offset], y_data[offset : end_offset]
        
        data_loss     = 0
        data_accuracy = 0

        _,batch_loss = sess.run([training_operation,loss], 
                                        feed_dict={  
                                                     x: batch_x, 
                                                     y: batch_y, 
                                                     keep_prob: kp 
                                                  }
                               ) 
          
        batch_accuracy  = sess.run(accuracy, 
                                   feed_dict={
                                                x: batch_x,
                                                y: batch_y,
                                                keep_prob: kp
                                            }
                                  )
        
        # print('Batch {:>3} -Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
        #                                         (offset/batch_size),
        #                                         batch_loss,
        #                                         batch_accuracy))
        
        data_accuracy += (batch_accuracy * num_of_samples)
        data_loss += batch_loss
        
    return data_loss, (data_accuracy/num_of_samples)

def get_accuracy ( x, y , keep_prob, accuracy, x_data, y_data , batch_size ) :
    
    num_of_samples = len(x_data)
    
    sess = tf.get_default_session()
    for offset in range (0, num_of_samples, batch_size) :
        end_offset = offset + batch_size
        batch_x , batch_y = x_data[offset:end_offset], y_data[offset : end_offset]
       
        data_accuracy = 0
        
        # Calculate batch loss and accuracy
        batch_accuracy  = sess.run(accuracy, 
                                    feed_dict={
                                                x : batch_x,
                                                y : batch_y,
                                                keep_prob : 1.0
                                              }
                                  )
        #print('Batch {:>3} - Validation Accuracy: {:.6f}'.format((offset/batch_size),batch_accuracy))
        data_accuracy += (batch_accuracy * num_of_samples)
        
    return (data_accuracy/num_of_samples)

def train_model (x_train, y_train, x_valid, y_valid, learning_rate, kp, batch_size, epochs) :
    
    # Placeholder for Inputs
    x = tf.placeholder(tf.float32, (None, 32, 32, 1),name='x')
    y = tf.placeholder(tf.int32, (None),name='y')
    one_hot_y = tf.one_hot(y, n_classes)   
    keep_prob = tf.placeholder(tf.float32)

    # Generate Logits using LeNet 
    logits = LeNet(x, keep_prob)
    # Provide the logits the identifier
    logits = tf.identity(logits, name='logits')
  
    # Code straight from class
    loss = tf.reduce_mean(\
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(loss)
    
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    #            .minimize(loss)

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy     = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    saver = tf.train.Saver()
        
    # Launch the graph
    with tf.Session() as sess:
        # Initialize the session
        sess.run(tf.global_variables_initializer())
                 
        # Start Training
        print ('Started the training process ...')
     
    
        for epoch in range(epochs):

            # Shuffle the dataset duing the each epoch
            # Machines remember the old data very well 
            x_train, y_train = shuffle(x_train,y_train)
            x_valid, y_valid = shuffle(x_valid,y_valid)

            train_loss, train_accuracy = get_loss_accuracy ( x,
                                                             y,
                                                             keep_prob,
                                                             training_operation,
                                                             loss,
                                                             accuracy,
                                                             x_train,
                                                             y_train,
                                                             kp,
                                                             batch_size
                                                           )
            
            valid_accuracy = get_accuracy ( x, y, keep_prob, accuracy, x_valid, y_valid, batch_size )
            
        print("EPOCH {}: Training Accuracy = {:.3f} -- Validation Accuracy = {:.3f} -- Loss = {:.3f}"
                        .format(epoch+1, train_accuracy, valid_accuracy, train_loss))
        
        saver.save(sess, './model/Model')
        print("Model Saved")


### Once a final model architecture is selected, 
epochs = 10
batch_size = 128
learning_rate = 0.001
keep_prob = 0.6

tf.reset_default_graph()
with tf.Graph().as_default():
    train_model (x_train, y_train, x_valid, y_valid, learning_rate, keep_prob,  batch_size, epochs )

