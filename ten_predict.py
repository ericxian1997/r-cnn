# Imports
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from skimage.io import imread, imshow, imsave
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def cnn_model_fn(features, labels, mode):
  # Input layer
  input_layer = tf.reshape(features, [-1, 227, 227, 3])
  
  # Conv layer 1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=24,
      strides=4,
      kernel_size=[11, 11],
      activation=tf.nn.relu
      )

  # Pooling layer 1
  pool1 = tf.layers.max_pooling2d(
      inputs=conv1,
      pool_size=[3, 3],
      strides=2)

  # Nomorlization layer 1
  norm1 = tf.nn.local_response_normalization(
  	  pool1, 
  	  depth_radius = 2,
  	  alpha = 2e-05,
  	  beta = 0.75,
  	  bias = 1.0,
  	  name = 'norm1')
  
  # Conv layer 2
  conv2 = tf.layers.conv2d(
      inputs=norm1,
      filters=64,
      kernel_size=[5, 5],
      padding='SAME',
      activation=tf.nn.relu)

  # Pooling layer 2
  pool2 = tf.layers.max_pooling2d(
      inputs=conv2,
      pool_size=[3, 3],
      strides=2)

  # Nomorlization layer 2
  norm2 = tf.nn.local_response_normalization(
  	  pool2, 
  	  depth_radius = 2,
  	  alpha = 2e-05,
  	  beta = 0.75,
  	  bias = 1.0,
  	  name = 'norm2')

  # Conv layer 3
  conv3 = tf.layers.conv2d(
      inputs=norm2,
      filters=256,
      kernel_size=[3, 3],
      padding='SAME',
      activation=tf.nn.relu)

  # Conv layer 4
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=256,
      kernel_size=[3, 3],
      padding='SAME',
      activation=tf.nn.relu)
  
  # Conv layer 5
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=128,
      kernel_size=[3, 3],
      padding='SAME',
      activation=tf.nn.relu)
  
  # Pooling layer 5
  pool5 = tf.layers.max_pooling2d(
      inputs=conv5,
      pool_size=[3, 3],
      strides=2)

  # Dense layer(full-connected layer)
  flat = tf.reshape(pool5, [-1, 6*6*128])
  fc6 = fc(flat, 6*6*128, 4096, name='fc6')
  dropout6 = tf.nn.dropout(fc6, keep_prob=0.5)
  fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
  dropout7 = tf.nn.dropout(fc7, keep_prob=0.5)
  logits = tf.layers.dense(inputs=dropout7, units=10)

  loss = None
  train_op = None
  
  # Calculate Loss
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.001,
      optimizer='SGD')

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name='softmax_tensor')
      }

  # Return a ModelFnOps objects
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)
    
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)      
      return relu
    else:
      return act

def main(unused_argv):
  # Load classifier
  print('Begin to load')
  ten_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/ten_alex_convnet_model")
  print('Loading finished')

  # Load region proposals
  regions = np.load("C:/Users/eric/selectivesearch/cut_photo.npy")
  regions = regions[1:,:]
  regions = np.array(regions, dtype = np.float32)

  p1 = ten_classifier.predict(x=regions, as_iterable=True)
  max_p = 0;
  for i, p in enumerate(p1):
    #for k in p:
   		#print(k, p[k])
    print("Prediction %s: %s" % (i+1, p['classes']))
    print("Probabilities: %s " % p['probabilities'].max())
    if p['probabilities'].max() > max_p:
    	max_p = p['probabilities'].max()
    	index = i
    	predict = p['classes']
  print("Region: %s" % str(index+1))
  print("Predict: %s" % predict)
  print("Probabilities: %s" % max_p)

  # read 原图
  img = imread('C:/Users/eric/selectivesearch/example/dog (3).JPEG')
  endpoint = np.load("C:/Users/eric/selectivesearch/endpoint_4.npy")

  a = index+1
  name = ['airplane', 'bird', 'car', 'cat', 'deer',
  'dog', 'frog', 'horse', 'ship', 'truck']

  # draw rectangles on the original image
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
  ax.imshow(img)

  rect = mpatches.Rectangle(
      (endpoint[a][0], endpoint[a][1]), endpoint[a][2], endpoint[a][3], fill=False, edgecolor='red', linewidth=1)
  ax.add_patch(rect)
  ax.annotate(name[predict]+'  %.2f'%max_p, xy = (endpoint[a][0], endpoint[a][1]), fontsize=16, color='red')
  plt.show()  


  # Draw
if __name__ == "__main__":
  tf.app.run()
  
