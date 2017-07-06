from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

#tf.logging.set_verbosity(tf.logging.INFO)

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

  norm2 = tf.nn.local_response_normalization(
  	  pool2, 
  	  depth_radius = 2,
  	  alpha = 2e-05,
  	  beta = 0.75,
  	  bias = 1.0,
  	  name = 'norm2')

  conv3 = tf.layers.conv2d(
      inputs=norm2,
      filters=256,
      kernel_size=[3, 3],
      padding='SAME',
      activation=tf.nn.relu)

  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=256,
      kernel_size=[3, 3],
      padding='SAME',
      activation=tf.nn.relu)

  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=128,
      kernel_size=[3, 3],
      padding='SAME',
      activation=tf.nn.relu)

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

  # Output layer
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
  # Load training and eval data
  print('Begin to load')
  #train_data = np.load("train_x_10.npy") # Returns np.array
  #train_labels = np.load("train_y_10.npy")
  eval_data = np.load("test_x_10.npy") # Returns np.array
  eval_labels = np.load("test_y_10.npy")
  print('Loading finished')

  # Create the Estimator
  ten_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/ten_alex_convnet_model")

  # Set up logging for predictions
  #tensors_to_log = {"probabilities": "softmax_tensor"}
  #logging_hook = tf.train.LoggingTensorHook(
        #tensors=tensors_to_log, every_n_iter=50)
  
  # Train the model
  ten_classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=100,
      steps=50000)
      #monitors=[logging_hook])
  
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
            }
  
  # Evaluate the model and print results
  eval_results = ten_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)
  

if __name__ == "__main__":
  tf.app.run()
  
