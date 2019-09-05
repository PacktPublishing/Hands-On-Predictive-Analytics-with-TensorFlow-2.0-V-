import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing



df = pd.read_csv('rating.csv', sep=',', names=['user', 'item', 'rating', 'timestamp'], header=None)
df = df.drop('timestamp', axis=1)

num_items = df.item.nunique()
num_users = df.user.nunique()

print("USERS: {} ITEMS: {}".format(num_users, num_items))



#Pandas will load training set into a DataFrame with three columns: user, item and ratings.
# Normalize in [0, 1]

r = df['rating'].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(r.reshape(-1,1))
df_normalized = pd.DataFrame(x_scaled)
df['rating'] = df_normalized

#Pandas DataFrames cannot be directly used to feed a model, the best option is to convert a DataFrame into a matrix:

# Convert DataFrame in user-item matrix

matrix = df.pivot(index='user', columns='item', values='rating')
matrix.fillna(0, inplace=True)

#Rows in the matrix correspond to users and columns to items;
#therefore entries correspond to ratings given by users to items.
#Our matrix is still an object of DataFrame type, we need to convert it to a numpy matrix.


# Users and items ordered as they are in matrix

users = matrix.index.tolist()
items = matrix.columns.tolist()

matrix = matrix.as_matrix()



# Network Parameters

num_input = num_items
num_hidden_1 = 10
num_hidden_2 = 5

X = tf.placeholder(tf.float64, [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], dtype=tf.float64)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input], dtype=tf.float64)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([num_input], dtype=tf.float64)),
}


# Building the encoder

def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


# Building the decoder

def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


# Construct model

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)


# Prediction

y_pred = decoder_op


# Targets are the input data.

y_true = X



# Define loss and optimizer, minimize the squared error

loss = tf.losses.mean_squared_error(y_true, y_pred)
optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)

predictions = pd.DataFrame()

# Define evaluation metrics

eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)



# Initialize the variables (i.e. assign their default value)

init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()



#We can finally start to train our model.


