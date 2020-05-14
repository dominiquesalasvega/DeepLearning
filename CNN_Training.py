# CNN Training
# We will go over the detailed implementation of a training loop for a CNN model.

# You will have to identify overfitting scenarios and adjust your training process.

# Imports
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import importlib as ipl
import numpy as np
import random
import pickle
from timeit import default_timer as timer

print("TF version: {}".format(tf.__version__))
print("GPU availability: {}".format(tf.test.is_gpu_available()))


# $ !mkdir data
# $ !cd data
# $ !wget --content-disposition https://seafile.unistra.fr/f/d209c83b56ec441fa887/?dl=1
# $ !cd ..

DATA_PATH = "data/rps.pickle"

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

x_train = data["x_train"]
y_train = data["y_train"]
x_val   = data["x_val"]
y_val   = data["y_val"]
x_test  = data["x_test"]
y_test  = data["y_test"]

# QUESTION 1: Preview 10 images from the training set. Display the type of move (rock, paper or scissors) in the title.

class_names = ["rock", "paper", "scissors"]

fig, axes = plt.subplots(8, figsize=(20, 20))

for i, ax in enumerate(axes):
    ax.imshow(x_train[i])
    ax.axis("off")
    ax.set_title(class_names[y_train[i]])

# We will be packaging the data into the following TF datasets with a batch size of 16 for speed and convenience.

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(10000).batch(12)
val_dataset = val_dataset.batch(12)
test_dataset = test_dataset.batch(12)

#Model

class CNN(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self._conv_1 = layers.Conv2D(
      filters=16,
      kernel_size=5,
      activation="relu"
    )
    self._max_pool_1 = layers.MaxPooling2D(
      pool_size=(5,5)
    )
    self._conv_2 = layers.Conv2D(
      filters=32,
      kernel_size=3,
      activation="relu"
    )
    self._flatten = layers.Flatten()
    self._dense_1 = layers.Dense(
      units=2048,
      activation="relu"
    )
    self._dense_2 = layers.Dense(
      units=4096,
      activation="relu"
    )
    self._dense_3 = layers.Dense(
      units=3,
      activation="softmax"
    )

  def call(self, x):
    res = x
    res = self._conv_1(res)
    res = self._max_pool_1(res)
    res = self._conv_2(res)
    res = self._flatten(res)
    res = self._dense_1(res)
    res = self._dense_2(res)
    res = self._dense_3(res)
    return res

cnn_0 = CNN()

# Blank forward pass to trigger parameter allocation
inp = np.zeros([1, 128, 128, 3], dtype=np.float32)
_ = cnn_0(inp)

param_save = cnn_0.get_weights()

# Here is an overview of the model's architecture:
cnn_0.summary()

@tf.function
def train_step(inp, labels, cnn_model):
    outp = cnn_model(inp)
    loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(labels, outp))
    return loss

# Similarly, the following function computes the accuracy of the model on a batch
@tf.function
def eval_step(inp, labels, cnn_model):
    outp = cnn_model(inp)
    pred = tf.cast(tf.argmax(outp, axis=1), tf.int32)
    n_correct = tf.reduce_sum(
        tf.cast(
            tf.equal(pred, labels),
            tf.float32
        )
    )
    acc = n_correct / tf.cast(tf.shape(outp)[0], tf.float32)
    return acc

# QUESTION 2: Run the model on the training set for 100 iterations. Report the loss at each iteration.
cnn_0.set_weights(param_save)
times = list(range(50))
losses = []

for id_iter, (x_train_in, y_train_in) in zip(times, train_dataset):
    losses.append(train_step(x_train_in, y_train_in, cnn_0))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, losses)
ax.set_xlabel("iteration")
ax.set_ylabel("classification loss")

# QUESTION 3: Fix the train_step function to incorporate the update:
LEARNING_RATE = 0.01

@tf.function
def train_step(inp, labels, cnn_model):
    with tf.GradientTape() as tape:
        outp = cnn_model(inp)
        loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(labels, outp))
    gradients = tape.gradient(loss, cnn_model.trainable_variables)
    for grad, var in zip(gradients, cnn_model.trainable_variables):
        var.assign(var - LEARNING_RATE * grad)
    return loss

# QUESTION 4: Train the model on the training set for 6000 iterations. Every 200th iteration, report the average loss over the 200 previous iterations. Plot it.
start_t = timer()

cnn_0.set_weights(param_save)

average_training_losses = []
training_losses = []

for id_iter, (x_train_in, y_train_in) in zip(range(6000), train_dataset.repeat()):
    training_losses.append(train_step(x_train_in, y_train_in, cnn_0))
    if id_iter % 200 == 0:
        average_training_losses.append(sum(training_losses) / len(training_losses))
        training_losses = []
        print("ITER  {}: training loss       = {:.4f}".format(id_iter, average_training_losses[-1]))
        print("______________________________________")

end_t = timer()
print("time_elapsed: {}".format(end_t - start_t))

times = list(range(len(average_training_losses)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, average_training_losses)
ax.set_xlabel("iteration * 100")
ax.set_ylabel("classification loss")

# QUESTION 5: Evaluate (report the average accuracy) on the training set, then on the test set.
train_accs = []
for x_train_in, y_train_in in train_dataset:
    train_accs.append(eval_step(x_train_in, y_train_in, cnn_0))

average_train_acc = sum(train_accs) / len(train_accs)
print("ACCURACY - TRAINING SET: {}".format(average_train_acc))

test_accs = []
for x_test_in, y_test_in in test_dataset:
    test_accs.append(eval_step(x_test_in, y_test_in, cnn_0))

average_test_acc = sum(test_accs) / len(test_accs)
print("ACCURACY - TEST SET: {}".format(average_test_acc))

# QUESTION 6: Reset the model's weights, then remove 75% of the data from the training set. Repeat the training process from question 4 on this diminished dataset. Plot the loss, evaluate this model on its training set and the test set.
start_t = timer()

cnn_0.set_weights(param_save)

n_train         = len(x_train)

x_train_smaller = x_train[:int(0.25 * n_train)]
y_train_smaller = y_train[:int(0.25 * n_train)]
train_dataset_smaller = tf.data.Dataset.from_tensor_slices((x_train_smaller, y_train_smaller))
train_dataset_smaller = train_dataset_smaller.shuffle(10000).batch(16)

average_training_losses = []
training_losses = []

for id_iter, (x_train_in, y_train_in) in zip(range(6000), train_dataset_smaller.repeat()):
    training_losses.append(train_step(x_train_in, y_train_in, cnn_0))
    if id_iter % 200 == 0:
        average_training_losses.append(sum(training_losses) / len(training_losses))
        training_losses = []
        print("ITER  {}: training loss       = {:.4f}".format(id_iter, average_training_losses[-1]))
        print("______________________________________")

end_t = timer()
print("time_elapsed: {}".format(end_t - start_t))

times = list(range(len(average_training_losses)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, average_training_losses)
ax.set_xlabel("iteration / 100")
ax.set_ylabel("classification loss")

train_accs = []
for x_train_in, y_train_in in train_dataset_smaller:
    train_accs.append(eval_step(x_train_in, y_train_in, cnn_0))

average_train_acc = sum(train_accs) / len(train_accs)
print("ACCURACY - TRAINING SET: {}".format(average_train_acc))


test_accs = []
for x_test_in, y_test_in in test_dataset:
    test_accs.append(eval_step(x_test_in, y_test_in, cnn_0))

average_test_acc = sum(test_accs) / len(test_accs)
print("ACCURACY - TEST SET: {}".format(average_test_acc))

# QUESTION 7: We are going to modify the training loop to incorporate validation; every 100th iteration, evaluate the model on the entire validation dataset and report the average accuracy (do this for 0 and 75% of data removed). Plot the validation accuracy over the course of the training process
start_t = timer()

cnn_0.set_weights(param_save)

n_train         = len(x_train)

average_training_losses = []
training_losses = []
average_val_accs = []
val_accs = []

for id_iter, (x_train_in, y_train_in) in zip(range(6000), train_dataset.repeat()):
    training_losses.append(train_step(x_train_in, y_train_in, cnn_0))
    if id_iter % 200 == 0:
        for x_val_in, y_val_in in val_dataset:
            val_accs.append(eval_step(x_val_in, y_val_in, cnn_0))
        average_val_accs.append(sum(val_accs) / len(val_accs))
        average_training_losses.append(sum(training_losses) / len(training_losses))
        training_losses = []
        val_accs = []
        print("ITER  {}: training loss       = {:.4f}".format(id_iter, average_training_losses[-1]))
        print("        : validation accuracy = {:.4f}".format(average_val_accs[-1]))
        print("______________________________________")

end_t = timer()
print("time_elapsed: {}".format(end_t - start_t))

times = list(range(len(average_training_losses)))

fig = plt.figure()

ax = fig.add_subplot(121)
ax.plot(times, average_training_losses)
ax.set_xlabel("iteration * 100")
ax.set_ylabel("classification loss")

ax = fig.add_subplot(122)
ax.plot(times, average_val_accs)
ax.set_xlabel("iteration * 100")
ax.set_ylabel("validation accuracy")

train_accs = []
for x_train_in, y_train_in in train_dataset:
    train_accs.append(eval_step(x_train_in, y_train_in, cnn_0))

average_train_acc = sum(train_accs) / len(train_accs)
print("ACCURACY - TRAINING SET: {}".format(average_train_acc))


test_accs = []
for x_test_in, y_test_in in test_dataset:
    test_accs.append(eval_step(x_test_in, y_test_in, cnn_0))

average_test_acc = sum(test_accs) / len(test_accs)
print("ACCURACY - TEST SET: {}".format(average_test_acc))

start_t = timer()

cnn_0.set_weights(param_save)

n_train         = len(x_train)

x_train_smaller = x_train[:int(0.25 * n_train)]
y_train_smaller = y_train[:int(0.25 * n_train)]
train_dataset_smaller = tf.data.Dataset.from_tensor_slices((x_train_smaller, y_train_smaller))
train_dataset_smaller = train_dataset_smaller.shuffle(10000).batch(16)

average_training_losses = []
training_losses = []
average_val_accs = []
val_accs = []

for id_iter, (x_train_in, y_train_in) in zip(range(10000), train_dataset_smaller.repeat()):
    training_losses.append(train_step(x_train_in, y_train_in, cnn_0))
    if id_iter % 200 == 0:
        for x_val_in, y_val_in in val_dataset:
            val_accs.append(eval_step(x_val_in, y_val_in, cnn_0))
        average_val_accs.append(sum(val_accs) / len(val_accs))
        average_training_losses.append(sum(training_losses) / len(training_losses))
        training_losses = []
        val_accs = []
        print("ITER  {}: training loss       = {:.4f}".format(id_iter, average_training_losses[-1]))
        print("        : validation accuracy = {:.4f}".format(average_val_accs[-1]))
        print("______________________________________")

end_t = timer()
print("time_elapsed: {}".format(end_t - start_t))

times = list(range(len(average_training_losses)))

fig = plt.figure()

ax = fig.add_subplot(121)
ax.plot(times, average_training_losses)
ax.set_xlabel("iteration * 100")
ax.set_ylabel("classification loss")

ax = fig.add_subplot(122)
ax.plot(times, average_val_accs)
ax.set_xlabel("iteration * 100")
ax.set_ylabel("validation accuracy")

train_accs = []
for x_train_in, y_train_in in train_dataset_smaller:
    train_accs.append(eval_step(x_train_in, y_train_in, cnn_0))

average_train_acc = sum(train_accs) / len(train_accs)
print("ACCURACY - TRAINING SET: {}".format(average_train_acc))


test_accs = []
for x_test_in, y_test_in in test_dataset:
    test_accs.append(eval_step(x_test_in, y_test_in, cnn_0))

average_test_acc = sum(test_accs) / len(test_accs)
print("ACCURACY - TEST SET: {}".format(average_test_acc))

# QUESTION 8: Retrain on the small training set, but this time validate every 50 iterations and save the model's weights when it reaches peak accuracy. You may decrease the number of training iterations. Evaluate on the test set to confirm the improvement.
start_t = timer()

cnn_0.set_weights(param_save)

average_training_losses = []
training_losses = []
average_val_accs = []
val_accs = []

best_acc = 0
best_params = cnn_0.get_weights() 

for id_iter, (x_train_in, y_train_in) in zip(range(4000), train_dataset_smaller.repeat()):
    training_losses.append(train_step(x_train_in, y_train_in, cnn_0))
    if id_iter % 50 == 0:
        for x_val_in, y_val_in in val_dataset:
            val_accs.append(eval_step(x_val_in, y_val_in, cnn_0))
        average_val_accs.append(sum(val_accs) / len(val_accs))
        if best_acc < sum(val_accs) / len(val_accs):
            best_params = cnn_0.get_weights()
            best_acc = sum(val_accs) / len(val_accs)
        average_training_losses.append(sum(training_losses) / len(training_losses))
        training_losses = []
        val_accs = []
        print("ITER  {}: training loss       = {:.4f}".format(id_iter, average_training_losses[-1]))
        print("        : validation accuracy = {:.4f}".format(average_val_accs[-1]))
        print("______________________________________")

end_t = timer()
print("time_elapsed: {}".format(end_t - start_t))


cnn_0.set_weights(best_params)

train_accs = []
for x_train_in, y_train_in in train_dataset_smaller:
    train_accs.append(eval_step(x_train_in, y_train_in, cnn_0))

average_train_acc = sum(train_accs) / len(train_accs)
print("ACCURACY - TRAINING SET: {}".format(average_train_acc))


test_accs = []
for x_test_in, y_test_in in test_dataset:
    test_accs.append(eval_step(x_test_in, y_test_in, cnn_0))

average_test_acc = sum(test_accs) / len(test_accs)
print("ACCURACY - TEST SET: {}".format(average_test_acc))

