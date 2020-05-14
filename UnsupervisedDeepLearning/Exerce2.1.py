from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
# Model
input_size = 28*28
output_size = input_size
hidden_units = 128
optimiser = Adadelta(learning_rate=10.0)
loss = 'mean_squared_error'
batch_size = 256
epochs = 20

input_layer = Input(shape=(input_size,))
#########
# Define the model here
#########
layer = Dense(hidden_units[i], activation='relu', input_shape=(input_size,))(input_layer)
decoded = Dense(input_size,activation='sigmoid',input_shape=(hidden_units[i],))(layer)

#decoded =

model = Model(inputs=input_layer, outputs=decoded)
model.compile(optimizer=optimiser, loss=loss)

# Load the MNIST dataset
mnist_dataset = mnist.load_data()
(trainset, testset) = (mnist_dataset[0], mnist_dataset[1])
(X_train, y_train) = trainset
(X_test, y_test) = testset

# Preprocess data (convert to float and scale to between 0 and 1)
X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255

# Display the data
n_images = 4
for i in range(n_images):
    plt.imshow(X_test[i], cmap='gray')
    plt.title('MNIST Example Image ' + str(i + 1))
    plt.show()

# Flatten data (turn images into vectors)
X_train_fl = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
X_test_fl = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))


# Train the model
#########
# Use the model.fit function to train the model on the flattened training data using the batch_size, epochs defined above,
# 20% validation data, and you should shuffle the data
#########
history = model.fit(X_train_fl, X_train_fl, batch_size, epochs,validation_split=0.2,validation_data=(X_test_fl,X_test_fl),shuffle=True)

# Visualise the training process and result

# Plot losses
losses = history.history
plt.plot(losses['loss'], label='train loss')
plt.plot(losses['val_loss'], label='val loss')
plt.legend()
plt.show()

# Plot AE reconstructions
#########
# Use the model.predict function to get the reconstructions of the flattened test data
#########
decoded_images = model.predict(X_test_fl)
n_images = 10
for i in range(n_images):
    plt.imshow(X_test[i], cmap='gray')
    plt.title('Test Image ' + str(i+1))
    plt.show()
    plt.imshow(decoded_images[i].reshape(28, 28), cmap='gray')
    plt.title('Reconstruction ' + str(i+1))
    plt.show()
