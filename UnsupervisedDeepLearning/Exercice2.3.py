from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

# Model
input_size = 28*28
output_size = input_size
optimiser = Adadelta(learning_rate=1.0)
loss = 'mean_squared_error'
batch_size = 256
epochs = 20

input_layer = Input(shape=(28, 28, 1))

#########
# Define the model here
#########
# Encoded
layer = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
layer = MaxPooling2D((2, 2), padding='same')(layer)
layer = Conv2D(8, (3, 3), activation='relu', padding='same')(layer)
layer = MaxPooling2D((2, 2), padding='same')(layer)
layer = Conv2D(8, (3, 3), activation='relu', padding='same')(layer)
encoded = MaxPooling2D((2, 2), padding='same')(layer)

# Decoded
layer = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
layer = UpSampling2D((2, 2))(layer)
layer = Conv2D(8, (3, 3), activation='relu', padding='same')(layer)
layer = UpSampling2D((2, 2))(layer)
layer = Conv2D(16, (3, 3), activation='relu')(layer)
layer = UpSampling2D((2, 2))(layer)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(layer)

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


# Adding new axis
X_train = X_train[...,np.newaxis] 
X_test = X_test[...,np.newaxis]


# Train the model
#########
# Use the model.fit function to train the model on the training data using the batch_size, epochs defined above,
# 20% validation data, and you should shuffle the data
#########
history = model.fit(X_train, X_train, batch_size, epochs,validation_split=0.2,validation_data=(X_test,X_test),shuffle=True)

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
decoded_images = model.predict(X_test)
n_images = 10
for i in range(n_images):
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')
    plt.title('Test Image ' + str(i+1))
    plt.show()
    plt.imshow(decoded_images[i].reshape(28, 28), cmap='gray')
    plt.title('Reconstruction ' + str(i+1))
    plt.show()
