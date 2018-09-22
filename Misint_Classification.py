from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
import keras.backend as K
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
rows = 28
column = 28
# Specifying Batch Size
batch_size = 256
# Specifying the number of classes
num_classes = 10
# Specifying the number of epochs
num_epochs = 20

if K.image_data_format() == 'channels_first':
    x_train = X_train.reshape(X_train.shape[0], 1, rows, column)
    x_test = X_test.reshape(X_test.shape[0], 1, rows, column)
    input_shape = (1, rows, column)
else:
    x_train = X_train.reshape(X_train.shape[0], rows, column, 1)
    x_test = X_test.reshape(X_test.shape[0], rows, column, 1)
    input_shape = (rows, column, 1)

# more reshaping
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(X_train.shape)
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(128, (3, 3), activation='relu'))
# Polling from the images
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
model.summary()
# Training of the model

model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# i = 0
# for img in X_train[:5]:
#     plt.title(y_train[i])
#     plt.imshow(img, cmap='gray')
#     plt.show()
#     i += 1
#
