from matplotlib import pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, Layer
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator


class MyDense(Layer):
    def __init__(self, num_outputs):
        super(MyDense, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


class MyConv(Layer):
    def __init__(self, num_outputs):
        # TODO
        return

    def build(self, input_shape):
        # TODO
        return

    def __call__(self, input):
        # TODO
        return


class MyPool(Layer):
    def __init__(self, num_outputs):
        # TODO
        return

    def build(self, input_shape):
        # TODO
        return

    def __call__(self, input):
        # TODO
        return

class MyDataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        #TODO
        return

    def __len__(self):
        #TODO
        return

    def __getitem__(self, item):
        #TODO
        return


def create_model():
    #Replace preimplemented layers with our own when done
    model = Sequential()
    model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(150, 150, 3)))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(MyDense(512))
    model.add(MyDense(512))
    model.add(MyDense(2))
    #model.add(Dense(units=512, activation="relu"))
    #model.add(Dense(units=512, activation="relu"))
    #model.add(Dense(units=2, activation="softmax"))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fit_model():
    model = create_model()
    train_datagen = ImageDataGenerator(rescale=1.0/255, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train = train_datagen.flow_from_directory('filtered/train', class_mode='binary', batch_size=16, target_size=(150, 150))
    validation = val_datagen.flow_from_directory('filtered/validation', class_mode='binary', batch_size=16, target_size=(150, 150))

    model = model.fit(train, validation_data=validation, epochs=30)
    return model


def plot_data(history):
    plt.subplot(2, 1, 1)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='red')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='blue')
    plt.plot(history.history['val_accuracy'], color='red')

    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()

history = fit_model()
plot_data(history)