# Import the neccesary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class StridedNet:
    # width: Image width in pixels
    # height: Image height in pixels
    # classes: The number of classes that the model need to predict
    # reg: Regularization method
    # init: The kernel initializer
    @staticmethod
    def build(width, height, depth, classes, reg, init="he_normal"):
        # initialize the model along with the input shape to be
        # "channels last" and the channel dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channel dimensions
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # our first CONV layer will learn a total of 16 filters each
        # of which are 7x7 – we'll then apply 2x2 strides to reduce the
        # spatial dimesions of the volume
        model.add(Conv2D(16, (7,7), strides=(2,2), padding="valid",
            kernel_initializer=init, kernel_regularizer=reg,
            input_shape=inputShape))

        # here we stack two CONV layers on top of each other were
        # each layer will learn a total of 32 (3x3) filters
        model.add(Conv2D(32, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # stack two more CONV layers, keeping the size of each filter
        # as 3x3 but increasing to 64 total learned filters
        model.add(Conv2D(64, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3,3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # increase the number of filters again, this time to 128
        model.add(Conv2D(128, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # fully-connected layer
        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
