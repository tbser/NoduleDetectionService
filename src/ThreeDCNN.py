from keras.optimizers import SGD
from keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, Flatten, Dropout, BatchNormalization
from keras.models import Model
from keras.metrics import binary_accuracy, binary_crossentropy, mean_absolute_error
from keras.constraints import maxnorm

from dnn_model import DNN_model
import helpers
import random

random.seed(2)

default_logger = helpers.getlogger("3DCNN")

import tensorflow as tf
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


class ThreeDCNN(DNN_model):
    def __init__(self, logger=default_logger):
        DNN_model.__init__(self)
        self.logger = logger

    def generate_model(self, input_shape, dropout=False, batchnormalization=False, load_weight_path=None):
        if self.model is not None:
            return
        inputs = Input(shape=input_shape, name="input_1")
        x = inputs
        constraint=None
        if dropout:
            x = Dropout(rate=0.1)(x)
            constraint=maxnorm(4)
        x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding="same")(x)
        x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', kernel_initializer='lecun_normal', kernel_constraint=constraint,
                   name='conv1')(x)
        if batchnormalization:
            x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)
        if dropout:
            x = Dropout(rate=0.25)(x)

        # 2nd layer group
        x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', kernel_initializer='lecun_normal', kernel_constraint=constraint,
                   name='conv2')(x)
        if batchnormalization:
            x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
        if dropout:
            x = Dropout(rate=0.25)(x)

        # 3rd layer group
        x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', kernel_initializer='lecun_normal', kernel_constraint=constraint,
                   name='conv3a')(x)
        if batchnormalization:
            x = BatchNormalization()(x)
        x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', kernel_initializer='lecun_normal', kernel_constraint=constraint,
                   name='conv3b')(x)
        if batchnormalization:
            x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)
        if dropout:
            x = Dropout(rate=0.5)(x)

        # 4th layer group
        x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', kernel_initializer='lecun_normal', kernel_constraint=constraint,
                   name='conv4a')(x)
        if batchnormalization:
            x = BatchNormalization()(x)
        x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', kernel_initializer='lecun_normal', kernel_constraint=constraint,
                   name='conv4b')(x)
        if batchnormalization:
            x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)
        if dropout:
            x = Dropout(rate=0.5)(x)

        last64 = Conv3D(64, (2, 2, 2), kernel_initializer='lecun_normal', activation="relu", name="last_64")(x)
        out_class = Conv3D(1, (1, 1, 1), kernel_initializer='lecun_normal', activation="sigmoid", kernel_constraint=constraint, name="out_class_last")(last64)
        out_class = Flatten(name="out_class")(out_class)

        out_malignancy = Conv3D(1, (1, 1, 1), kernel_initializer='lecun_normal', activation=None, kernel_constraint=constraint, name="out_malignancy_last")(
            last64)
        out_malignancy = Flatten(name="out_malignancy")(out_malignancy)

        model = Model(inputs=inputs, outputs=[out_class, out_malignancy])
        if load_weight_path is not None:
            model.load_weights(load_weight_path, by_name=False)

        MOMENTUM = 0.9
        NESTEROV = True
        if dropout:
            MOMENTUM = 0.95
            NESTEROV = False

        model.compile(optimizer=SGD(lr=self.LEARN_RATE, momentum=MOMENTUM, nesterov=NESTEROV),
                      loss={"out_class": "binary_crossentropy", "out_malignancy": mean_absolute_error},
                      metrics={"out_class": [binary_accuracy, binary_crossentropy], "out_malignancy": mean_absolute_error})

        self.model_summary(model)
        self.model = model


if __name__ == "__main__":
    CUBE_SIZE = 32
    cnnmodel = ThreeDCNN()
    cnnmodel.generate_model((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), batchnormalization=True)
