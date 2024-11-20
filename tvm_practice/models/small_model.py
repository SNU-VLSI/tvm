import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.regularizers import l2

#define model
def model():
    # Resnet parameters
    input_shape=[8,8,28] # default size for cifar10
    num_classes=10 # default class number for cifar10
    inputs = Input(shape=input_shape)

    x = Conv2D(64,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    # Weight layers
    x = Conv2D(64,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
  
def getTestModel():
  from tvm.contrib.download import download_testdata
  from PIL import Image
  from tensorflow.keras.applications.resnet50 import preprocess_input
  import tvm.relay as relay
  data = np.ones((1, 28, 8, 8)).astype("float32")
  shape_dict = {"input_1": data.shape}
  mod, params = relay.frontend.from_keras(model(), shape_dict)
  return mod, params, shape_dict