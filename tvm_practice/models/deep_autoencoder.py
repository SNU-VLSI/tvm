"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# from import
import tensorflow.keras as keras
import tensorflow.keras.models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
import yaml

def yaml_load():
    with open("/root/project/tvm/tvm_practice/models/baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################
# keras model
########################################################################
def get_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder
    (128*128*128*128*8*128*128*128*128)
    """
    inputLayer = Input(shape=(inputDim,))

    h = Dense(128)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(inputDim)(h)

    return Model(inputs=inputLayer, outputs=h)
#########################################################################


def load_model(file_path):
    return keras.models.load_model(file_path)

def getTestModel():
  import tvm.relay as relay
  import numpy as np
  param = yaml_load()
  model = get_model(param["feature"]["n_mels"] * param["feature"]["frames"])
  data = np.ones((1, 640)).astype(np.float32)
  shape_dict = {"input_1": data.shape}
  mod, params = relay.frontend.from_keras(model, shape_dict)
  return mod, params, shape_dict