import tvm
from tvm import te
import tvm.relay as relay

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

def conv2d():
  input_shape = [32, 32, 3]
  inputs = Input(shape=input_shape)
  outputs = Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(inputs)
  model = Model(inputs=inputs, outputs=outputs)
  return model

mod, params = relay.frontend.from_keras(conv2d(), shape={'input_1': (1, 3, 32, 32)})
# graph_module = relay.build(mod, params=params, target="llvm -mtriple=aarch64-linux-gnu")
graph_module = relay.build(mod, params=params, target="c")
# print(graph_module.get_graph_json())

lib = graph_module.get_lib()
print(lib.get_source())