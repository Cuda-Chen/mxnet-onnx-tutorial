import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
import numpy as np

# Import the ONNX model into MXNet's symbolic interface
model = "fish-resnet50.onnx"
sym, arg, aux = onnx_mxnet.import_model(model)
print("Loaded %s", model)
print(sym.get_internals())
