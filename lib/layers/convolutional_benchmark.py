import time

import numpy as np
from convolutional import Convolutional

rng = np.random.default_rng(seed=42)

DEPTH = 3
KERNELS = 32
FILTER_SIZE = 3
STRIDE = 1
PADDING = 0

TEST_ARRAY = rng.uniform(0, 100, size=(10000, 25, 25, DEPTH))
TEST_KERNELS = rng.uniform(-1, 1, size=(KERNELS, FILTER_SIZE, FILTER_SIZE, DEPTH))
INPUT_SHAPE = TEST_ARRAY.shape[1:]

TEST_BIASES = np.zeros(KERNELS)

print(TEST_ARRAY.shape, TEST_KERNELS.shape, TEST_BIASES.shape)

myConvArray = TEST_ARRAY

myConvLayer = Convolutional(INPUT_SHAPE, (KERNELS, FILTER_SIZE, STRIDE, PADDING))
myConvLayer.activation_method = "None"
myConvLayer.kernels = TEST_KERNELS

myConvLayer.biases = TEST_BIASES

myConvLayer.biases = myConvLayer.biases .astype(int)

time_vectorized = time.time()

myConvLayer.process(myConvArray, vectorized=True)

print(f"TIME VECTORIZED: {time.time() - time_vectorized}")

time_unvectorized = time.time()

myConvLayer.process(myConvArray, vectorized=False)


print(f"TIME UNVECTORIZED: {time.time() - time_unvectorized}")

print("Output Shapes: ", myConvLayer.outputs_nd.shape, myConvLayer.outputs.shape)
