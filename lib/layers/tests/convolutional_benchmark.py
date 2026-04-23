import time

import numpy as np

# for imports
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
# end for imports

from convolutional import Convolutional

rng = np.random.default_rng(seed=42)

DEPTH = 3
KERNELS = 2
FILTER_SIZE = 3
STRIDE = 2
PADDING = 1

TEST_ARRAY = rng.uniform(0, 100, size=(10, 5, 5, DEPTH))
TEST_KERNELS = rng.uniform(-1, 1, size=(KERNELS, FILTER_SIZE, FILTER_SIZE, DEPTH))
INPUT_SHAPE = TEST_ARRAY.shape[1:]

TEST_BIASES = np.array([i % 2 for i in range(KERNELS)])

print(TEST_ARRAY.shape, TEST_KERNELS.shape, TEST_BIASES.shape)

myConvArray = TEST_ARRAY

myConvLayer = Convolutional(INPUT_SHAPE, (KERNELS, FILTER_SIZE, STRIDE, PADDING))
myConvLayer.activation_method = "None"
myConvLayer.kernels = TEST_KERNELS

myConvLayer.biases = TEST_BIASES

myConvLayer.biases = myConvLayer.biases .astype(int)

time_vectorized = time.time()

myConvLayer.process(myConvArray)

print(f"TIME VECTORIZED: {time.time() - time_vectorized}")

print("Output Shapes: ", myConvLayer.outputs_nd[0, 0, 0, 0], myConvLayer.outputs_nd.shape)
