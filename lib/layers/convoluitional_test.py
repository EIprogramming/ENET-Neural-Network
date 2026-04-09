import numpy as np
from convolutional import Convolutional


TEST_ARRAY = np.array([[[1, 0, 0], [2, 1, 0], [1, 1, 1], [2, 0, 2], [2, 2, 2]],
                        [[1, 2, 1], [0, 2, 2], [0, 1, 0], [0, 0, 1], [2, 0, 0]],
                        [[0,0,1], [1,0,1], [1,2,1], [0,1,1], [1,0,1]],
                        [[0,1,0], [2,0,1], [2,2,0], [2,0,1], [0,2,0]],
                        [[0,1,2], [2,0,2], [1,0,2], [1,1,0], [2,1,0]]])

TEST_KERNELS = np.array([[[[0, 1, 0], [1, -1, 0], [-1, 1, 0]],
                                 [[1, 0, -1], [-1, 0, 1], [1, 1, -1]],
                                 [[0, -1, 0], [0, -1, 0], [-1, -1, 0]]],

                                 [[[-1, 0, 1], [0, 0, 0], [-1, 1, 1]],
                                 [[-1, 0, 0], [1, 1, -1], [1, 0, -1]],
                                 [[-1, 0, -1], [1, -1, -1], [-1, 1, -1]]]])

TEST_BIASES = np.array([1, 0])

myConvArray = TEST_ARRAY

myConvArray = np.array([myConvArray, myConvArray, myConvArray])

myConvLayer = Convolutional(myConvArray.shape[1:], (2, 3, 2), 1)
myConvLayer.activation_method = "None"
myConvLayer.kernels = TEST_KERNELS

myConvLayer.biases = TEST_BIASES

# expected output:
# -1 0 1
# 0 -2 0
# 0 2 -2
#
# 1 -5 1
# -1 -3 4
# -2 -4 1

myConvLayer.biases = myConvLayer.biases .astype(int)

myConvLayer.process(myConvArray)



print(myConvLayer.outputs_nd.shape)
print(myConvLayer.outputs_nd[0, :, :, 0])
print(myConvLayer.outputs_nd[0, :, :, 1])
print("-"*16)
print(myConvLayer.outputs_nd[1, :, :, 0])
print(myConvLayer.outputs_nd[1, :, :, 1])

print(myConvLayer.outputs_nd.shape, myConvLayer.outputs.shape)

output_test = myConvLayer.convolve_specific(myConvArray, TEST_KERNELS, TEST_BIASES, S=2)
print("="*16)
print(output_test.shape)
print(output_test[0, :, :, 0])
print(output_test[0, :, :, 1])
print("-"*16)
print(output_test[1, :, :, 0])
print(output_test[1, :, :, 1])

#Inputs Shape: (3, 7, 7, 3)
#Filters Shape: (2, 3, 3, 3)
