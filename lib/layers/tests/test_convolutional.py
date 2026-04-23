import numpy as np

from convolutional import Convolutional

def test1():
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
    TEST_OUTPUT = np.array([[[-1, 1], [0, -5], [1, 1]],
                            [[0, -1], [-2, -3], [0, -4]],
                            [[0, -2], [2, -4], [-2, 1]]])
    TEST_ARRAYS = np.array([TEST_ARRAY, TEST_ARRAY, TEST_ARRAY])

    conv_layer_1 = Convolutional(TEST_ARRAY.shape, (2, 3, 2, 1))
    conv_layer_1.activation_method = "None"
    conv_layer_1.kernels = TEST_KERNELS
    conv_layer_1.biases = TEST_BIASES.astype(int)

    conv_layer_1.process(TEST_ARRAYS)

    assert((conv_layer_1.outputs == TEST_OUTPUT).all()), "TEST 1 FAILED: output is not equal to expected output"

def test_convolutional():
    test1()
