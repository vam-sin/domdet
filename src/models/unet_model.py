# outer product of the 1D feature set
# 2d contact map directly sent into the model
# mean pool for each row
# softmax over the output labels
# try dilated convolutions?
 
# libraries
import numpy as np
from tensorflow import keras
import keras.backend as K
from keras.regularizers import l1_l2
import tensorflow as tf
from keras.layers import Activation
from keras.layers.core import Lambda
from keras.models import Model, load_model
# from keras.utils import np_utils, plot_model
from keras.layers.merge import concatenate
from keras.layers import Input, Dropout, BatchNormalization, Reshape, Flatten, Dense, LSTM
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers import Conv2D, Conv1D, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
# from keras_self_attention import SeqSelfAttention

# model definition function
# param
dropout = 0.1
smooth = 1.
act = ELU
init = "glorot_uniform"
reg_strength = 1e-12
reg = l1_l2(reg_strength)
num_filters = 2

def self_outer(x):
    outer_x = x[ :, :, None, :] * x[ :, None, :, :]
    return outer_x

def column_mean_pool(x):
	x_mean = K.mean(x, axis=1)
	return x_mean

def add_2D_conv(inp, filters, kernel_size, data_format="channels_last", padding="same", depthwise_initializer=init, pointwise_initializer=init, depthwise_regularizer=reg, 
        pointwise_regularizer=reg):

	x = inp
	for j in range(1):
		x = Conv2D(filters, kernel_size, data_format=data_format, padding=padding, kernel_initializer=depthwise_initializer, kernel_regularizer=depthwise_regularizer)(x)
		x = act()(x)
		x = BatchNormalization()(x)

	return x

def unet():
	inp_2d = [Input(shape=(None, None, 1), name="cmap", dtype=K.floatx())] 
              
	seq = [Input(shape = (None, 32), dtype=K.floatx(), name = "ft_vec_1d")]
	model_1D_outer = Lambda(self_outer)(seq[0])
	model_1D_outer = BatchNormalization()(model_1D_outer)

	unet = keras.layers.concatenate(inp_2d +  [model_1D_outer])
	unet = add_2D_conv(unet, num_filters, 1)
	unet = add_2D_conv(unet, num_filters, 3)
	unet = add_2D_conv(unet, num_filters, 3)

	# unet = MaxPooling2D(pool_size=(2, 2), data_format = "channels_last", padding='same')(unet)
	# unet = add_2D_conv(unet, num_filters*2, 3)
	# unet = add_2D_conv(unet, num_filters*2, 3)
	# unet = add_2D_conv(unet, num_filters*2, 3)

	# unet = UpSampling2D((2,2), data_format = "channels_last")(unet)
	# unet = add_2D_conv(unet, num_filters, 2)

	unet = Lambda(column_mean_pool)(unet)

	unet = Dense(32, activation ="relu", kernel_initializer = init, kernel_regularizer = reg)(unet)
	output = Dense(1, activation ="sigmoid", kernel_initializer = init, kernel_regularizer = reg, name="out_dom_labels")(unet)

	# output = Reshape((1, -1))(unet)

	model = Model(inputs = inp_2d + seq, outputs = output)
	print(model.summary())

	return model 

if __name__ == '__main__':
	model = unet()
	dict_x = {}
	dict_x['ft_vec_1d'] = np.zeros((1, 10, 1038))
	dict_x['cmap'] = np.zeros((1, 10, 10))
	dict_x['y'] = np.zeros((1, 10, 1))
	fin = model.predict(dict_x)
	print(fin.shape, dict_x['y'].shape)
	# print(fin)


