from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Conv2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import models, layers
from keras.models import Sequential
import keras

from keras.layers.normalization import BatchNormalization
# fix random seed for repr18oducibility
from keras.layers.advanced_activations import LeakyReLU, PReLU
def build_model(dim, num_classes,Opti,chennels):
# Create the model
		model = Sequential()
		model.add(Convolution2D(16, 5,5  , input_shape=( 50, 50 ,1), border_mode='same', activation='relu'))
		#model.add(BatchNormalization(axis =-1,momentum=0.8))
		model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.01, center=True, scale=True, beta_initializer='zeros'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Convolution2D(16,  5,5   , border_mode='same', activation='relu'))
		model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.01, center=True, scale=True, beta_initializer='zeros'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Convolution2D(32,  5,5   , border_mode='same', activation='relu'))
		model.add(Convolution2D(32,  5,5   , border_mode='same', activation='relu'))
		model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.01, center=True, scale=True, beta_initializer='zeros'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		model.add(Dense(1024, activation='relu'))
		model.add(Dropout(0.3))
		model.add(Dense(1024, activation='relu'))
		model.add(Dropout(0.3))
		model.add(Dense(num_classes, activation='softmax'))

		return model
