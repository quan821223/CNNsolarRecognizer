import numpy as np
import pandas as pd
import prucedure
import os
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.misc import toimage
from matplotlib import pyplot
from keras.utils import np_utils
def find_data(extra_name, data_path,filename):
	# Set working directory
	os.chdir(data_path) #'/home/x/Desktop/num/8000s'
	wheres=filename#"name"
	lst = sorted(os.listdir(wheres))
	os.sep
	if extra_name == "":
		extra_name=0

	imageName1=lst[extra_name]
	tmp1 = ndimage.imread(wheres + os.sep + imageName1)
	ylst=[]
	imList = []
	for imageName in sorted(os.listdir(wheres)):
	    valueString = imageName[1]
	    ylst.append(int(valueString))
	    imList.append(ndimage.imread(wheres + os.sep + imageName))#.transpose((0,1,2)))
	imTensor = np.asarray(imList).astype("float")
	y_train =np_utils.to_categorical( np.asarray(ylst))

	return  imTensor , y_train
