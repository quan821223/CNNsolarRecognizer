from PIL import Image, ImageDraw
import numpy as np
import math 


def maxminehance(imTensor):
	for i in range(imTensor.shape[0]):
    		imTensor[i,:,:] = (imTensor[i,:,:]-np.min(imTensor[i,:,:])) *( math.floor(255/np.max(imTensor[i,:,:]-np.min(imTensor[i,:,:]))))
	imTensor3=  np.zeros((imTensor.shape[0],imTensor.shape[1],imTensor.shape[2],1), dtype=np.float)
	imTensor3[:,:,:,0]=imTensor[:,:,:]/255
	return imTensor3


