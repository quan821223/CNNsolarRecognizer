import pandas as pd
import matplotlib.pyplot as plt
import prucedure
import models
import DATA
import prucedure
from keras.optimizers import SGD   , RMSprop ,Adam 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from matplotlib import pyplot
import os
import time

banner =  '''\nWhat would you like to do ?
    1- save models
    2- Train the model 
    3- Visualize feature maps of different layers of trained model
    '''



def main():
	x=0
	def rawInputTest():
		x= raw_input(">>> Input:")
	print banner 
	#ans = raw_input( "choose one!")

	while(True):
		ans = int(raw_input( ">>> Input:"))
		#rawInputTest()
		if ans == 1:
			print("you want get 1")
			SaveDirectory = os.getcwd() 
			SaveAs = os.path.join(SaveDirectory)
			print SaveAs
		elif ans == 2:
		    	print("you want get 2")
		elif ans == 3:
			print("Press any key to continuesssssssss")
			continue
		
		else:
			print "what do you want!!!"
		#rawInputTest()




if __name__ == '__main__':
    main()
