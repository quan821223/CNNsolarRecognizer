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
from keras.models import Sequential, model_from_json
from keras.layers import *
import os 
import time

banner =  '''\nWhat would you like to do ?
    1- Train models
    2- plot training model loss and accurcy 
    3- save the model
    4- loading the model		
    5- Visualize feature maps of different layers of trained model
    '''
SaveDirectory = os.getcwd() 
SaveAs = os.path.join(SaveDirectory)
print SaveAs
def main():
	dim=50
	batch_size = 70
	epochs = 10
	num_classes=2
	
	filesnames_train = "train4"
	filesnames_test = "test1"
	modelnames_save = "test1"

	Opti=Adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-01)
	imTensor1 ,  y_train1= DATA.find_data( 1 , SaveDirectory+"/sample/"  , filesnames_train  )
	imTensor2 ,  y_train2= DATA.find_data( 1 , SaveDirectory+"/sample/"  , filesnames_test  )
	#Ehance image that using Max-min normalize
	imTensor3 = prucedure.maxminehance(imTensor1)
	imTensor4 = prucedure.maxminehance(imTensor2)
	#image chennels size 
	chennels = imTensor3.shape[3]


	#user first table
	x=0
	def rawInputTest():
		x= raw_input(">>> Input:")
	print banner 
	#ans = raw_input( "choose one!")

	while(True):
		ans = int(raw_input( ">>> Input:"))
		#rawInputTest()
		if ans == 0:
			savedirectory = os.getcwd()
			saveas = os
			print SaveAs
		elif ans == 1:
			model = models.build_model(dim, num_classes,Opti,chennels)
			model.compile(loss='categorical_crossentropy', optimizer=Opti , metrics=['accuracy'])
			model.summary()
			history = model.fit(imTensor3, y_train1,
			      batch_size=batch_size,
			      epochs=epochs,
			      validation_data=(imTensor4, y_train2))
			score = model.evaluate(imTensor4,y_train2)
			print("the end, you trained a CNN model.")


		elif ans == 2:

			print()
			print('Test loss:', score[0])					
			print('Test accuracy:', score[1])
	
			prediction=model.predict_classes(imTensor4)
			# Loss Curves
			plt.figure(figsize=[8,6])
			plt.plot(history.history['loss'],'r',linewidth=3.0)
			plt.plot(history.history['val_loss'],'b',linewidth=3.0)
			plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
			plt.xlabel('Epochs ',fontsize=16)
			plt.ylabel('Loss',fontsize=16)
			plt.title('Loss Curves',fontsize=16)

			# Accuracy Curves
			plt.figure(figsize=[8,6])
			plt.plot(history.history['acc'],'r',linewidth=3.0)
			plt.plot(history.history['val_acc'],'b',linewidth=3.0)
			plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
			plt.xlabel('Epochs ',fontsize=16)
			plt.ylabel('Accuracy',fontsize=16)
			plt.title('Accuracy Curves',fontsize=16)
			pyplot.show()##15-77
			plt.show()

		elif ans == 3:

			# serialize model to JSON
			model_json = model.to_json()
			with open(SaveDirectory+"/checkpoints/"+modelnames_save+ ".json", "w") as json_file:
			    json_file.write(model_json)
			model.save_weights(SaveDirectory+"/checkpoints/"+modelnames_save+".h5")
			print("Saved model to disk")
			
		elif ans == 4:

			# loading model to JSON
			config = open(SaveDirectory+"/checkpoints/"+modelnames_save+".json", "r").read()
			model = model_from_json(config)
			model.load_weights(SaveDirectory+"/checkpoints/"+modelnames_save+".h5")	
			print("loading model from files")
		else:
			print "what do you want!!!"
			print banner
			rawInputTest()
			
		print banner 








if __name__ == '__main__':
    main()
