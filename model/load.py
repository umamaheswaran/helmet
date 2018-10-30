import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf
from keras.utils.data_utils import get_file

            
def init(): 
	json_file = open('model/classifier.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	#load woeights into new model
	
	loaded_model.load_weights("classifier_weights.h5")
	print(weights_path)
	print("Loaded Model")

	#compile and evaluate loaded model
	loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	graph = tf.get_default_graph()

	return loaded_model,graph

