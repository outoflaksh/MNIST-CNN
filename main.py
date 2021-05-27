from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
from os import system, path

def prep_img(img_url):
	# making the apt dimensions while also grayscaling the img
	test_img = load_img(img_url, color_mode="grayscale", target_size=(28, 28))

	# converting so prepped img to array
	test_img = img_to_array(test_img)

	#flattening the array to feed to model
	test_img = test_img.reshape((-1, 28,28, 1))

	return test_img


def preprocess(x):
	#normalising the value between 0 to 1
	x = x.astype("float32")/255

	#making the apt dimensions for the multi-dimensional array
	x = np.expand_dims(x, -1)

	return x

def predict_class(x):
	return np.argmax(model.predict([x]))

# loading pretained model
model = keras.models.load_model("./model1.h5")

system('cls')
# take input for file name
img_url = "./"+input("\r\n\nEnter image name: ").strip()+".png"

if path.isfile(img_url):

	# prepare image to array
	test_img = prep_img(img_url)

	# preprocessing for the model
	x = preprocess(test_img)

	# get the predicted digit from the model
	prediction = predict_class(x)

	print("\r\n\nPREDICTION MADE BY THE MODEL: ")
	print(prediction)

else:
	print("\n\nFILE DOESN'T EXIST!! Please enter a VALID file name")

