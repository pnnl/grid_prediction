# Code description:

	- deepDMD_Training.ipynb (Main code that helps train the deepDMD model. This code uses helper_fcns.py to load the data. This code saves the neural network weights, biases and the Koopman operator into a .mat file.)
	- deepDMD_Testing.ipynb  (Use this code to evaluate the learnt model on new test cases. This code uses SurrogateModel.py while applying the trained deepDMD model) 
	- helper_fcns.py                  (This code helps read the datasets for the IEEE 68 bus system. User can define their own helper function for their application.)
	- SurrogateModel.py           (This code loads the .mat file that is obtained from deepDMD_Training.ipynb) 