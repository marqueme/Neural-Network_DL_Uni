# Neural-Network_Uni
These set of files were used in the pre-processing of a text document containing speech dialogue acts. It was used in an overall bigger project where a system of neural networks were built. The input of the project was both text and speech. The sole purpose of the these files are related to the pre-processing of text data.  

Train.txt serves as the input for the text data. The algorithm, Word2Vec, was used to transform the individual utterances into their vectorial form. The model allows for the semantic relatedness of the sentences to be learned and represented in the form that is necessary for the neural network’s input layer. 

The output of the word2vec.py script is used as the input for the TextCNN.py script. This script predicts the correct dialogue act for the given utterances (vectorial form) and is then compared to the gold standard text file. A percentage indicating the accuracy of the predictions are given as well as a graph illustrating the accuracy. 



SETUP
=====
Python 2.7 is needed to run the code. 
Several external libraries are needed for the textCNN.py script including numpy, time, and matplotlib.pyplot. 
The toolkits theano, theano.tensor, and lasagne are also needed. 
