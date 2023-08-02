# hpc-nn-image-classifier
Neural Network for image classification designed to be run on HPC system.  
Utilizes Tensorflow Keras.

The script performs the following steps:
1. Generates a log file that will record function time and model performance
2. Extracts images from a zip file and puts them into a new directory
3. Preprocesses the images by changing them to grayscale and sizing them to all be the same dimensions. I wrote this function to utilize multiproccessing to increase performance.
4. Load the picture data into memory.
5. Generate a model using the tensorflow library
6. Save and export that model to the /models directory
7. Clean the build so it can be re-run immediately. 
