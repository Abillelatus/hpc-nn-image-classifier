#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 23:31:25 2023

@author: Ryan Herrin

This is the python file that will create and deliver a classifier model made
from the pre-processed images.
"""

import os
import cv2 as cv
import numpy as np
from logger import Logger

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split


class ModelGen:
    """Class to generate a CNN classification model"""

    def __init__(self, log_file_loc, img_dir, model_out_dir):
        # Locations
        self.img_dir = img_dir
        self.model_out_dir = model_out_dir

        # Image Data
        self.img_data = {'image_data': [], 'file_name': [], 'class': [],
                         'x_train': None, 'x_test': None, 'y_train': None,
                         'y_test': None}

        # Key words in file name
        self.keyword = 'football'
        self.avoid_keyword = 'american'  # Avoid this in the keyword string

        # Initialize logger
        self.log_obj = Logger(self.__class__.__name__)
        self.log_obj.set_log_file(log_file_loc)
        self.log_it = self.log_obj.log_it

        # Model attributes
        self.model = None
        self.img_rows = 300
        self.img_cols = 300
        self.n_classes = 2
        self.m_batch_size = 32
        self.m_n_classes = 2
        self.m_filters = 16
        self.m_pool_size = 2
        self.m_kernel_size = 3
        self.n_epochs = 7

        # Format image shape that's fed into the model
        if K.image_data_format() == 'channels_first':
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            self.input_shape = (self.img_rows, self.img_cols, 1)

        # Model layers
        self.feature_layers = [
            Conv2D(self.m_filters, self.m_kernel_size, padding='valid',
                   input_shape=self.input_shape),
            Activation('relu'),
            Conv2D(self.m_filters, self.m_kernel_size),
            Activation('relu'),
            MaxPooling2D(pool_size=self.m_pool_size),
            Dropout(0.25),
            Flatten()
        ]

        self.classification_layers = [
            Dense(32),
            Activation('relu'),
            Dropout(0.5),
            Dense(self.n_classes),
            Activation('sigmoid')
        ]

    def _load_data(self):
        '''Load images into memory and format it so that it can be inserted
        into the machine learning model'''
        self.log_it("Loading in image Data...")

        # Create a list of files to read in
        filepath_lst = []
        for file in os.listdir(self.img_dir):
            filepath_lst.append(os.path.join(self.img_dir, file))

        # Read in the image and convert to grayscale
        for img in filepath_lst:
            self.img_data['image_data'].append(cv.imread(
                img, cv.IMREAD_GRAYSCALE))

            # Set the file name ttribute
            self.img_data['file_name'].append(img)

            # Find out if the key value to determine if the image is positive
            # or negative exists in the file name. If so set the attribute to
            # 0 for positive and 1 for negative.
            if self.keyword in img and self.avoid_keyword not in img:
                self.img_data['class'].append(0)
            else:
                self.img_data['class'].append(1)

        # Convert list of image data into numpy array
        self.img_data['image_data'] = np.asarray(self.img_data['image_data'])

        # Format the data into a format that tensroflow accepts
        self.img_data['image_data'] = self.img_data['image_data'].reshape(
            (self.img_data['image_data'].shape[0],) + self.input_shape)

        # Convert to float32
        self.img_data['image_data'] = self.img_data['image_data'].astype(
            'float32')

        # Divide the values by 255 to reduce the range between 0 and 1. We
        # divide by 255 because that's the range of the color scale.
        self.img_data['image_data'] /= 255

        # Convert the y values
        self.img_data['class'] = keras.utils.to_categorical(
            self.img_data['class'], self.n_classes)

    def _split_data(self):
        '''Split the data into a train test split'''
        (self.img_data['x_train'], self.img_data['x_test'],
         self.img_data['y_train'], self.img_data['y_test']) = train_test_split(
             self.img_data['image_data'], self.img_data['class'], shuffle=True,
             test_size=.2)

    def _generate_model(self):
        '''Function that trains and generates the model'''
        self.log_it('Creating Model...')
        # Initiate the model
        self.model = Sequential(
            self.feature_layers + self.classification_layers)

        # Compile the model
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy'])

        # Fit the model
        self.model.fit(
            self.img_data['x_train'], self.img_data['y_train'],
            batch_size=self.m_batch_size, epochs=self.n_epochs, verbose=1,
            validation_data=(self.img_data['x_test'], self.img_data['y_test']))

        self.log_it('Done generating model...')

        # Get the final score
        score = self.model.evaluate(
            self.img_data['x_test'], self.img_data['y_test'], verbose=0)

        self.log_it("Test Score: {}".format(score[0]))
        self.log_it("Test Accuracy: {}".format(score[1]))

    def export_model(self, dest, timestamp):
        '''Save and export the tensorflow model'''
        self.log_it('Saving and exporting model to: {}'.format(dest))

        self.model.save(dest + '/model-{}.keras'.format(timestamp))

        self.log_it(
            'Finished exporting: model-{}.keras'.format(timestamp))

    def run(self):
        '''Function to coordinate model generation'''
        self.log_it("Starting model generation...")
        self._load_data()
        self._split_data()
        self._generate_model()
