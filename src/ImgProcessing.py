#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 20:23:12 2023

@author: Ryan Herrin
"""

import os
import cv2 as cv
from logger import Logger
from multiprocessing import Pool


class ImgProcessing:
    def __init__(self, log_file=None):
        """Class to transform the images to prepare them to be fed into a
        machine learning model (Convolutional Neural Network). The images
        will be read in, transformed, and the file renamed so that the
        image can be identified as positive or negative target. Then copied
        into a processed folder so that the original image is left unmodified.
        """
        # Image transormation settings
        self.img_size = (300, 300)
        self.file_lst = []
        self.num_cores = None
        self.img_dir = None
        self.img_count = int()

        # Initiate logger
        if log_file is not None:
            self.log_obj = Logger(self.__class__.__name__)
            self.log_obj.set_log_file(log_file)
            self.log_it = self.log_obj.log_it

    def _build_file_lst(self):
        '''Create a list of img paths. If multiple course are used then split
        the files evenly between multiple lists the cores.'''
        # Git the initial file list
        img_names = os.listdir(self.img_dir)
        _tmp_list = []

        # Get the count of how many files are present
        self.img_count = len(img_names)

        for img in img_names:
            _tmp_list.append(os.path.join(self.img_dir, img))

        # Append as many lists as we do cores available
        self.log_it("Building img lists...")

        for c in range(self.num_cores):
            self.file_lst.append([])

        lst_indx = 0
        for path in _tmp_list:
            self.file_lst[lst_indx].append(path)
            lst_indx += 1
            # Verify the index is still in range of the num of cores
            if lst_indx >= self.num_cores:
                lst_indx = 0

    def _process(self, file_list):
        '''Function to process the images'''
        for file in file_list:
            try:
                # Convert to gray scale
                img = cv.imread(file, cv.IMREAD_GRAYSCALE)
                # Resize the image
                img = cv.resize(img, self.img_size)
                cv.imwrite(file, img)
            except Exception as err:
                self.log_it(str(err))

    def run(self, num_cores, img_dir):
        '''Start image pre-proccessing'''
        # Assign object vars
        self.num_cores = num_cores
        self.img_dir = img_dir

        # Start
        self.log_it("Starting pre-processing of images...")
        self._build_file_lst()

        # Begin multiprocessing
        self.log_it("Processing Images...")
        with Pool(self.num_cores) as p:
            p.map(self._process, self.file_lst)

        self.log_it("Complete. Processed {} images...".format(self.img_count))
