#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 14:56:25 2023

@author: Ryan Herrin
"""

import os
import sys
import time
import platform
import zipfile
import multiprocessing
from logger import Logger
from datetime import datetime
from CNNGenerator import ModelGen
from ImgProcessing import ImgProcessing


desc = """
Main python script that will coordinate the proceeses of the build. The only
input it will take is the number of processing cores is allocated. This script
will aslo create and write to a log file that track the performance of the
build.
"""


class Track:
    """
    Generic time tracking app used to keep metric information for functions.
    """

    def __init__(self):
        # Starting times
        self.start_time = None
        self.start_datetime = None
        # Ending times
        self.end_time = None
        self.end_datetime = None
        # Duration of function
        self.total_time = None

    def start(self):
        '''Get the starting date and times'''
        self.start_time = time.time()
        self.start_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

    def end(self):
        '''Get the endings dates and times'''
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        self.end_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

        return {'start': self.start_datetime, 'end': self.end_datetime,
                'total_time': self.total_time}


class Build:

    def __init__(self, num_cores=1):
        self.num_cores = num_cores
        self.log_file = None
        self.summary_file = None
        self.model_out_loc = "../models"
        self.img_zip_loc = "../img/imgs.zip"
        self.img_loc = "../img/imgs"
        self.ts_fn = None  # Timestamp to be used in generated filenames

        # Initiate logger
        self.log_obj = Logger(self.__class__.__name__)
        self.log_it = self.log_obj.log_it

        # System information
        self.sys_info = dict()

        # Performance Metrics
        # Values starting with "t_" indicate time metrics
        self.metrics = {
            'build_start': None,
            'build_end': None,
            'build_start_time': str(),
            'num_pics': int(),
            "t_unzip": dict(),
            't_preproc_pics': dict(),
            't_create_model': dict()
        }

        # Find actual number of usable cores if -1 was given for num_cores
        if num_cores == -1 or num_cores == '-1':
            self.num_cores = multiprocessing.cpu_count()

    def _create_log_file(self):
        '''Create log file to write log information'''
        # Get current date and time used for the log file name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.ts_fn = timestamp

        # Assign log file var and Create log file
        self.log_file = "../jobs/build_" + str(timestamp) + ".log"

        with open(self.log_file, 'w') as outfile:
            # Don't need to write anything, just create the file
            outfile.close()
            self.log_obj.set_log_file(self.log_file)

        self.log_it("Created Log file at: {}".format(self.log_file))

        # Write first line to log file
        self.log_it(
            "Build started at: {}".format(self.metrics['build_start_time']),
            write_log_only=True)

    def _get_sys_info(self):
        '''Gather and log information about the system being used'''
        self.log_it("Gathering system information...")

        # Not all system may like this step so we need to loop through and
        # handle exceptions as needed
        target_info = ['Platform', 'Release', 'Version', 'Architecture']
        target_fx = [platform.system, platform.release, platform.version,
                     platform.architecture]

        ti_indx = 0  # Target info index

        def get_info(fx):
            '''Nested function to handle gathering machine info'''
            try:
                plat_info = str(target_info[ti_indx] + ': ' + str(fx()))
                self.sys_info[target_info[ti_indx]] = str(fx())
                self.log_it(plat_info)
            except Exception as err:
                self.log_it(str(err), writelog_it_only=True)
                self.log_it(target_info[ti_indx] + ": NA")
                self.sys_info[target_info[ti_indx]] = "NA"

        # Loop though the platform functions and assign log the values
        for fx in target_fx:
            get_info(fx)
            ti_indx += 1

        # Display # of CPU cores
        self.log_it('# of CPU Cores: {}'.format(self.num_cores))

    def _extract_imgs(self):
        '''Unzip the images from the zip file in the build folder'''
        self.log_it("Extracting Images...")

        track = Track()  # Function Metrics
        track.start()

        # Start extracting the files
        with zipfile.ZipFile(self.img_zip_loc, 'r') as zip_r:
            try:
                zip_r.extractall(self.img_loc)
            except Exception as err:
                self.log_it("Falied to extract images from Zip file")
                self.log_it(str(err))
                sys.exit()

        self.metrics['t_unzip'] = track.end()
        self.log_it("Successfully extracted images. Time taken: {:.4f}".format(
            self.metrics['t_unzip']['total_time']))

    def _process_images(self):
        '''Function to preprocess the extracted images'''
        track = Track()  # Track function metrics
        track.start()

        # Create an object to process the images
        ImgProcessing(log_file=self.log_file).run(
            num_cores=self.num_cores, img_dir=self.img_loc)

        self.metrics['t_preproc_pics'] = track.end()

        self.log_it("Done Pre-Processing Images. Time taken: {:.4f}".format(
            self.metrics['t_preproc_pics']['total_time']))

    def _get_model(self):
        '''Stage that will create and return a classification model'''
        track = Track()  # Track Function metrics
        track.start()

        # Run model generator
        self.model_gen = ModelGen(
            log_file_loc=self.log_file, img_dir=self.img_loc,
            model_out_dir=self.model_out_loc)

        self.model_gen.run()

        # Export model
        self.model_gen.export_model(self.model_out_loc, self.ts_fn)

        self.metrics['t_create_model'] = track.end()

        self.log_it("Done generating model. Time taken: {:.4f}".format(
            self.metrics['t_create_model']['total_time']))

    def _clean(self):
        '''Clean the build and extracted files for the next run'''
        self.log_it("Cleaning Build...")

        # Removed extracted images and the root directory
        lst_of_imgs = os.listdir(self.img_loc)

        for img in lst_of_imgs:
            img_path = os.path.join(self.img_loc, img)
            if os.path.isfile(img_path):
                os.remove(img_path)

        os.rmdir(self.img_loc)

        self.log_it("Finished cleaning build...")

    def run(self):
        '''Coordinator of entire build. All stages are ran through this
        function.
        '''
        self.metrics['build_start'] = time.time()
        self.metrics['build_start_time'] = datetime.now().strftime(
            "%m/%d/%Y %H:%M:%S")

        # Stages
        self._create_log_file()
        self._get_sys_info()
        self._extract_imgs()
        self._process_images()
        self._get_model()
        self._clean()

        # End time
        self.metrics['build_end'] = time.time()

        # Display total time to run the build
        self.log_it("Total Build Time: {:.4f} seconds".format(
            self.metrics['build_end'] - self.metrics['build_start']))


if __name__ == "__main__":
    # Create the build object
    build = Build(num_cores=-1)
    build.run()
