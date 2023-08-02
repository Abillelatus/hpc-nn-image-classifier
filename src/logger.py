#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:18:30 2023

@author: Ryan Herrin
"""

from datetime import datetime


class Logger:
    """Simple logging class for standardized output"""

    def __init__(self, class_name, log_file_loc=None, verbose=True):
        self.verbose = verbose
        self.log_file_loc = log_file_loc
        self.class_name = '[' + class_name + ']'

    def set_log_file(self, log_file_loc):
        '''Set the path to the log file if it wasn't initially defined'''
        self.log_file_loc = log_file_loc

    def log_it(self, x, write_log_only=False):
        '''Generic logging function to standardize output'''
        # Time stamp and formatted string
        ts = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        output = str('[' + ts + ']' + self.class_name + ' > ' + x)

        # If write_log_only is set to true then we do not output to the
        # command screen.
        if self.verbose and write_log_only is False:
            print(output)

        # Write to the log file
        if self.log_file_loc is not None:
            with open(self.log_file_loc, 'a') as out_log:
                out_log.write(output + '\n')
                out_log.close()
