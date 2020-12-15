#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:00:21 2020

@author: eja26438
"""

import h5py
from swmr_tools.KeyFollower import Follower
import numpy as np
from queue import Queue
from threading import Thread


class BinOp():
    
    def __init__(self,
                 hdf5_file,
                 key_datasets,
                 dataset_1,
                 dataset_2):
        
        self.hdf5_file = hdf5_file
        self.key_datasets = key_datasets
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.queue = Queue()
        
    
    def _key_generator(self):
        with h5py.File(self.hdf5_file, "r", swmr = True) as f:
            kf = Follower(f, self.key_datasets, timeout = 0.1)
            for key in kf:
                self.queue.put(key)
            self.queue.put("End")
            
    def _frame_consumer(self, fn):
        return_list = []
        key = self.queue.get()
        while key != 'End':
            return_list.append(fn(key))
            key = self.queue.get()
        return return_list
            
    
    def run(self, fn):
        queue = Queue()



