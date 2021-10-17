"""A module containing convenient methods for general machine learning"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from builtins import zip
from builtins import int
from builtins import range
from future import standard_library
standard_library.install_aliases()
from past.utils import old_div
from builtins import object
__author__ = 'patel'

import autograd.numpy as np
import time 

class NumpySeedContext(object):
    """
    A context manager to reset the random seed by numpy.random.seed(..).
    Set the seed back at the end of the block. 
    """
    def __init__(self, seed):
        self.seed = seed 

    def __enter__(self):
        rstate = np.random.get_state()
        self.cur_state = rstate
        np.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.cur_state)

# end NumpySeedContext