#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    :   setup.py
@Time    :   2021-08-01 19:22:37
@Author  :   null
@Link    :   https://blog.csdn.net/REAL_liudebai
@Version :   1.0
"""

# here put the import lib
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["run_cnn_fortest.py"]))
