#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:14:13 2022

@author: shubham
"""

import argparse

args=argparse.ArgumentParser()
args.add_argument("--Num1",default="55")
args.add_argument("--Num2",default="55")
parsed_args = args.parse_args()
print(parsed_args)
