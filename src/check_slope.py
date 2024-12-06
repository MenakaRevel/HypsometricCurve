#!/usr/python
'''
check slope
'''
import copy
import os
import numpy as np
import pandas as pd

#########################
# read final cat 
final_cat=pd.read_csv("./out/finalcat_hru_info_updated_AEcurve.csv")
final_cat.dropna()

print (final_cat['slope'].min(),final_cat['slope'].max())