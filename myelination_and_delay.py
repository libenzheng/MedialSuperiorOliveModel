#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def get_conduction_velocity(myelin_thickness):
    _df_mye = pd.read_csv('myelination_parameters.csv').iloc[:,1:]
    _cv_mapping = interp1d(_df_mye['MyelinThickness'].to_numpy(),_df_mye['CV'].to_numpy())
    if myelin_thickness<_df_mye.MyelinThickness.min():
        print('** warning myelin thickness',myelin_thickness,'um is below the interpolation range(0.117-1.5678um)')
        myelin_thickness = _df_mye.MyelinThickness.min()
        print('** auto set myelin thickness as', myelin_thickness,'um')
    if myelin_thickness>_df_mye.MyelinThickness.max():
        print('** warning myelin thickness',myelin_thickness,'um is below the interpolation range(0.117-1.5678um)')
        myelin_thickness = _df_mye.MyelinThickness.max()
        print('** auto set myelin thickness as', myelin_thickness,'um')
    _cv = _cv_mapping(myelin_thickness)
    return _cv 


def get_delay(cv,
              axon_length = 4500,
              heminode_length = 500,
              synaptic_delay = 600,
             ):
    _conduction_delay =  (axon_length - heminode_length)/cv
    _delay = _conduction_delay + synaptic_delay
    return _delay






