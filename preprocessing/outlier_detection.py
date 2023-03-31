import pandas as pd
import numpy as np

indices_time = ['Index - 9',
 'Index - 8',
 'Index - 7',
 'Index - 6',
 'Index - 5',
 'Index - 4',
 'Index - 3',
 'Index - 2',
 'Index - 1',
 'Index - 0']

def remove_outlier(returns_df):
    mask = np.abs(returns_df[indices_time]) > 1e-4
    mask = mask.sum(axis=1) > len(indices_time)//2
    return returns_df[mask]
