import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.mod_det import *

k_arr = [10, 5, 15]
d_arr = [2, 4, 4]
a_arr = [1, 1, 1]
h_arr = [0.3*1.6, 0.1*1.6, 0.2*1.6]
a_limit = 25
a_mult = np.arange(0, 6)

res = [eoq_multi(k_arr, d_arr, h_arr, a_arr, a_limit + 5*i) for i in a_mult]

