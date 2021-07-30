# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:57:18 2019

@author: Benjamin
"""
	
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

single = np.loadtxt('Dwave_energies.txt')
time_single = np.linspace(0,len(single)+1,len(single))
coeffs_single = acf(single, unbiased=False, nlags=int(len(single)), qstat=False, fft=None, alpha=None, missing='none')

#%%
plt.figure(figsize=(10,10))
plt.semilogx((coeffs_single))
plt.hlines(0.0,0,np.amax(time_single))
plt.xlim(0,np.amax(time_single))
plt.show()

np.savetxt('single_step_ACF_coefficients.txt',coeffs_single)
