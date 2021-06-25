# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:46:42 2019

@author: Benjamin
"""

import numpy as np


J = np.loadtxt('Random_couplings.txt')
Lx= 4
f= open("spins_sites_and_couplings.txt","w+")


for kx in range(Lx):
    for ky in range(Lx):
        
        k = (kx + (Lx*ky))
        
        
        if kx<Lx-1:
            kr = k-ky
            strout_right="%6d %6d %10.6f" % (k+1,k+1+1,J[kr,0]) +'\n'
            f.write(strout_right)
            
        if ky<Lx-1:
            kd = k
            strout_up="%6d %6d %10.6f" % (k+1,k+1+Lx,J[kd,1]) +'\n'
            f.write(strout_up)
        
f.close()


        
        
        
        
