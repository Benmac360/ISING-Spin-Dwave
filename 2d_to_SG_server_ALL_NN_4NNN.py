"""
Created on Mon Nov 11 15:46:42 2019

@author: Benjamin
"""

import numpy as np


J = np.loadtxt('Random_couplings.txt')
J2 = np.loadtxt('Random_couplings2.txt')
J3 = np.loadtxt('Random_couplings3.txt')
Lx= 10
f= open("spins_sites_and_couplings.txt","w+")


for ky in range(Lx):
    for kx in range(Lx):
        
        k = (kx + (Lx*ky)) + 1
        
        sr = k+1
        sd = k+Lx
        sur = k - (Lx-1)
        sdr = k + Lx + 1
        srr = k+2
        sdd = k+(2*Lx)
        
        kr = k-ky-1
        kd = k-1
        kur = k-Lx-ky
        kdr = k-1-ky
        krr = k-1 -(2*ky)
        kdd = k-1
             
        if kx<Lx-1: # Right
            strout_right="%6d %6d %10.6f" % (k,sr,J[kr,0]) +'\n'
            f.write(strout_right)           
        if ky<Lx-1:  # down   
            strout_up="%6d %6d %10.6f" % (k,sd,J[kd,1]) +'\n'
            f.write(strout_up)                 
        if ky > 0 and kx !=Lx-1: # Up Right
            # print(k,sur,kur)
            strout_right="%6d %6d %10.6f" % (k,sur,J2[kur,0]) +'\n'
            f.write(strout_right)            
        if ky < (Lx-1) and kx !=Lx-1:  # Down right
            # print(k,sdr,kur)
            strout_up="%6d %6d %10.6f" % (k,sdr,J2[kdr,1]) +'\n'
            f.write(strout_up)           
        if kx<Lx-2: # Right right
            # print(k,srr,krr)
            strout_right="%6d %6d %10.6f" % (k,srr,J3[krr,0]) +'\n'
            f.write(strout_right)  
        if ky<Lx-2:  # down down  
            print(k,sdd,kdd)
            strout_up="%6d %6d %10.6f" % (k,sdd,J3[kdd,1]) +'\n'
            f.write(strout_up)
        
f.close()
