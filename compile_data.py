# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:51:29 2021

@author: Benjamin
"""

import numpy as np 

Lx=10
N=Lx**2
np.random.seed(12345)
J = (np.random.normal(0.0,1.0,size=(N-Lx,2)))*-1.
configurations = []
energy_list = []


for i in range(10):
    

    configurations.append(np.load('configs'+str(i)+'.npy'))
    energy_list.append(np.loadtxt('energies'+str(i)+'.txt'))
    
configurations =  np.reshape(np.array(configurations),(100000,Lx,Lx), order='F')
energy_list = np.array(energy_list).reshape(100000)

np.save('Dwave_configs.npy',configurations)
np.savetxt('Dwave_energies.txt',energy_list)

def econf(Lx,J,S0):
  energy = 0.
  for kx in range(Lx):
      for ky in range(Lx):
          
          k = kx +(Lx*ky)
          R = (kx+1)  #right spin 
          L = (kx-1)  #left spin 
          U = (ky-1)  #up spin
          D = (ky+1)  #down spin 

          kR  = (k-ky) #coupling to the right of S0[kx,ky]
          kU  = (k-Lx) #coupling to the up of S0[kx,ky]   
          kL  = k-ky-1 #coupling to the left of S0[kx,ky]
          kD  = k      #coupling to the down of S0[kx,ky]
           
          try: Rs = S0[R,ky]*J[kR,0]   # Tries to find a spin to right, if no spin, contribution is 0.
          except: Rs = 0.0

          try: Ds = S0[kx,D]*J[kD,1]
          except: Ds = 0.0

          nb = Rs + Ds #+ Ls + Us
          S = S0[kx,ky]
          energy += -S*nb 
  return energy/(Lx**2)


E = econf(10,J,configurations[1,:,:])