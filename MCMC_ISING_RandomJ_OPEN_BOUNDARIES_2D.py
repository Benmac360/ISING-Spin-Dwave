# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:50:19 2019

@author: Benjamin
"""
import numpy as np
import sys
import scipy
import math
from numpy.random import rand
# import matplotlib.pyplot as plt
from numba import jit
#%%
'''
performs a MCMC simulation of 2D ferrom. Ising model
uses single-spin Metropolis algorithm
if verbose==True prints additional info on screen
and writes configurations on file.
'''

def runMCMC(Nspins,Beta,Nvmcsteps,verbose,J,Lx):
    
  '''
  Returns:
    Average energy per spin
    Error of average energy per spin
    configuration generated
    energies of generated configurations
    squared energies
        
  Function also automatically saves configrations and corresponding energies. Also correlated energies, but commented out.
  '''

  print("MCMC simulation")

  configs = np.empty([Nvmcsteps, Lx, Lx]) # Used to record the config after sweeping the system.
  energies_data = []    # Used to record the energy after sweeping the system.
  # energies_correlated = [] # Used to record the energy at every Markov chain step, comment out if needed.
  eavs = []
  eavs2 = []
  sweeping = 8 #Number of times the algorithm sweeps the Lx*Lx matrix.
 
  Nsweep = Nspins*sweeping # number of sinle-spin updates every sweep, before saving configuration.

# initialise the energy values for summation  
  eave = 0.
  eave2 = 0.
  np.random.seed(96321)
  S0 = np.random.randint(0,2,size=(Lx,Lx))*2. -1. 

# perform the Monte Carlo steps 
  for ivmc in range(Nvmcsteps):
     
     
     for isweepk in range(Nsweep):
           

           kx = np.random.randint(0,Lx)
           ky = np.random.randint(0,Lx)
           k  = kx + (Lx *ky)
           
           R = (kx+1)%Lx #right spin with PBC
           L = (kx-1)%Lx #left spin with PBC
           U = (ky+1)%Lx #up spin with PBC
           D = (ky-1)%Lx #down spin with PBC
           
           
           kRx = (kx)
           kRy = ky
           kR  = (kRx + Lx*kRy)
           
           kUx = kx
           kUy = (ky)
           kU  = kUx + (Lx * kUy)
           
           kLx = (kx-1)%Lx
           kLy = ky
           kL  = (kLx + Lx*kLy)
           
           kDx = kx
           kDy = (ky-1)%Lx
           kD  = kDx + (Lx * kDy)
           
           
           
    #   Hastings-Metropolis algorithm S0[k,n]-->-S0[k,n]
           DeltaH = -2.*(-S0[kx,ky])*(S0[R,ky]*J[kR,0]+S0[L,ky]*J[kL,0]\
                                     +S0[kx,U]*J[kU,1]+S0[kx,D]*J[kD,1])
           if DeltaH < 0.: #If the change if negative, towards lower energy we accept.
             S0[kx,ky]=-S0[kx,ky] #accept flip
           elif   np.random.ranf() < np.exp(-Beta*DeltaH): #otherwise we accept based on the probability.
             S0[kx,ky]=-S0[kx,ky] #accept flip
            
            # energies_correlated.append(econf(Lx,J,S0)) # Here we record the energy after each spin flip attempt.
         

#    compute energy every Nsweep updates
     enow = econf(Lx,J,S0)

     eave+= enow
     eave2+=enow**2
     eavs.append(eave)
     eavs2.append(eave2)
     

     ivmcp=ivmc+1 #count sweeps
     configs[ivmc,:,:]=(S0)
     energies_data.append(enow)

     
     if verbose and ivmcp > 1:
       strout="%6d %7.4f %10.6f %9.5f" % (ivmcp,enow,eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.))) +'\n'
       print(strout.strip())

      
       
  ### Saves the data here ####     
  np.save(str(Nspins)+'_lattice_2d_ising_spins',configs)
  np.savetxt(str(Nspins)+'_lattice_2d_ising_avg_energy.txt',energies_data)
  # np.savetxt('correlated_energy.txt',energies_correlated)

  strout='MCMC: %6f %10.6f %9.5f' % (Nvmcsteps,eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)))
  print(strout)
  print("end MCMC")
  
  
# return average E/N and errorbar
  return eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)),configs,eavs,eavs2

'''
econf : Calculates the energy of the configuration using nearest neighbours.

Lx - The length of the side of a square grid.
J - The Spin coupling matrix
S0 - The spins matrix

Returns - Energy per spin
'''
@jit(nopython=True)
def econf(Lx,J,S0):
  energy = 0.
  for i in range(Lx):
      for j in range(Lx):
          
          k = i +(Lx*j)
          
          S = S0[i,j]
          nb = S0[(i+1)%Lx, j]*J[k,0] + S0[i,(j+1)%Lx]*J[k,1]
          energy += -S*nb 
  return energy/(Lx**2)

####################################################################


np.random.seed(12345)
#### Set variables for MC simulation ####
Lx = 10
Nspins = Lx**2
N = Nspins
Beta = 2.0
Nvmcsteps = 100000
verbose = True

##J = np.loadtxt('Random_couplings.txt')
J = (np.random.normal(0.0,1.0,size=(N-Lx,2)))
np.savetxt('Random_couplings.txt',J)

####### Pad the couplings for open boundaries ########
Jh = J[:,0]
Jv = J[:,1]

Jv = np.concatenate([Jv, np.zeros(Lx)])
for i in range(Lx**2):
    if (i+1)%Lx ==0: Jh = np.insert(Jh,i,0.0)
    
J = np.zeros(((N,2)))
J[:,0] = Jh
J[:,1] = Jv
########################################################

### RUNS THE MCMC SIMULATION ###
RETURN_runMCMC=runMCMC(Nspins,Beta,Nvmcsteps,verbose,J,Lx)
