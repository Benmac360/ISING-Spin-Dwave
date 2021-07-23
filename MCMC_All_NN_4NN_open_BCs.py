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
# @jit(nopython=True)
def runMCMC(Nspins,Beta,Nvmcsteps,verbose,J,J2,J3,Lx,connect):
    
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
  np.random.seed(235)
  S0 = np.random.randint(0,2,size=(Lx,Lx))*2. -1. 

# perform the Monte Carlo steps 
  for ivmc in range(Nvmcsteps):
     
     
     for isweepk in range(Nsweep):
           

           kx = np.random.randint(0,Lx)
           ky = np.random.randint(0,Lx)
           k  = kx + (Lx *ky)
           
           R = (kx+1)%Lx #right spin with PBC
           L = (kx-1)%Lx #left spin with PBC
           U = (ky-1)%Lx #up spin with PBC
           D = (ky+1)%Lx #down spin with PBC
           
           S_R = S0[(kx+1)%Lx,ky]
           S_L = S0[(kx-1)%Lx,ky]
           S_U = S0[kx,(ky-1)%Lx]
           S_D = S0[kx,(ky+1)%Lx]
           
           kRx = (kx)
           kRy = ky
           kR  = (kRx + Lx*kRy)
           
           kUx = kx
           kUy = (ky-1)%Lx
           kU  = kUx + (Lx * kUy)
           
           kLx = (kx-1)%Lx
           kLy = ky
           kL  = (kLx + Lx*kLy)
           
           kDx = kx
           kDy = (ky)
           kD  = kDx + (Lx * kDy)
           
           
           
           ########### Diagonal -neighbours #############
           S_UR = S0[(kx+1)%Lx,(ky-1)%Lx]
           S_UL = S0[(kx-1)%Lx,(ky-1)%Lx]
           S_DR = S0[(kx+1)%Lx,(ky+1)%Lx]
           S_DL = S0[(kx-1)%Lx,(ky+1)%Lx]
           
           kUL = (kx-1)%Lx +((ky-1)%Lx * Lx)
           kDR = kx + (Lx *ky)
           kUR = kx + (Lx *ky)
           kDL = (kx-1)%Lx + ((ky+1)%Lx * Lx)
           
           
           ########### Next nearest neighbours #############
           S_RR = S0[(kx+2)%Lx,ky]
           S_LL = S0[(kx-2)%Lx,ky]
           S_UU = S0[kx,(ky-2)%Lx]
           S_DD = S0[kx,(ky+2)%Lx]
           
           kRR = k +(ky*2)+2
           kLL = k + (ky*2)
           kUU = k
           kDD = k +(Lx*2)
           
           

      
    #   Hastings-Metropolis algorithm S0[k,n]-->-S0[k,n]
           if connect == 1:
               DeltaH = -2.*(-S0[kx,ky])*(S_R*J[kR,0]+S_L*J[kL,0]\
                                         +S_U*J[kU,1]+S_D*J[kD,1])
           if connect == 2:
               DeltaH = -2.*(-S0[kx,ky])*(S_R*J[kR,0]+S_L*J[kL,0]\
                                         +S_U*J[kU,1]+S_D*J[kD,1]\
                                         +S_UR*J2[kUR,0]+S_UL*J2[kUL,1]\
                                         +S_DR*J2[kDR,1]+S_DL*J2[kDL,0])
                   
           if connect == 3:
               DeltaH = -2.*(-S0[kx,ky])*(S_R*J[kR,0]+S_L*J[kL,0]\
                                         +S_U*J[kU,1]+S_D*J[kD,1]\
                                         +S_UR*J2[kUR,0]+S_UL*J2[kUL,1]\
                                         +S_DR*J2[kDR,1]+S_DL*J2[kDL,0]\
                                         +S_RR*J3[kRR,0]+S_LL*J3[kLL,0]\
                                         +S_UU*J3[kUU,1]+S_DD*J3[kDD,1])
           # print(DeltaH)
           if DeltaH < 0.: #If the change if negative, towards lower energy we accept.
             S0[kx,ky]=-S0[kx,ky] #accept flip
           elif   np.random.ranf() < np.exp(-Beta*DeltaH): #otherwise we accept based on the probability.
             S0[kx,ky]=-S0[kx,ky] #accept flip
            
           # energies_correlated.append(econf(Lx,J,J2,J3,S0)) # Here we record the energy after each spin flip attempt.
         

#    compute energy every Nsweep updates
     enow = econf(Lx,J,J2,J3,S0)
     
     # energy = 0.
     # for i in range(Lx):
     #     for j in range(Lx):
     #         k = i +(Lx*j)
     #         if connect ==1:
     #             S = S0[i,j]
     #             nb = S0[(i+1)%Lx, j]*J[k,0] + S0[i,(j+1)%Lx]*J[k,1]
     #             energy += -S*nb 
     #         if connect ==2:
     #             S = S0[i,j]
     #             nb = S0[(i+1)%Lx, j]*J[k,0] + S0[i,(j+1)%Lx]*J[k,1]\
     #                 +S0[(i+1)%Lx,(j+1)%Lx]*J2[k,1]+S0[(i+1)%Lx,(j-1)%Lx]*J2[k,0]
     #             energy += (-S*nb)/(Lx**2)
     # enow = energy

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
  np.save(str(Nspins)+'_lattice_2d_ising_spins_at_beta='+str(Beta),configs)
  np.savetxt(str(Nspins)+'_lattice_2d_ising_avg_energy_at_beta='+str(Beta)+'.txt',energies_data)
  # np.savetxt('correlated_energy_at_beta='+str(Beta)+'.txt',energies_correlated)

  strout='MCMC: %6f %10.6f %9.5f' % (Nvmcsteps,eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)))
  print(strout)
  print("end MCMC")
  
  
# return average E/N and errorbar
  return eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)),configs,eavs,eavs2,energies_data

'''
econf : Calculates the energy of the configuration using nearest neighbours.
Lx - The length of the side of a square grid.
J - The Spin coupling matrix
S0 - The spins matrix
Returns - Energy per spin
'''
@jit(nopython=True)
def econf(Lx,J,J2,J3,S0,connect =3):
  energy = 0.
  for i in range(Lx):
      for j in range(Lx):
          
          k = i +(Lx*j)
          if connect ==1:
              S = S0[i,j]
              nb = S0[(i+1)%Lx, j]*J[k,0] + S0[i,(j+1)%Lx]*J[k,1]             # right and down
              energy += -S*nb 
              
          if connect ==2:
              S = S0[i,j]
              nb = S0[(i+1)%Lx, j]*J[k,0] + S0[i,(j+1)%Lx]*J[k,1]\
                  +S0[(i+1)%Lx,(j+1)%Lx]*J2[k,1]+S0[(i+1)%Lx,(j-1)%Lx]*J2[k,0]
              energy += -S*nb 
              
          if connect ==3:
              S = S0[i,j]
              nb = S0[(i+1)%Lx, j]*J[k,0] + S0[i,(j+1)%Lx]*J[k,1]\
                  +S0[(i+1)%Lx,(j+1)%Lx]*J2[k,1]+S0[(i+1)%Lx,(j-1)%Lx]*J2[k,0]\
                  +S0[(i+2)%Lx, j]*J3[(k +(j*2)+2),0] + S0[i,(j+2)%Lx]*J3[k+(2*Lx),1]
              energy += -S*nb 
  return energy/(Lx**2)

####################################################################


np.random.seed(12345)
#### Set variables for MC simulation ####
Lx = 10
Nspins = Lx**2
N = Nspins
Beta = 2.0
Nvmcsteps =10000
verbose = True
connect = 1
##J = np.loadtxt('Random_couplings.txt')
J = (np.random.normal(0.0,1.0,size=(N-Lx,2)))
J2 = (np.random.normal(0.0,1.0,size=((Lx-1)**2,2)))
J3 = (np.random.normal(0.0,1.0,size=(((Lx-2)*Lx),2)))

np.savetxt('Random_couplings.txt',J)
np.savetxt('Random_couplings2.txt',J2)
np.savetxt('Random_couplings3.txt',J3)

####### vertical and horizontal neighbours ########
Jh = J[:,0]
Jv = J[:,1]

Jv = np.concatenate([Jv, np.zeros(Lx)])
for i in range(Lx**2):
    if (i+1)%Lx ==0: Jh = np.insert(Jh,i,0.0)
    
J = np.zeros(((N,2)))
J[:,0] = Jh
J[:,1] = Jv

################## Diagonal neighbours ##################
Jlrd = J2[:,1]
Jlru = J2[:,0]
for i in range(Lx**2-Lx):
    if (i+1)%Lx ==0: Jlrd = np.insert(Jlrd,i,0.0)
Jlrd = np.concatenate([Jlrd, np.zeros(Lx)])

for i in range(Lx):
    Jlru = np.insert(Jlru,i,0.0)
    
for i in range(Lx,Lx**2):
    if (i+1)%Lx ==0: Jlru = np.insert(Jlru,i,0.0)
       
J2 = np.zeros(((Lx**2,2)))
J2[:,1] = Jlrd
J2[:,0] = Jlru

################## next nearest neighbours ##################
Jhh = J3[:,0]
Jvv = J3[:,1]

for i in range(Lx*2):
    Jvv = np.insert(Jvv,i,0.0)
Jvv = np.concatenate([Jvv, np.zeros(Lx*2)])



Jhh = Jhh.reshape(Lx,Lx-2)
Jhh = np.insert(Jhh,0,np.zeros(Lx),axis=1)
Jhh = np.insert(Jhh,0,np.zeros(Lx),axis=1)
Jhh = np.insert(Jhh,len(Jhh),np.zeros(Lx),axis=1)
Jhh = np.insert(Jhh,len(Jhh),np.zeros(Lx),axis=1)
Jhh = Jhh.reshape(np.size(Jhh))


J3 = np.zeros(((np.size(Jvv),2)))
J3[:,0] = Jhh
J3[:,1] = Jvv

########################################################

### RUNS THE MCMC SIMULATION ###

RETURN_runMCMC=runMCMC(Nspins,Beta,Nvmcsteps,verbose,J,J2,J3,Lx,connect=3)
    

