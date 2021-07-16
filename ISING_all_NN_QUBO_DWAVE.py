# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:19:49 2021
@author: Benjamin
"""

import numpy as np
from numpy.random import rand
from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector
import dimod


def get_token():
    '''Return your personal access token'''
    
    # TODO: Enter your token here
    return 'DEV-72e917fd4ea4ddce6f280027334fe5b8133d71c4'


Lx=10
N=Lx**2
np.random.seed(12345)
J = (np.random.normal(0.0,1.0,size=(N-Lx,2)))*-1.
J2 = (np.random.normal(0.0,1.0,size=((Lx-1)**2,2)))*-1.
np.savetxt('coplings.txt',J)

Js = {} 
hs = {}   
def get_Js(J=J,Lx=Lx):
    for ky in  range(Lx):
        for kx in  range(Lx):
            
            
            k = kx +(Lx*ky)
            kR  = (k-ky) #coupling to the right of S0[kx,ky]
            kU  = (k-Lx) #coupling to the up of S0[kx,ky]   
            kL  = k-ky-1 #coupling to the left of S0[kx,ky]
            kD  = k      #coupling to the down of S0[kx,ky]
            
            D = k + Lx   #Spin down
            R = k+1      #Spin right
            
            if k < ((Lx*ky) + (Lx-1)):
                JR=J[int(kR),0]*1.
                Js.update({(k,R):JR})
              
            
            if k < (Lx**2-1)-Lx+1:
                JD=J[int(kD),1]*1.
                Js.update({(k,D):JD})
               
                
            UR = k - (Lx-1)         # Spin up right
            DR = k + Lx +1          # spin down right
            
            
            kDR = k-ky              #coupling down right
            kUR = k-Lx - ky     #coupling up right
            
            
            
            
            if k < (Lx*(Lx-1)) and kx !=Lx-1:
                JDR=J2[int(kDR),1]*1.
                Js.update({(k,DR):JDR})
                
                
            if ky > 0 and kx !=Lx-1:
                JUR=J2[int(kUR),0]*1.
                Js.update({(k,UR):JUR})
               
        
    return Js


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

def run_on_qpu(Js,hs, sampler):
    """Runs the QUBO problem Q on the sampler provided.
    Args:
        Q(dict): a representation of a QUBO
        sampler(dimod.Sampler): a sampler that uses the QPU
    """

    sample_set = sampler.sample_ising(h=hs,J=Js, num_reads=numruns, label='ISING Glass open BCs'\
                                     ,reduce_intersample_correlation=True\
                                         ,programming_thermalization=0\
                                             ,annealing_time = 150\
                                                 ,readout_thermalization=0,answer_mode='raw')

    return sample_set

## ------- Main program -------
if __name__ == "__main__":

    
    numruns = 1000
    Js = get_Js()
    
    # bqm = dimod.BQM.from_qubo(Js)
    # sample_set = EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=numruns)
    
    
    qpu_2000q = DWaveSampler(solver={'topology__type': 'pegasus'})

    sampler = EmbeddingComposite(qpu_2000q)
    for k in range(1):
        sample_set = run_on_qpu(Js,hs, sampler)

        print(sample_set)
        configs = []
        energies = []
        
        for i in range(sample_set.record.size):
            for j in range(sample_set.record[i][2]):
                
                S0 = sample_set._record[i]['sample']
        
                S0d = np.reshape(S0,(Lx,Lx),order='F')
                energy = econf(Lx,J,S0d)
                
                
                configs.append(S0d)
                energies.append(energy)
                
        
        
        
        np.save('configs'+str(k)+'.npy',np.asarray(configs))
        np.savetxt('energies'+str(k)+'.txt',energies)


# dwave.inspector.show(sample_set)
