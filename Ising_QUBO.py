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
    return 'DEV-aa60123b45c8ffd88aabc6ab22ae723efe81e6e8'


Lx=3
N=Lx**2
np.random.seed(23451)
J = (np.random.normal(0.0,1.0,size=(N,2)))
np.savetxt('coplings.txt',J)

Js = {} 
hs = {}   
def get_Js(J=J,Lx=Lx):
    for ky in  np.linspace(0,Lx-1,Lx):
        for kx in  np.linspace(0,Lx-1,Lx):
    
            
            k  = kx + (Lx *ky)
            print('index k is',k)
            
            kRx = (kx)
            kRy = ky
            kR  = (kRx + Lx*kRy)
            print(kR)
            kUx = kx
            kUy = (ky)
            kU  = kUx + (Lx * kUy)
            print(kU)
            kLx = (kx-1)%Lx
            kLy = ky
            kL  = (kLx + Lx*kLy)
            print(kL)
            kDx = kx
            kDy = (ky-1)%Lx
            kD  = kDx + (Lx * kDy)
            print(kD)
            print()
            print()
            JL=J[int(kL),0]*1.
            JD=J[int(kD),1]*1.
            JR=J[int(kR),0]*1.
            JU=J[int(kU),1]*1.
            hs.update({(k):0.0})
            Js.update({(k,kL):JL}) 
            Js.update({(k,kD):JD}) 
            # Js.update({(kR,k):JR}) 
            # Js.update({(kU,k):JU}) 
    return Js


def econf(S0,Lx=Lx,J=J):
  energy = 0.
  for i in range(Lx):
      for j in range(Lx):
          
          k = i +(Lx*j)
          
          S = S0[i,j]
          nb = S0[(i-1)%Lx, j]*J[k,0] + S0[i,(j-1)%Lx]*J[k,1]
          energy += -S*nb 
  return energy/(Lx**2)

def run_on_qpu(Js,hs, sampler):
    """Runs the QUBO problem Q on the sampler provided.

    Args:
        Q(dict): a representation of a QUBO
        sampler(dimod.Sampler): a sampler that uses the QPU
    """

    sample_set = sampler.sample_ising(h=hs,J=Js, num_reads=numruns, label='Training - Choosing Boxes'\
                                     ,reduce_intersample_correlation=True\
                                         ,programming_thermalization=100\
                                             ,annealing_time = 100\
                                                 ,readout_thermalization=100\
                                     ,postprocess='sampling',beta=2,answer_mode='histogram',chain_strength = 4.0)

    return sample_set

## ------- Main program -------
if __name__ == "__main__":

    
    numruns = 5
    Js = get_Js()
    
    # bqm = dimod.BQM.from_qubo(Js)
    # sample_set = EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=numruns)
    
    
    qpu_2000q = DWaveSampler(solver={'topology__type': 'chimera'})

    sampler = EmbeddingComposite(qpu_2000q)

    sample_set = run_on_qpu(Js,hs, sampler)

    print(sample_set)
    configs = []
    energies = []
    
    for i in range(sample_set.record.size):
        for j in range(sample_set.record[i][2]):
            
            S0 = sample_set._record[i]['sample']
      
            # S0d = np.reshape(S0,(Lx,Lx),order='C')
            # energy = econf(S0d)
            
            
            configs.append(S0)
            # energies.append(energy)
            
    
    
    # configs = sample_set._record[:]['sample']
    configs = np.asarray(configs)


# dwave.inspector.show(sample_set)
