# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:19:30 2019

@author: Benjamin
"""

from numba import jit
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt



#Defind the function to calculate the Hamiltonian
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

'Some Functions'
#Sigmoid fucntion, arguement x can be an array. This ios the fastest call of the function.
@jit(nopython=True)
def Sigm(x):        
    return (1./(1. + np.exp(-x)))

@jit(nopython=True)
def CondProb(V,hd,bf,d):                            # Function to calculate the conditional probability.    
    return Sigm((bf[d] + (np.dot(V[d,:],hd[:]))))

@jit(nopython=True)
def Prob(E,Beta,Nspins):        # Boltzmann probability distribution
    return np.exp(-Beta*Nspins*E)

def minibatch_wf(skm,Nbatch):
   intvect=np.random.randint(0,len(skm),Nbatch)
   lintvect=list(intvect)
   return skm[lintvect]


'Generate test sets from learnt model'
def GenInstance(Nspins,W,V,bf,cf):
    a         = np.copy(cf)
    Px       = 1.
    out_spins = np.zeros(Nspins)
    prob      = np.zeros(Nspins)
     
    
    for i in range(0,Nspins):        
        hd         = np.copy(Sigm(a))       
        prob[i]    = (CondProb(V,hd,bf,i))
         # Log-probability spin up            

        dummy_rand = (np.random.rand())        
        if dummy_rand < prob[i]:
            out_spins[i] = 1.        
        
        Px  = np.copy(Px)*((prob[i]**out_spins[i])*((1-prob[i])**(1-out_spins[i])))
        
        a += np.dot(W[:,i],out_spins[i])  
    return out_spins,Px

def transform_data_to_zero(input_data,Nspins):
    for k in range(input_data.shape[0]):
        for i in range(Nspins):
            if (input_data[k,i] +1.)**2.<0.001:              # Swap the -1 to 0 to make the binary behaviour of algorithm
                input_data[k,i]= 0.
    return input_data


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


def Algorithm(V,W,Sbatch,bf,cf):                    # NADE algorithm for a mini batch of instnaces
    
    # Initialise the summation of the gradients to average at the end of minibatch 
    batchLPx = 0.                                     
    batchdV  = 0.
    batchdW  = 0.
    batchdbf = 0.
    batchdcf = 0.
    
    for j in range(len(Sbatch)):
        
        S   = (Sbatch[j])                             # This sets the spins config to be trained on in algorithm
        af  = np.copy(cf)                                     # a <--- c in psuedocode                
        LPx = 0.                                    # Inialise probability
        Px  = 1.
        
        #Initialise the empty matrices for gradients
        daf = np.zeros(Nhidden)
        dbf = np.zeros(Nspins)
        dcf = np.zeros(Nhidden)
        dV  = np.zeros((Nspins,Nhidden))
        dW  = np.zeros((Nhidden,Nspins))
        hds = np.zeros((Nspins,Nhidden))
              
        for d in range(D):          
#            S[d]         = S[d]                              # input x_{od}
            hd          = np.copy(Sigm(af))                           # dth-hidden layer
            hds[d,:]    = np.copy(hd)                           # Store hidden layer units
            Pcond       = CondProb(V,hds[d,:],bf,d)             # Function call
#            print(Pcond)
#            sys.exit()
            outprobs[d] = np.copy(Pcond)                     # Save conditional probabilites to be used later! 
#            Px          = Px*(Pcond**S[d] * (1-Pcond)**(1-S[d]))
#            LPx         = np.log(Px)  
            LPx         = LPx + S[d]*np.log(CondProb(V,hds[d,:],bf,d)+ 1e-14) + ((1-S[d])*np.log((1-CondProb(V,hds[d,:],bf,d))+ 1e-14)) # Log-probability spin up            
            af         += np.dot(W[:,d],S[d])                # Update a

        for d in np.arange(D-1,-1,-1):
            dbf[d]      = (outprobs[d]-S[d])       
            dV[d,:]     = np.dot((outprobs[d]-S[d]) ,np.transpose(hds[d,:]))
            dh          = np.dot((outprobs[d]-S[d]) ,np.transpose(V[d,:]))
            dcf        += np.multiply(dh,np.multiply(hds[d,:],(1.-hds[d,:])))
            dW[:,d]     = daf*S[d]
            daf        += np.multiply(dh,np.multiply(hds[d,:],(1.-hds[d,:])))
            
    # Mini batch sums to average         
        batchLPx += LPx
        batchdV  += dV
        batchdW  += dW
        batchdbf += dbf
        batchdcf += dcf
    # The returned quantities of this algorithm are averaged   
    return batchLPx/Nbatch,batchdV/Nbatch,batchdW/Nbatch,batchdbf/Nbatch,batchdcf/Nbatch,outprobs



def NEpochs(Nepochs,Nspins,Nhidden,Nbatch,W,bf,cf,input_data,Nconfigs,V): 
    #learning rate
    eta    = 0.45
    decay  = 0.97       
    LPx    = 0. 
    
    
    for Nsteps in range(Nepochs):
        print("Epoch %d" % (Nsteps+1))
        dum_LPx = 0.
        eta = eta * decay
        for k in range(0,int(Nconfigs/Nbatch)):
            
            if (k % LOG_EVERY_N) == 0:
                print("Batch step %d" % (k+1))
                
                
            
            sumW = 0.
            sumb = 0.
            sumc = 0.
            sumV = 0.
           
            Sbatch = minibatch_wf(input_data,Nbatch)  #Get a mini batch from inputs
                      
            LPx,dV,dW,dbf,dcf,outprobs = Algorithm(V,W,Sbatch,bf,cf) # Call mini batch algorithm
                
            #Update gradient terms with a learning rate            
            #Values updated ready to pass onto next epoch
            
            W  -= eta*dW
            bf -= eta*dbf
            cf -= eta*dcf
            V  -= eta*dV
            dum_LPx += LPx
#            LPx += LPx
        glob_avg_LPx = dum_LPx/int(Nconfigs/Nbatch)
        glob_avg_data[Nsteps] = glob_avg_LPx
        print("Global Average LPx")
        print(float(glob_avg_LPx))
     
    return LPx,W,V,bf,cf,outprobs


def MC_NADE(Lx,Nspins,Beta,Nvmcsteps,verbose,J,W,V,bf,cf):

  print("MCMC simulation")
   
# initialisation 
  energies_data = np.zeros(int(Nvmcsteps/10))
  configs_data = np.zeros((int(Nvmcsteps/10),Nspins))
  eave = 0.
  eave2 = 0.
  accepted = 0.
  
# Generate the first instance
  S0_accepted,Px_accepted = GenInstance(Nspins,W,V,bf,cf)
  
  for j in range(Nspins):
         if S0_accepted[j]   == 0.:              # Swap the 0 to -1 to change zeros back to -1
            S0_accepted[j]   = -1.
            
  S0_accepted_reshaped = (S0_accepted).reshape(Lx,Lx)
            
  enow_accepted = econf(Lx,J,S0_accepted_reshaped)    # Calculate the energy.
  
# perform the Monte Carlo steps 
  for ivmc in range(Nvmcsteps):
           
     S0_trial,Px_trial = GenInstance(Nspins,W,V,bf,cf) # Generate the trial configuration
     for j in range(Nspins):
         if S0_trial[j]   == 0.:              # Swap the 0 to -1 to change zeros back to -1
            S0_trial[j]   = -1.
            
     S0_trial_reshaped = (S0_trial).reshape(Lx,Lx)
            
     enow_trial = econf(Lx,J,S0_trial_reshaped)    # Calculate the energy.
         
     '''Here we make the acceptance criterion'''
     Ptrial    = Prob((enow_trial),Beta,Nspins)        
     Paccept   = Prob((enow_accepted),Beta,Nspins) 
     qxx = ((Px_accepted/Px_trial) * (Ptrial/Paccept))    
     Axx = np.amin([1.,qxx]) 

    
     '''Do we accept or reject'''    
     if Axx >= 1.0:         
         enow_accepted  = np.copy(enow_trial)   # Update energy
         Px_accepted    = np.copy(Px_trial)     # Update probability from NADE
         S0_accepted    = np.copy(S0_trial)     # Accept and update the new spin configuration
         accepted       += 1                    # Count new accepted configs

     elif np.random.ranf() < Axx :          
         enow_accepted  = np.copy(enow_trial)   # Update energy
         Px_accepted    = np.copy(Px_trial)     # Update probability from NADE
         S0_accepted    = np.copy(S0_trial)     # Accept and update the new spin configuration
         accepted       += 1                    # Count new accepted configs
         
     
     eave  += enow_accepted                     # Accumlate average
     eave2 += enow_accepted**2                  # Accumlate average squared
     ivmcp=ivmc+1                               # Count steps
     
     if ivmc%10==0:
         energies_data[int(ivmc/10)] = enow_accepted        # Record energies at MC step
         configs_data[int(ivmc/10),:] = S0_accepted         # Record configsurations at MC step.

     
     if verbose and ivmcp > 1:
       strout="%6d %7.4f %10.6f %9.5f" % (ivmcp,enow_accepted,eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.))) +'\n'
       print(strout.strip())
       
       
  configs_data = configs_data.reshape(int(Nvmcsteps/10),Lx,Lx)
  np.save(str(Nspins)+'_lattice_2d_ising_spins_NADEMCMC.npy',configs_data)
  
  strout='MCMC: %6f %10.6f %9.5f' % (Nvmcsteps,eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)))
  print(strout)
  print("end MCMC")
 
  return eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)),configs_data,J,accepted,energies_data


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

Lx = 10
Nspins = Lx**2
N = Nspins
Beta = 0.8 #First beta
Betas = np.array([1.0,1.3,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.0001])     # Betas for iterative process.
Nvmcsteps_start = 100000
verbose = True

np.random.seed(12345)
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
'''
Firstly Perform the MCMC simulation to gain initial data at high temperature (low beta)
Run the function for the inital starting MCMC
'''
RETURN_runMCMC=runMCMC(Nspins,Beta,Nvmcsteps_start,verbose,J,Lx)
np.save('MCMC_2D_spin_glass_'+str(Beta)+'_report.npy',RETURN_runMCMC)

''' 
Obtain and set the configs to train from initial MCMC.
'''
configs_start = RETURN_runMCMC[2]  

''' 
Now we can perform the iterative process of training from high to low temperature
Now we will train the NADE
Initialise first'''
Nhidden         = int(64)    # Set the number of hidden units
Nbatch          = 4         # Number of instances in minibatch
Nepochs         = 50    # Number of trianing epochs

'''To print ever N loops'''
LOG_EVERY_N     = 10000
'''Define D (dimensionality) for the length of the loop muat be Nspins'''
D               = Nspins  


'load Inputs'
'''The spins are -1 and +1, for the algortithm to be correct you must rum this small script once to load in the input data'''
'''then to change the inputs of -1 to 0. Once run it can be highlighted out to test the code.'''
 
input_data = configs_start      # Spin inputs
input_data = input_data.reshape(len(input_data),N)
Nconfigs = int(25000)

input_data = transform_data_to_zero(input_data,Nspins)

'''Run the sequential tempering procedure'''

for kk in range(len(Betas)):
    
    if kk==0: # This randomly initalises the weights at the first step in the procedure. Following steps take weights from the previous beta step.
    
        #initial random values of weight parameters
        Wave            = 0.03
        Wran            = 0.0075
        #initialse weights
        W               = np.ones([Nhidden,Nspins])*Wave-np.random.rand(Nhidden,Nspins)*Wran
        V               = np.ones([Nspins,Nhidden])*Wave-np.random.rand(Nspins,Nhidden)*Wran             # Set visible units as spins
        glob_avg_data   = np.zeros(Nepochs)
        #Initialise the bias terms
        bf              = np.ones(Nspins)#*0.8-np.random.rand(Nspins)*0.001
        cf              = np.ones(Nhidden)#*0.65-np.random.rand(Nhidden)*0.002   
        #Empty matrix to store the conditionla probabilites to use later!

    outprobs        = np.zeros(Nspins)                       
    rep= NEpochs(Nepochs,Nspins,Nhidden,Nbatch,W,bf,cf,input_data,len(input_data),V)  
    np.savetxt(str(Beta)+'log_probs.txt',glob_avg_data)
                
#Run the program to generate random spin configurations and check if they follow the distribution'''
    Nvmcsteps2 = 1000000
    verbose   = True  
    V   = rep[2]
    bf  = rep[3]
    W   = rep[1]
    cf  = rep[4]
     
    
    Beta = Betas[kk]
    if Beta == 4.0:
        report = MC_NADE(Lx,Nspins,Beta,Nvmcsteps2,verbose,J,W,V,bf,cf)
    else:
        report = MC_NADE(Lx,Nspins,Beta,Nvmcsteps2,verbose,J,W,V,bf,cf)
         
    averaged = sum(report[5])/Nvmcsteps2
    
     
    np.savetxt(str(Beta)+'_lattice_2d_ising_avg_energy_MCMC+NADE.txt',report[5])
    np.save(str(Beta)+'_MCMC+NADE_report.npy',report)
    
    input_data = np.copy(report[2])
    input_data = input_data.reshape(len(input_data),Nspins)
    input_data = transform_data_to_zero(input_data,Nspins)
    

np.savetxt('V-weights.txt',V)
np.savetxt('W-weights.txt',W)
np.savetxt('bf-bias.txt',bf)
np.savetxt('cf-bias.txt',cf)

