# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:00:42 2021

@author: Benjamin
"""

from pathlib import Path
import datetime

import numpy as np
from numba import jit
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset

from tqdm import trange, tqdm



class NADE(nn.Module):
    """NADE for binary MNIST"""

    def __init__(self, input_dim, hidden_dim):
        super(NADE, self).__init__()
        self.D = input_dim
        self.H = hidden_dim
        self.params = nn.ParameterDict(
            {
                "V": nn.Parameter(torch.zeros(self.D, self.H)),
                "b": nn.Parameter(torch.zeros(self.D)),
                "W": nn.Parameter(torch.zeros(self.H, self.D)),
                "c": nn.Parameter(torch.zeros(self.H)),
            }
        )
        # init params
        nn.init.kaiming_uniform_(self.params["V"])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.params["V"])
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.params["b"], -bound, bound)

        nn.init.kaiming_uniform_(self.params["W"])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.params["W"])
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.params["c"], -bound, bound)

    def forward(self, x):
        # a: (H, N)
        a = self.params["c"].view(-1, 1).expand(self.params["c"].size(0), x.size(0))
        # b : (N, D)
        b = self.params["b"].expand(x.size(0), -1)
        # Compute p(x)
        x_hat = self._cal_prob(a, b, x)
        return x_hat

    def _cal_prob(self, a, b, x, sample=False):
        """
        assert 'x = None' when sampling
        Parameters:
         - a : (H, N)
         - b : (N, D)
         - x : (B, D)

        Return:
         - x_hat: (B, D), estimated probability dist. of batch data
        """
        if sample:
            assert x is None, "No input for sampling as first time"

        x_hat = []  # (N, 1) x D
        xs = []
        for d in range(self.D):
            # h_d: HxN
            h_d = torch.sigmoid(a)
            # p_hat: b_d + h_d.t x V_d.t -> (NxH x Hx1) + Nx1
            p_hat = torch.sigmoid(
                b[:, d : d + 1] + torch.mm(h_d.t(), self.params["V"][d : d + 1, :].t())
            )
            # fill the out prob array
            x_hat.append(p_hat)

            if sample:
                # random sample
                x = torch.bernoulli(p_hat)
                xs.append(x)
                a = a + self.params["W"][:, d : d + 1] * x.t()

            else:
                # a: a + W_d*x_d.t -> HxN +(Hx1 * 1xN)
                a = a + self.params["W"][:, d : d + 1] * x[:, d : d + 1].t()

        # x_hat: (N, D), estimated probability dist. of batch data
        x_hat = torch.cat(x_hat, 1)
        if sample:
            xs = torch.cat(xs, 1)
            return x_hat, xs
        return x_hat

    def _cal_nll(self, x_hat, x):
        nll_loss = x * torch.log(x_hat) + (1 - x) * torch.log(1 - x_hat)
        return nll_loss.view(nll_loss.shape[0], -1).sum(dim=-1)

    def sample(self, n=1):
        a = self.params["c"].view(-1, 1).expand(self.params["c"].size(0), n)
        b = self.params["b"].expand(n, -1)
        # Compute p(x)
        x_hat, xs = self._cal_prob(a, b, x=None, sample=True)
        # reshape x_hat amd sample
        L = int(math.sqrt(x_hat.shape[1]))
        x_hat = x_hat.view((-1, L, L))
        xs = xs.view((-1, L, L))
        # compute prob of the sample
        nll_loss = self._cal_nll(x_hat, xs)
        # output shold be {-1,1}, xs is {0,1}
        xs = xs * 2 - 1
        return {
            "sample": xs.detach().numpy(),
            "log_prob": nll_loss.detach().numpy(),
        }
    
class Ising2D(Dataset):
    def __init__(self, name: str, path: str, **kwargs):
        super().__init__()
        self.path = path
        self.name = name

        self.dataset = TensorDataset(
            torch.from_numpy(np.load(self.path)).unsqueeze(1).float()
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        # [0] is needed because only one element is returned
        return self.dataset[index][0]

    def __repr__(self) -> str:
        return f"MyDataset(name={self.name}, path={self.path})"


def train(train_loader, criterion, optimizer, model, device):
    model.train()
    total_loss = 0.0

    prog_bar = tqdm(train_loader, leave=False)
    for count, imgs in enumerate(prog_bar):
        optimizer.zero_grad()
        # preprocess to binary
        inputs = imgs.view(imgs.size(0), -1).gt(0.0).float().to(device)
        x_hat = model(inputs)
        loss = criterion(x_hat, inputs)
        loss.backward()
        optimizer.step()

        # record
        total_loss += loss.item()

        prog_bar.set_description(f"Train Loss {loss.item():.4f}", refresh=True)
    return total_loss / (count + 1)


def test(test_loader, criterion, model, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for count, imgs in enumerate(test_loader):
            # preprocess to binary
            inputs = imgs.view(imgs.size(0), -1).gt(0.0).float().to(device)
            x_hat = model(inputs)
            loss = criterion(x_hat, inputs)
            total_loss += loss.item()
    return total_loss / (count + 1)


def sample(model, n=1):
    model.eval()
    out = model.sample(n)

    save_path = Path(".").absolute()
    size = out["sample"].shape[-1]
    save_name = "size-" + str(size) + "_sample-" + str(n) + "_nade"

    print("\nSaving sample generated by NADE as", save_name)
    np.savez(save_path / save_name, **out)

    # fig, axes = plt.subplots(1, 2)
    # for x, ax in zip([x_hat, xs], axes):
    #     ax.matshow(x.cpu().detach().squeeze().view(28, 28).numpy())
    #     ax.axis("off")
    # plt.title(
    #     f"\t[Random Sampling] NLL loss: {nll_loss:.4f}", fontdict={"fontsize": 16}
    # )
    # plt.show()


def decreasing(val_losses, best_loss, min_delta=1e-3):
    """for early stopping"""
    try:
      is_decreasing = val_losses[-1] < best_loss - min_delta
    except:
      is_decreasing = True
    return is_decreasing

def runMCMC(Nspins,Beta,Nvmcsteps,verbose,J,J2,J3,Lx,connect=3):
    
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
  sweeping = 12 #Number of times the algorithm sweeps the Lx*Lx matrix.
 
  Nsweep = Nspins*sweeping # number of sinle-spin updates every sweep, before saving configuration.

# initialise the energy values for summation  
  eave = 0.
  eave2 = 0.
  np.random.seed(56235)
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

@jit(nopython=True)
def econf(Lx,J,J2,J3,S0,connect=3):
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


def MC_NADE(Lx,Nspins,Beta,Nvmcsteps,verbose,J,J2,J3,W,V,bf,cf,ar):

  print("MCMC simulation")
   
# initialisation 
  energies_data = []
  configs_data = []
  eave = 0.
  eave2 = 0.
  accepted = 0.
  
# Generate the first instance
  S0_accepted,Px_accepted = GenInstance(Nspins,W,V,bf,cf)
  
  for j in range(Nspins):
         if S0_accepted[j]   == 0.:              # Swap the 0 to -1 to change zeros back to -1
            S0_accepted[j]   = -1.
            
  S0_accepted_reshaped = (S0_accepted).reshape(Lx,Lx)
            
  enow_accepted = econf(Lx,J,J2,J3,S0_accepted_reshaped)    # Calculate the energy.
  
# perform the Monte Carlo steps 
  for ivmc in range(Nvmcsteps):
           
     S0_trial,Px_trial = GenInstance(Nspins,W,V,bf,cf) # Generate the trial configuration
     for j in range(Nspins):
         if S0_trial[j]   == 0.:              # Swap the 0 to -1 to change zeros back to -1
            S0_trial[j]   = -1.
            
     S0_trial_reshaped = (S0_trial).reshape(Lx,Lx)
            
     enow_trial = econf(Lx,J,J2,J3,S0_trial_reshaped)    # Calculate the energy.
         
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
     
     if ivmc%int(1/ar)==0:
         energies_data.append(enow_accepted)        # Record energies at MC step
         configs_data.append(S0_accepted)         # Record configsurations at MC step.

     
     if verbose and ivmcp > 1:
       strout="%6d %7.4f %10.6f %9.5f" % (ivmcp,enow_accepted,eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.))) +'\n'
       print(strout.strip())
       
       
  configs_data = np.array(configs_data)
  configs_data = configs_data.reshape(int((configs_data.size)/Lx**2),Lx,Lx)

  np.save(str(Nspins)+'_lattice_2d_ising_spins_NADEMCMC.npy',configs_data)
  
  strout='MCMC: %6f %10.6f %9.5f' % (Nvmcsteps,eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)))
  print(strout)
  print("end MCMC")
 
  return eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)),configs_data,J,accepted,energies_data




def main(model_path=None, num_sample=1):
    Lx = 10
    Nspins = Lx**2
    N = Nspins
    input_dim = N
    Beta = 0.8 #First beta
    Betas = np.array([1.0,1.3,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.0001])     # Betas for iterative process.
    Nvmcsteps_start = 10000
    verbose = True
    
    np.random.seed(12345)
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
    
    '''
    Firstly Perform the MCMC simulation to gain initial data at high temperature (low beta)
    Run the function for the inital starting MCMC
    '''
    RETURN_runMCMC=runMCMC(Nspins,Beta,Nvmcsteps_start,verbose,J,J2,J3,Lx)
    np.save('MCMC_2D_spin_glass_'+str(Beta)+'_report.npy',RETURN_runMCMC)
    
    



    now = datetime.datetime.now()
    date = now.strftime("%Y%m%d_%H%M")
    # check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load the model
    model = NADE(input_dim=N, hidden_dim=64).to(device)
    
    
    # Start the training
    data_path = "./"

    # parameters
    epochs = 30
    batch_size = 250
    # scheduler_step = 80
    # scheduler_gamma = 0.98
    learning_rate = 0.03
    patience = 5
    ''' 
    Obtain and set the configs to train from initial MCMC.
    '''
    configs_start = RETURN_runMCMC[2] 
    np.save("train_"+str(N)+"_lattice_2d_ising_spins.npy",configs_start[:int(len(configs_start)/9)])
    np.save("val_"+str(N)+"_lattice_2d_ising_spins.npy",configs_start[int(len(configs_start)/9):])
    train_set = Ising2D(
        name="Train Ising", path=data_path + "train_"+str(N)+"_lattice_2d_ising_spins.npy"
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = Ising2D(
        name="Valid Ising", path=data_path + "val_"+str(N)+"_lattice_2d_ising_spins.npy"
    )
    test_loader = DataLoader(test_set, batch_size=2*batch_size, shuffle=False)

    criterion = nn.BCELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # start main
    train_losses = []
    test_losses = []
    best_loss = 9999
    wait = 0
    prog_bar = trange(epochs, leave=True, desc="Epoch")
    for step in prog_bar:
        criterion = nn.BCELoss(reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # print(f"Running Step: [{step+1}/{epochs}]")
        train_loss = train(train_loader, criterion, optimizer, model, device)
        test_loss = test(test_loader, criterion, model, device)
        prog_bar.set_postfix(
            test_loss=f"{test_loss:.4f}", train_loss=f"{train_loss:.4f}"
        )
        scheduler.step()
        
        learning_rate = learning_rate

        # record
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        wait += 1
        if decreasing(test_losses, best_loss):
          wait = 0 

        if (wait>patience):
          print(f"Early Stopping")
          break

        if test_loss <= best_loss:
            best_loss = test_loss
            # torch.save(model.state_dict(), f"/content/drive/MyDrive/DL-projects/nade/nade-{date}.pt")
            torch.save(model.state_dict(), f'./nade-{date}'+str(Beta)+'.pt')
            
    for Beta in Betas:
    
    
                
    #Run the program to generate random spin configurations and check if they follow the distribution'''
        
        check_steps = 20000
        verbose   = True  
       
        # Beta = Betas[kk]
        W = model.params["W"].cpu().detach().numpy().astype('float64')
        V = model.params["V"].cpu().detach().numpy().astype('float64')
        bf = model.params["b"].cpu().detach().numpy().astype('float64')
        cf = model.params["c"].cpu().detach().numpy().astype('float64')
        rep_check = MC_NADE(Lx,Nspins,Beta,check_steps,verbose,J,J2,J3,W,V,bf,cf,ar=1.0)
        ar = rep_check[4]/check_steps
        Nvmcsteps2 = int(10000/ar)
        report = MC_NADE(Lx,Nspins,Beta,Nvmcsteps2,verbose,J,J2,J3,W,V,bf,cf,ar)             
        averaged = sum(report[5])/Nvmcsteps2
        configs_start = report[2] 
        np.save("train_"+str(N)+"_lattice_2d_ising_spins.npy",configs_start[:int(len(configs_start)/9)])
        np.save("val_"+str(N)+"_lattice_2d_ising_spins.npy",configs_start[int(len(configs_start)/9):])
        train_set = Ising2D(
            name="Train Ising", path=data_path + "train_"+str(N)+"_lattice_2d_ising_spins.npy"
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        
        test_set = Ising2D(
            name="Valid Ising", path=data_path + "val_"+str(N)+"_lattice_2d_ising_spins.npy"
        )
        test_loader = DataLoader(test_set, batch_size=2*batch_size, shuffle=False)
        
        
        
        
        for step in prog_bar:
            criterion = nn.BCELoss(reduction="mean")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            # print(f"Running Step: [{step+1}/{epochs}]")
            train_loss = train(train_loader, criterion, optimizer, model, device)
            test_loss = test(test_loader, criterion, model, device)
            prog_bar.set_postfix(
                test_loss=f"{test_loss:.4f}", train_loss=f"{train_loss:.4f}"
            )
            scheduler.step()
            
            learning_rate = learning_rate
    
            # record
            train_losses.append(train_loss)
            test_losses.append(test_loss)
    
            wait += 1
            if decreasing(test_losses, best_loss):
              wait = 0 
    
            if (wait>patience):
              print(f"Early Stopping")
              break
    
            if test_loss <= best_loss:
                best_loss = test_loss
                # torch.save(model.state_dict(), f"/content/drive/MyDrive/DL-projects/nade/nade-{date}.pt")
                torch.save(model.state_dict(), f'./nade-{date}'+str(Beta)+'.pt')
        
         
    #     np.savetxt(str(Beta)+'_lattice_2d_ising_avg_energy_MCMC+NADE.txt',report[5])
    #     np.save(str(Beta)+'_MCMC+NADE_report.npy',report)
        
    #     input_data = np.copy(report[2])
    #     input_data = input_data.reshape(len(input_data),Nspins)
    #     input_data = transform_data_to_zero(input_data,Nspins)
                


main()
