# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 14:20:17 2021

@author: Benjamin
"""

from typing import Dict
from pathlib import Path
import argparse
import datetime
import math
from numba import jit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn import Module
from torch.optim import Optimizer

from tqdm import trange, tqdm





###############################################################
class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8)))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = "relu",
        num_masks: int = 1,
        natural_ordering: bool = True,
        seed: int = 42,
    ):
        """Masked Autoencoder for Distribution Estimation (MADE).
        From original article http://proceedings.mlr.press/v37/germain15.pdf

        Args:
            input_size (int): Number of inputs.
            hidden_size (int): Number of units in hidden layers.
            activation (str, optional): Activation function to be used. Defaults to "relu".
            num_masks (int, optional): Can be used to train ensemble over orderings/connections. Defaults to 1.
            natural_ordering (bool, optional): If False, use random permutations in the input layer. Defaults to True.
            seed (int, optional): Seed to create masks. Defaults to 42.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # define a simple MLP neural net
        layers = nn.ModuleList()

        hs = [input_size] + hidden_size + [input_size]
        for h0, h1 in zip(hs, hs[1:]):
            if h1 == input_size:
                continue
            layers.extend([MaskedLinear(h0, h1), self._get_activation_func(activation)])
        # last layer has no activation function
        layers.append(MaskedLinear(h0, h1))

        self.net = nn.Sequential(*layers)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = seed  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def _get_activation_func(self, activation: str) -> nn.modules.activation:
        """Returns the requested activation function.

        Args:
            activation (str): Name of the activation function.

        Returns:
            nn.modules.activation: The chosen activation function.
        """
        if activation == "relu":
            function = nn.ReLU()
        elif activation == "prelu":
            function = nn.PReLU()
        elif activation == "rrelu":
            function = nn.RReLU()
        elif activation == "leakyrelu":
            function = nn.LeakyReLU()
        elif activation == "gelu":
            function = nn.GELU()
        elif activation == "selu":
            function = nn.SELU()
        else:
            raise NotImplementedError("Activation function not implemented")
        return function

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return  # only a single seed, skip for efficiency
        L = len(self.hidden_size)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs
        self.m[-1] = (
            np.arange(self.input_size)
            if self.natural_ordering
            else rng.permutation(self.input_size)
        )
        # sample the connectivity of all neurons
        for l in range(L):
            self.m[l] = rng.randint(
                self.m[l - 1].min(), self.input_size - 1, size=self.hidden_size[l]
            )

        # construct the mask matrices for hidden layer
        masks = [self.m[l][:, None] >= self.m[l - 1][None, :] for l in range(L)]
        # construct the mask for the last layer
        masks.append(self.m[-1][:, None] > self.m[L - 1][None, :])

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)

    def _compute_prob(self, x, x_hat):
        # BCEWithLogitsLoss with reduction='none' is nothing than
        # the positive log-likelihood of the sample
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        log_prob = -criterion(x_hat, x)
        return log_prob.sum(dim=-1)

    def sample(self, n, device) -> Dict[np.ndarray, np.ndarray]:
        # initialize the samples
        x = torch.zeros((n, self.input_size), device=device,)

        prog_bar = trange(self.input_size, leave=True)
        for d in prog_bar:
            # compute x_hat sequentally
            x_hat = self.forward(x)
            # generate x_d according to the conditional prob x_hat
            sigm = nn.Sigmoid()
            x[:, d] = torch.bernoulli(sigm(x_hat[:, d])).detach()
        log_prob = self._compute_prob(x, x_hat)

        # reshape output and set to {-1,+1}
        L = int(math.sqrt(self.input_size))
        x = x.view((-1, L, L)) * 2 - 1
        x = x.detach().numpy()
        return {
            "sample": np.reshape(x,  (-1,L,L), order='F'),
            "log_prob": log_prob.detach().numpy(),
        }
    
######################################################








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
    
    



#######################################################################





def train_step(
    data_loader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    model: Module,
    resample_every: float,
    device: str,
) -> float:
    """Train the network

    Args:
        data_loader (DataLoader): Dataloader.
        criterion (Module): Criterion to compute the loss.
        optimizer (Optimizer): Optimizer.
        model (Module): Model used.
        resample_every (int): Step after resample masks.
        device (str): 'cpu' or 'cuda'. 

    Returns:
        float: Loss over the epoch.
    """
    model.train()
    epoch_loss = 0
    prog_bar = tqdm(data_loader, leave=False)
    for count, imgs in enumerate(prog_bar):
        inputs = imgs.view(imgs.size(0), -1).gt(0.0).float().to(device)
        logits = model(inputs)
        loss = criterion(logits, inputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Connectivity agnostic and order agnostic
        if count + 1 % resample_every == 0:
            model.update_masks()

        # save losses
        epoch_loss += loss.item()

        # update bar description
        prog_bar.set_description(f"Train Loss {loss.item():.4f}", refresh=True)
    return epoch_loss / (count + 1)


def validation_step(
    data_loader: DataLoader, criterion: Module, model: Module, device: str,
) -> float:
    """Validation Step

    Args:
        data_loader (DataLoader): Dataloader.
        criterion (Module): Criterion to compute the loss.
        model (Module): Model used.
        device (str): 'cpu' or 'cuda'. 

    Returns:
        float: Loss over the epoch.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for count, imgs in enumerate(data_loader):
            # preprocess to binary
            inputs = imgs.view(imgs.size(0), -1).gt(0.0).float().to(device)
            logits = model(inputs)
            loss = criterion(logits, inputs)

            # Connectivity agnostic (update each step)
            model.update_masks()

            # save losses
            val_loss += loss.item()
    return val_loss / (count + 1)

def sample(
    model: nn.Module, n: int = 1, device: str = "cpu"
) -> Dict[np.ndarray, np.ndarray]:
    """Sample n samples using the pretrained model.

    Args:
        model (nn.Module): Pretrained model.
        n (int, optional): Number of samples to generate. Defaults to 1.
        device (str, optional): Device used, cuda or cpu. Defaults to cpu.
    """
    model.eval()
    out = model.sample(n, device)

    save_path = Path(".").absolute()
    size = out["sample"].shape[-1]
    save_name = "size-" + str(size) + "_sample-" + str(n) + "_MADE"

    #print("\nSaving sample generated by MADE as", save_name)
    np.savez(save_path / save_name, **out)
    return out


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

@jit(nopython=True)
def Prob(E,Beta,Nspins):        # Boltzmann probability distribution
    return (-Beta*Nspins*E)     #log(exp(beta*E))=beta*E

def MC_MADE(Lx,Nspins,Beta,Nvmcsteps,verbose,J,J2,J3,samples,log_probs,ar,save_MADE = True):

  print("MCMC simulation")
   
# initialisation 
  energies_data = []
  configs_data = []
  MADE_configs = []
  MADE_energies = []
  eave = 0.
  eave2 = 0.
  accepted = 0.
  
# Generate the first instance
  S0_accepted = samples[0]
  Px_accepted = log_probs[0]
  
            
  enow_accepted = econf(Lx,J,J2,J3,S0_accepted)    # Calculate the energy.
  
# perform the Monte Carlo steps 
  for ivmc in range(1,Nvmcsteps):
           
     S0_trial = samples[ivmc]
     Px_trial = log_probs[ivmc] # Generate the trial configuration
            
     
            
     enow_trial = econf(Lx,J,J2,J3,S0_trial)    # Calculate the energy.
     
     if ivmc < 100000 and save_MADE is True:
         MADE_configs.append(S0_trial)
         MADE_energies.append(enow_trial)
         
     '''Here we make the acceptance criterion'''
     Ptrial    = Prob((enow_trial),Beta,Nspins)  
     # print(Ptrial)      
     Paccept   = Prob((enow_accepted),Beta,Nspins) 
     # qxx = ((Px_accepted/Px_trial) * (Ptrial/Paccept))  
     qxx = np.exp(Px_accepted-Px_trial+Ptrial-Paccept)      #log form of acceptance
     # print(qxx)
     Axx = np.amin([1.,qxx]) 
     # print(qxx)

    
     '''Do we accept or reject'''    
     if Axx >= 1.0:         
         enow_accepted  = np.copy(enow_trial)   # Update energy
         Px_accepted    = np.copy(Px_trial)     # Update probability from MADE
         S0_accepted    = np.copy(S0_trial)     # Accept and update the new spin configuration
         accepted       += 1                    # Count new accepted configs

     elif np.random.ranf() < Axx :          
         enow_accepted  = np.copy(enow_trial)   # Update energy
         Px_accepted    = np.copy(Px_trial)     # Update probability from MADE
         S0_accepted    = np.copy(S0_trial)     # Accept and update the new spin configuration
         accepted       += 1                    # Count new accepted configs
         
     
     eave  += enow_accepted                     # Accumlate average
     eave2 += enow_accepted**2                  # Accumlate average squared
     ivmcp=ivmc+1                               # Count steps
     
     if ivmc%int(1/ar)==0:
         energies_data.append(enow_accepted)        # Record energies at MC step
         configs_data.append(S0_accepted)         # Record configsurations at MC step.

     
     if verbose and ivmcp > 1:
       strout="%6d %6d %7.4f %10.6f %9.5f" % (ivmcp,Nvmcsteps,enow_accepted,eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.))) +'\n'
       print(strout.strip())
       
       
  configs_data = np.array(configs_data)
  configs_data = configs_data.reshape(int((configs_data.size)/Lx**2),Lx,Lx)

  np.save(str(Nspins)+'_lattice_2d_ising_spins_MADEMCMC.npy',configs_data)
  if save_MADE is True:
      np.save('MADE_configs.npy',MADE_configs)
      np.savetxt('MADE_energies.txt', MADE_energies)
  
  strout='MCMC: %6f %10.6f %9.5f' % (Nvmcsteps,eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)))
  print(strout)
  print("end MCMC")
 
  return eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)),configs_data,J,accepted,energies_data



# model
Lx = 22
Nspins = Lx*Lx
N=Nspins
size = Nspins # input size
hiddens = [2500] # list containing the size of the hidden layers
num_masks = 1 # number of different masks used in training
resample_every = 10 # after how many step change masks

# optimizer
batch_size = 500 
epochs = 40
patience = 2

# Training parameters #
data_path = "./"

########### COUPLINGS ################
np.random.seed(12345)
J = (np.random.normal(0.0,1.0,size=(N-Lx,2)))
J2 = (np.random.normal(0.0,1.0,size=((Lx-1)**2,2)))
J3 = (np.random.normal(0.0,1.0,size=(((Lx-2)*Lx),2)))

#J = J + (np.random.normal(0.0,coupling_noise,size=(N-Lx,2)))
#J2 = J2 + (np.random.normal(0.0,coupling_noise,size=((Lx-1)**2,2)))
#J3 = J3 + (np.random.normal(0.0,coupling_noise,size=(((Lx-2)*Lx),2)))

np.savetxt('Random_couplings_noise.txt',J)
np.savetxt('Random_couplings2_noise.txt',J2)
np.savetxt('Random_couplings3_noise.txt',J3)

## vertical and horizontal neighbours ##
Jh = J[:,0]
Jv = J[:,1]

Jv = np.concatenate([Jv, np.zeros(Lx)])
for i in range(Lx**2):
    if (i+1)%Lx ==0: Jh = np.insert(Jh,i,0.0)
    
J = np.zeros(((N,2)))
J[:,0] = Jh
J[:,1] = Jv

## Diagonal neighbours ##
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

## next nearest neighbours ##
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

J_noise = J
J2_noise = J2
J3_noise = J3

np.random.seed(12345)
J = (np.random.normal(0.0,1.0,size=(N-Lx,2)))
J2 = (np.random.normal(0.0,1.0,size=((Lx-1)**2,2)))
J3 = (np.random.normal(0.0,1.0,size=(((Lx-2)*Lx),2)))


np.savetxt('Random_couplings.txt',J)
np.savetxt('Random_couplings2.txt',J2)
np.savetxt('Random_couplings3.txt',J3)

## vertical and horizontal neighbours ##
Jh = J[:,0]
Jv = J[:,1]

Jv = np.concatenate([Jv, np.zeros(Lx)])
for i in range(Lx**2):
    if (i+1)%Lx ==0: Jh = np.insert(Jh,i,0.0)
    
J = np.zeros(((N,2)))
J[:,0] = Jh
J[:,1] = Jv

## Diagonal neighbours ##
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

## next nearest neighbours ##
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


######################## END of COUPLINGS ################
'''Start training of MADE for the first step in the process'''

data = np.load('start_484_lattice_2d_ising_spins_at_beta=1.0.npy')
np.save('train_484_lattice_2d_ising_spins.npy',data[:290000])
np.save('val_484_lattice_2d_ising_spins.npy',data[290000:300000])
train_path = "./train_484_lattice_2d_ising_spins.npy"
val_path = "./val_484_lattice_2d_ising_spins.npy"


# reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# get the current date to save the model
now = datetime.datetime.now()
date = now.strftime("%Y%m%d-%H%M")

# check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# construct model and ship to GPU
model = MADE(size, hiddens, num_masks=num_masks)
print(
    "Number of model parameters:",
    sum([np.prod(p.size()) for p in model.parameters()]),
)
print(model)
model.to(device)

# load the dataset
print("\n\nLoading dataset from", train_path)
train_set = Ising2D(name="train_ising", path=train_path)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

print(f"Loading validation dataset from {val_path}\n\n")
validation_set = Ising2D(name="val_ising", path=val_path)
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

# set up the optimizer and the scheduler
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs,
)

# set the criterion
criterion = nn.BCEWithLogitsLoss(reduction="mean")

# start the training
prog_bar = trange(epochs, leave=True, desc="Epoch")
best_loss = np.inf
no_decreasing = 0
for epoch in prog_bar:
    train_loss = train_step(
        train_loader, criterion, optimizer, model, resample_every, device
    )
    val_loss = validation_step(val_loader, criterion, model, device)
    prog_bar.set_postfix(val_loss=f"{val_loss:.4f}", train_loss=f"{train_loss:.4f}")
    # save best model
    if val_loss < best_loss:
        # early stopping (with tollerance of 1e-4)
        if val_loss >= best_loss - 1e-4:
            no_decreasing += 1
        else:
            no_decreasing = 0
        best_loss = val_loss
        torch.save(
            model.state_dict(), f"./made{hiddens[0]}-{date}.pt",
        )
    # elif val_loss > best_loss:
    #     print(f"Start overfitting val_loss:{val_loss} (best {best_loss})")
    #     break
    if no_decreasing > patience:
        print(f"Early Stopping after {patience} no-increasing epochs")
        break

    scheduler.step()
    
    
#%%
Beta = np.array([1.0])
Nvmcsteps2 = int(50000)
Ars = []
track_ar = 0.
beta_eff = 0.0

sampled_data = sample(model, n=Nvmcsteps2, device=device)
samples = sampled_data['sample']
log_probs = sampled_data['log_prob']



for Beta in Beta:
    
    report = MC_MADE(Lx,Nspins,Beta,Nvmcsteps2,True,J,J2,J3,samples,log_probs,ar=1.0)
    
    report = np.array(report, dtype="object")
    acc_rate = report[4]/Nvmcsteps2*100.
    if acc_rate > track_ar: 
        track_ar=acc_rate
        beta_eff = Beta
    averaged = sum(report[5])/Nvmcsteps2
    configs_MCNADE = report[2]
    
    Ars.append(acc_rate)

Ars = np.array(Ars)
np.savetxt('ar.txt',Ars)  

'''Run MC+NADE at Beff to get new data set'''

no_steps = int(100000* (100./track_ar))
samples = []
log_probs = []
for i in range(int(no_steps/10000)):

    sampled_data = sample(model, n=10000, device=device)
    samples_dum = sampled_data['sample']
    log_probs_dum = sampled_data['log_prob']
    
    samples.append(samples_dum)
    log_probs.append(log_probs_dum)

samples = np.array(samples)
samples = np.array(samples).reshape(((i+1)*10000),Lx,Lx)
log_probs = np.array(log_probs)
log_probs = np.array(log_probs).reshape(len(samples))

report = MC_MADE(Lx,Nspins,Beta,len(samples),True,J,J2,J3,samples,log_probs,ar=(track_ar/100))
configs_MCNADE = report[2]
np.save(str(beta_eff)+".npy",configs_MCNADE)

seq_acc_rates = []
betas = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.4,1.8,2.0])+beta_eff

for i in range(len(betas)-1):

    '''Train NADE on new data set at Beff'''
    
    configs_start = configs_MCNADE
    np.save(str(betas[i])+".npy",configs_start) 
    np.save("train_"+str(N)+"_lattice_2d_ising_spins.npy",configs_start[:int(len(configs_start)*9/10)])
    np.save("val_"+str(N)+"_lattice_2d_ising_spins.npy",configs_start[int(len(configs_start)*9/10):])
    
    train_set = Ising2D(name="Train Ising", path=data_path + "train_"+str(N)+"_lattice_2d_ising_spins.npy")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = Ising2D(name="Valid Ising", path=data_path + "val_"+str(N)+"_lattice_2d_ising_spins.npy")
    test_loader = DataLoader(test_set, batch_size=2*batch_size, shuffle=False)
    
    '''
    Prepare training and the NN
    '''
    
    
    # get the current date to save the model
    now = datetime.datetime.now()
    date = now.strftime("%Y%m%d-%H%M")
    
    # check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # construct model and ship to GPU
    #model = MADE(size, hiddens, num_masks=num_masks)
    print(
        "Number of model parameters:",
        sum([np.prod(p.size()) for p in model.parameters()]),
    )
    print(model)
    model.to(device)
    
    # load the dataset
    print("\n\nLoading dataset from", train_path)
    train_set = Ising2D(name="train_ising", path=train_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    print(f"Loading validation dataset from {val_path}\n\n")
    validation_set = Ising2D(name="val_ising", path=val_path)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    
    # set up the optimizer and the scheduler
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs,
    )
    
    # set the criterion
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    
    # start the training
    prog_bar = trange(epochs, leave=True, desc="Epoch")
    best_loss = np.inf
    no_decreasing = 0
    for epoch in prog_bar:
        train_loss = train_step(
            train_loader, criterion, optimizer, model, resample_every, device
        )
        val_loss = validation_step(val_loader, criterion, model, device)
        prog_bar.set_postfix(val_loss=f"{val_loss:.4f}", train_loss=f"{train_loss:.4f}")
        # save best model
        if val_loss < best_loss:
            # early stopping (with tollerance of 1e-4)
            if val_loss >= best_loss - 1e-4:
                no_decreasing += 1
            else:
                no_decreasing = 0
            best_loss = val_loss
            torch.save(
                model.state_dict(), f"./made{hiddens[0]}-{date}.pt",
            )
        # elif val_loss > best_loss:
        #     print(f"Start overfitting val_loss:{val_loss} (best {best_loss})")
        #     break
        if no_decreasing > patience:
            print(f"Early Stopping after {patience} no-increasing epochs")
            break
    
        scheduler.step()
        
        

    Nvmcsteps2_end = int(100000* (100./track_ar))
    
    samples = []
    log_probs = []
    for i in range(int(Nvmcsteps2_end/10000)):
        
        sampled_data = sample(model, n=10000, device=device)
        samples_dum = sampled_data['sample']
        log_probs_dum = sampled_data['log_prob']
        
        samples.append(samples_dum)
        log_probs.append(log_probs_dum)

    samples = np.array(samples)
    samples = np.array(samples).reshape(((i+1)*10000),Lx,Lx)
    log_probs = np.array(log_probs)
    log_probs = np.array(log_probs).reshape(len(samples))

    report_end = MC_MADE(Lx,Nspins,Beta,len(samples),True,J,J2,J3,samples,log_probs,ar=(track_ar/100))
        
    report_end = np.array(report_end, dtype="object")
    track_ar = report_end[4]/len(samples)*100.
    seq_acc_rates.append(track_ar)
    averaged_end = sum(report_end[5])/Nvmcsteps2_end
    configs_MCNADE = report_end[2]
    
seq_acc_rates = np.array(seq_acc_rates)
np.savetxt('sequential_acc_rates.txt',seq_acc_rates)
        
        
    
    
    
      