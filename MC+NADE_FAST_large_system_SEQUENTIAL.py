
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
    return (-Beta*Nspins*E)     #log(exp(beta*E))=beta*E

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
        
    # print(Px)
    return out_spins,np.log(Px)


def MC_NADE(Lx,Nspins,Beta,Nvmcsteps,verbose,J,J2,J3,W,V,bf,cf,ar,save_NADE = True):

  print("MCMC simulation")
   
# initialisation 
  energies_data = []
  configs_data = []
  NADE_configs = []
  NADE_energies = []
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
     
     if ivmc < 100000 and save_NADE is True:
         NADE_configs.append(S0_trial_reshaped)
         NADE_energies.append(enow_trial)
         
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
       strout="%6d %6d %7.4f %10.6f %9.5f" % (ivmcp,Nvmcsteps,enow_accepted,eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.))) +'\n'
       print(strout.strip())
       
       
  configs_data = np.array(configs_data)
  configs_data = configs_data.reshape(int((configs_data.size)/Lx**2),Lx,Lx)

  np.save(str(Nspins)+'_lattice_2d_ising_spins_NADEMCMC.npy',configs_data)
  if save_NADE is True:
      np.save('NADE_configs.npy',NADE_configs)
      np.savetxt('NADE_energies.txt', NADE_energies)
  
  strout='MCMC: %6f %10.6f %9.5f' % (Nvmcsteps,eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)))
  print(strout)
  print("end MCMC")
 
  return eave/ivmcp,np.sqrt((eave2/ivmcp-(eave/ivmcp)**2)/(ivmcp-1.)),configs_data,J,accepted,energies_data


#def main(model_path=None, num_sample=1):
Lx = 10
Nspins = Lx**2
N = Nspins
Beta = 1.0 #First beta
Nvmcsteps = 1000
verbose = True
coupling_noise = 0.1

# Training parameters #
data_path = "./"

# parameters
epochs = 100
batch_size = 300 
scheduler_step = 1
scheduler_gamma = 0.96
learning_rate = 0.01
patience = 20

check_steps = 2000 #number of MC+NADE steps to check the value of acceptance rate.
verbose   = True

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




configs_start = np.load('Dwave_configs.npy') 
np.save("train_"+str(N)+"_lattice_2d_ising_spins.npy",configs_start[:int(len(configs_start)*9/10)])
np.save("val_"+str(N)+"_lattice_2d_ising_spins.npy",configs_start[int(len(configs_start)*9/10):])

train_set = Ising2D(name="Train Ising", path=data_path + "train_"+str(N)+"_lattice_2d_ising_spins.npy")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = Ising2D(name="Valid Ising", path=data_path + "val_"+str(N)+"_lattice_2d_ising_spins.npy")
test_loader = DataLoader(test_set, batch_size=2*batch_size, shuffle=False)

'''
Prepare training and the NN
'''
now = datetime.datetime.now()
date = now.strftime("%Y%m%d_%H%M")
# check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
# load the model
model = NADE(input_dim=N, hidden_dim=N).to(device)

criterion = nn.BCELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# start main
train_losses = []
test_losses = []
best_loss = 9999
wait = 0
prog_bar = trange(epochs, leave=True, desc="Epoch")
for step in prog_bar: ########### Training over and epoch
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
        torch.save(model.state_dict(), f'./nade-{date}_beta='+str(Beta)+'.pt')
        
  
W = model.params["W"].cpu().detach().numpy().astype('float64')
V = model.params["V"].cpu().detach().numpy().astype('float64')
bf = model.params["b"].cpu().detach().numpy().astype('float64')
cf = model.params["c"].cpu().detach().numpy().astype('float64')

Beta = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5])

Nvmcsteps2 = int(1000)
Ars = []
track_ar = 0.
beta_eff = 0.0
for Beta in Beta:
    
    report = MC_NADE(Lx,Nspins,Beta,Nvmcsteps2,verbose,J,J2,J3,W,V,bf,cf,ar=1.0)
    
    report = np.array(report, dtype="object")
    acc_rate = report[4]/Nvmcsteps2*100.
    if acc_rate > track_ar: 
        track_ar=acc_rate
        beta_eff = Beta
    averaged = sum(report[5])/Nvmcsteps2
    configs_MCNADE = report[2]
    
    Ars.append(acc_rate)
    # np.save(str(N)+"_lattice_2d_ising_spins.npy",configs_MCNADE)
    # np.save("acceptance_rate.npy",np.array(acc_rate))
    
    # np.savetxt(str(Beta)+'_lattice_2d_ising_avg_energy_MCMC+NADE.txt',report[5])
    # np.save(str(Beta)+'_MCMC+NADE_report.npy',report)

Ars = np.array(Ars)
np.savetxt('ar.txt',Ars)

'''Run MC+NADE at Beff to get new data set'''
#%%
no_steps = int(100000* (100./track_ar))
report = MC_NADE(Lx,Nspins,beta_eff,no_steps,verbose,J,J2,J3,W,V,bf,cf,ar=(track_ar/100))
configs_MCNADE = report[2]
np.save(str(N)+"_lattice_2d_ising_spins_at_beta="+str(beta_eff)+".npy",configs_MCNADE)

seq_acc_rates = []
BETAS = np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0])+beta_eff

for i in range(len(BETAS)-1):

    '''Train NADE on new data set at Beff'''
    
    configs_start = np.load(str(N)+"_lattice_2d_ising_spins_at_beta="+str(BETAS[i])+".npy") 
    np.save("train_"+str(N)+"_lattice_2d_ising_spins.npy",configs_start[:int(len(configs_start)*9/10)])
    np.save("val_"+str(N)+"_lattice_2d_ising_spins.npy",configs_start[int(len(configs_start)*9/10):])
    
    train_set = Ising2D(name="Train Ising", path=data_path + "train_"+str(N)+"_lattice_2d_ising_spins.npy")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = Ising2D(name="Valid Ising", path=data_path + "val_"+str(N)+"_lattice_2d_ising_spins.npy")
    test_loader = DataLoader(test_set, batch_size=2*batch_size, shuffle=False)
    
    '''
    Prepare training and the NN
    '''
    now = datetime.datetime.now()
    date = now.strftime("%Y%m%d_%H%M")
    # check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load the model
    model = NADE(input_dim=N, hidden_dim=N).to(device)
    
    criterion = nn.BCELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # start main
    train_losses = []
    test_losses = []
    best_loss = 9999
    wait = 0
    prog_bar = trange(epochs, leave=True, desc="Epoch")
    for step in prog_bar: ########### Training over and epoch
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
            torch.save(model.state_dict(), f'./final_nade_at_beta='+str(BETAS[i])+'_after_MC+NADE.pt')
            
            
    #%%    
    model.load_state_dict(torch.load('final_nade_at_beta='+str(BETAS[i])+'_after_MC+NADE.pt'))    
    W2 = model.params["W"].cpu().detach().numpy().astype('float64')
    V2 = model.params["V"].cpu().detach().numpy().astype('float64')
    bf2 = model.params["b"].cpu().detach().numpy().astype('float64')
    cf2 = model.params["c"].cpu().detach().numpy().astype('float64')
    
    Nvmcsteps2_end = int(100000* (100./track_ar))
    
    
    report_end = MC_NADE(Lx,Nspins,BETAS[i+1],Nvmcsteps2_end,verbose,J,J2,J3,W2,V2,bf2,cf2,ar=(track_ar/100))
        
    report_end = np.array(report_end, dtype="object")
    track_ar = report_end[4]/Nvmcsteps2_end*100.
    seq_acc_rates.append(track_ar)
    averaged_end = sum(report_end[5])/Nvmcsteps2_end
    configs_MCNADE_end = report_end[2]
    np.save(str(N)+"_lattice_2d_ising_spins_at_beta="+str(BETAS[i+1])+".npy",configs_MCNADE_end)
    
seq_acc_rates = np.array(seq_acc_rates)
np.savetxt('sequential_acc_rates.txt',seq_acc_rates)