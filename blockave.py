import numpy as np

#input text file
filein = open('3.0_lattice_2d_ising_avg_energy_MCMC+NADE.txt')

#list of rows read from file
rows=[]
for line in filein:
   rows.append(line)
#rows = np.loadtxt('energy_outputs.txt')
#number of rows read
Ndata = len(rows)

#index of the column where the relevant data is, starting from 0
inddata =0

#vector of energies
energies=np.zeros(len(rows))
for i in range(Ndata):
  energies[i]= float(rows[i].split()[inddata])

print('How large are the blocks?')
Sblock = int(input())

print('How many data you want to skip?')
Nskip = int(input())

#energy vector without initial data skept for equilibration75
enecut = np.copy(energies[Nskip:])

#number of data used, after truncation for equilibration
Ndataused = len(enecut)

#number of blocks
Nblocks = Ndataused//Sblock #note: integer division
print('Number of blocks is %d' % Nblocks)

#reshape data vector
#each block is a row of the matrix
#the data in the remainder (>Nblocks*Sblock) are not used
eblocks = enecut[:Nblocks*Sblock].reshape((Nblocks,Sblock))

#vector with averages of each block
eaveblocks = np.zeros(Nblocks)
for i in range(Nblocks):
   eaveblocks[i] = np.average(eblocks[i,:])

#average of the averages
globave = np.average(eaveblocks)
#st.dev. of the averages, devided by sqrt(Nblocks-1)
globstd = np.std(eaveblocks)/np.sqrt(Nblocks-1)

print('The global average is: %f +-  %f' % (globave,globstd))


