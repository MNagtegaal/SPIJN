# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:17:17 2019

@author: Martijn Nagtegaal
@email: m.a.nagtegaal@tudelft.nl

Code to show how to use the SPIJN algorithm with a pre-computed dictionary and make 
a comparison to the NNLS voxel-wise approach.
"""
import numpy as np
from SPIJN import SPIJN,lsqnonneg
import matplotlib.pyplot as plt
import matplotlib as mpl
#%% Function for noise generation
def add_noise(I,SNR):
    if SNR is not None:
        for k in range(I.shape[1]):
            for j in range(I.shape[2]):
                i = I[:,k,j]        
                I_max = np.abs(i).max()
                i+=np.random.normal(0,I_max/SNR,i.shape)
#%% Load data
T1_phantom = np.asarray([67,1000,2000])
T2_phantom = np.asarray([13,100,500])

data = np.load('data.npz')

I = data['I'] #Simulated image 
T1_list = data['T1'] 
T2_list = data['T2']
D = data['D'] # Dictionary
D_norm_factors = data['D_norm_factors']
Dc = data['Dc'] # SVD compressed dictionary
Compr = data['Compr'] # Compression matrix
phantom_truth = data['phantom_truth'] #The 3 simulated components

#%% Add noise, compress and reshape data
add_noise(I,50)

Ic = np.einsum('ij,ikl->jkl',Compr,I) # Compress image
Ic_resh = Ic.reshape([Ic.shape[0],-1]) # reshape image to shape M,J
#%% Calculations
C_solutions = {}
C_solutions['NNLS'],_ = lsqnonneg(Ic_resh,Dc.T) # Perform NNLS solve

Cr,Sfull,rel_err,C1,C = SPIJN(Ic_resh,Dc.T,L=.0001,correct_im_size=True) # Too low regularization parameter

Cr,Sfull,rel_err,C1,C = SPIJN(Ic_resh,Dc.T,L=.1,correct_im_size=True,C1 = C1,num_comp=25,max_iter=20,tol=1e-10) #Example to reuse the first iteration
Cr /= D_norm_factors[Sfull]

C_solutions['SPIJN'] = C

for key,C in C_solutions.items():
    C_solutions[key] = np.einsum('i,ij->ij',1/D_norm_factors,C) # Correct match for normalisation of dictionary


#%% Reshape data to image shape
Cr = Cr.reshape(Ic.shape[1:]+(Cr.shape[1],))
Sfull = Sfull.reshape(Ic.shape[1:]+(Sfull.shape[1],))
for key,C in C_solutions.items():
    C_solutions[key] = C.reshape((C.shape[0],)+Ic.shape[1:])
#%% Group data for NNLS and SPIJN solutions
names = ['Myelin water','Intra-, extra-\n cellular water','Free water']
T1mins = [0,200,850]
T1maxs = [200,1800,1e5]
T2mins = [0,30,200]
T2maxs = [40,200,1e5]
bd = []

C_groups = {}
for key,C in C_solutions.items():
    C_groups[key] = np.empty((3,)+C.shape[1:])
    
for k,(T1min,T1max,T2min,T2max) in enumerate(zip(T1mins,T1maxs,T2mins,T2mins)):
    bd = np.logical_and(np.logical_and(T1min<T1_list,T1_list<=T1max), np.logical_and(T2min<T2_list,T2_list<=T1max)) 
    for key,C in C_solutions.items():

        C_groups[key][k] = C[bd].sum(axis=0)
        
#%% Plot SPIJN solution
S = np.unique(Sfull[Cr>.01]) # Determine the used indices
num_comp = len(S)

fig,axs = plt.subplots(1,num_comp,figsize=(10,3))
vmaxs = np.ones(num_comp)
vmaxs[0]=.2
for k,s in enumerate(S):
    ax = axs[k]
    ims = ax.imshow(C_solutions['SPIJN'][s],vmin=0,vmax=vmaxs[k])
    plt.colorbar(ims,ax=ax)
    ax.set_title('$T_1={:.1f}, T_2={:.1f}$'.format(T1_list[s],T2_list[s]))
    ax.axis('off')
plt.tight_layout()
#%% Plot grouped solution
fig,axs = plt.subplots(3,2,figsize=(7,5))
vmaxs = np.ones(3)
vmaxs[0]=.2
for i,(key,C_group) in enumerate(C_groups.items()):
    for k in range(3):
        ax = axs[k,i]
        ims = ax.imshow(C_group[k],vmin=0,vmax=vmaxs[k])
        plt.colorbar(ims,ax=ax)
    #    ax.set_title('$T_1={:.1f}, T_2={:.1f}$'.format(T1_list[s],T2_list[s]))
        ax.axis('off')
        if k==0:
            ax.set_title(key)
for k,ax in enumerate(axs[:,0]): #Row label
    row = names[k]
    row+='\n${:.0f}$ ms$<T_1$'.format(T1mins[k])
    if T1maxs[k]<9e4: row+= '$<{:.0f}$ ms'.format(T1maxs[k])
    row += '\n${:.0f}$ ms$<T_2$'.format(T2mins[k])
    if T2maxs[k]<9e4:
        row+= '$<{:.0f}$ ms'.format(T2maxs[k])
    ax.annotate(row, xy=(0, .5), xytext=(0, .5),xycoords ='axes fraction',
                size='large', horizontalalignment='right', va='center',fontsize=9)

#%% Scatter plots
cmap = plt.get_cmap('jet')
colors = [cmap(i) for i in np.linspace(0, 1, len(T1mins))]
fig,axs = plt.subplots(1,2,sharex=True,sharey=True)
for k,(key,C) in enumerate(C_solutions.items()):
    axs[k].scatter(T1_list,T2_list,s = C.sum(axis=(1,2)))

    axs[k].set_ylabel('$T_2$')
    axs[k].set_xlabel('$T_1$')
    axs[k].set_title(key)
    axs[k].scatter(T1_phantom,T2_phantom,c='k',marker='x')
    for j in range(len(T1mins)):
            axs[k].add_patch(mpl.patches.Rectangle((T1mins[j], T2mins[j] ), T1maxs[j]-T1mins[j],
                 T2maxs[j]-T2mins[j], fill=False, alpha=1,color=colors[j]))
axs[0].set_xscale("log", nonposx='clip')
axs[0].set_yscale("log", nonposy='clip')
axs[0].set_ylim(5,2.5e3)
axs[0].set_xlim(5,5e3)
