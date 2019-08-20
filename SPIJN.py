# -*- coding: utf-8 -*-
"""

@author: Martijn Nagtegaal
@email: m.a.nagtegaal@tudelft.nl
 
Code to use the Sparsity Promoting Iterative Joint NNLS (SPIJN) algorithm, designed for multi-component MRF.
When using this code. Please refer to the original paper

Nagtegaal, Martijn, Peter Koken, Thomas Amthor, and Mariya Doneva. “Fast Multi-Component Analysis Using a Joint Sparsity Constraint for MR Fingerprinting.” Magnetic Resonance in Medicine https://doi.org/10.1002/mrm.27947.

"""
import numpy as np
import time
import scipy as sp
from scipy import optimize
import signal
#import multiprocessing
def nnls(x,A):
    return sp.optimize.nnls(A,x)[0]

#@profile
def lsqnonneg(Y,red_dict,out_z = None,out_rel_err = None,return_r = False,S=None):
    """
    Function to perform the joint NNLS solve, where no regularisation is applied.
    
    """
    red_dict = red_dict.T
    if S is not None:
        diclen = len(S)
    else:
        diclen = red_dict.shape[0]
    if out_z is None:
        out_z = np.empty((diclen,Y.shape[1]))
#        out_z_2 = np.empty((diclen,Y.shape[1]))
    if out_rel_err is None:
        out_rel_err = np.empty(Y.shape[1])

    R = red_dict
    if S is not None:
        R = R[S]
    if Y.shape[1]>1e5:
        for k in range(int(Y.shape[1]/10000)+1): 
            st = k*10000
            end = np.min([(k+1)*10000,Y.shape[1]])
            sl = slice(st,end)
            Ysub = Y[:,sl]
            out_z[:,sl] = np.apply_along_axis(lambda x:sp.optimize.nnls(R.T,x)[0],1,Ysub.T).T
    else:
        out_z = np.apply_along_axis(lambda x:sp.optimize.nnls(R.T,x)[0],0,Y)
            
    Ycalc =  out_z.T @ R    
    r = Y - Ycalc.transpose()
    out_rel_err = np.linalg.norm(r,axis=0)/np.linalg.norm(Y,axis=0)
#    efficiency()
    if not return_r:
        return out_z, out_rel_err
    else:
        return out_z, out_rel_err,r

def rewlsqnonneg(Y,red_dict,w,out_z = None,out_rel_err = None,return_r = False,L=0,S=None):
    """
    Performs a reweighted NN least squares signal-wise
    """
    
    red_dict = red_dict.T

#    clear_pseudo_hist()
    if S is not None:
        diclen = len(S)
    else:
        diclen = red_dict.shape[0]
        
    if out_z is None:
        out_z = np.empty((diclen,Y.shape[1]))
    if out_rel_err is None:
        out_rel_err = np.empty(Y.shape[1])
        
#    W = scsp.diags(w**.5) 
    Y1 = np.vstack((Y,np.zeros(Y.shape[1])))    
    
    R = red_dict
    if S is not None:
        R = R[S]
    R = R.T*w**.5

    R = np.vstack((R,L*np.ones(R.shape[1])))
#        out_z = np.apply_along_axis(lambda x:sp.optimize.nnls(R,x)[0],0,Y1)
    if Y.shape[1]>1e5:
        for k in range(int(Y.shape[1]/10000)+1): 
            st = k*10000
            end = np.min([(k+1)*10000,Y.shape[1]])
            sl = slice(st,end)
            Ysub = Y1[:,sl]
            out_z[:,sl] = np.apply_along_axis(lambda x:sp.optimize.nnls(R,x)[0],0,Ysub)
    else:
        out_z = np.apply_along_axis(lambda x:sp.optimize.nnls(R,x)[0],0,Y1)
    out_z = (out_z.T*w**.5).T
    if S is not None:
        R1 = red_dict[S]
    else:
        R1 = red_dict
    Ycalc = out_z.T @ R1
    r = Y - Ycalc.transpose()
    out_rel_err = np.linalg.norm(r,axis=0)/np.linalg.norm(Y,axis=0)
    if not return_r:
        return out_z, out_rel_err
    else:
        return out_z, out_rel_err,r

def SPIJN(Y,red_dict,num_comp=10,p=0,max_iter=20,
              verbose = True,norm_sig = True,tol=1e-4,L=0,correct_im_size = False,prun = 2,
              C1=None):
    """Perform the SPIJN algorithm
    INPUT:
    - Y: Signals with shape M,J (M is the signal length, J the number of signals)
    - red_dict: Dictionary with shape M,N where N is the number of components
    - num_comp: number of components returned
    - p: value as used in the reweighting scheme 1. Default 0 works fine
    - max_iter: maximum number of iterations, 20 is normally fine
    - verbose: more output
    - norm_sig: Normalise the signal or not. False is fine
    - tol: Tolerance used, when are the iterations stopped, based on 
        ||C^k-C^{k+1}||_F^2/||C^k||_F^2<tol
    - L: the regularisation parameter used. 
    - correct_im_size: Experiments showed that the regularisation parameter
            can be log-scaled with the number of voxels to be less sensitive to the 
            number of voxels. When correct_im_size is True, this correction is performed.
    - prun: if False no pruning, otherwise the number of iterations afterwards 
            the pruning of unused atoms in the dictionary takes place
    - C1: When C1 is provided, the first Joint NNLS solve can be skipped. 
            This can be usefull when experiments are performed with regard to the
            regularization parameter.
    OUTPUT:
    - Cr : Weigths for the different components, ordered on weight, indices are in Sfull. Length is based on num_comp
    - Sfull : The indices of the weights as given in Cr
    - rel_err : Relative error
    - C1: The raw output of the first iteration
    - C: The full mixing matrix as defined as goal in the original paper.
    """
    signal.signal(signal.SIGINT, signal.default_int_handler) #To stop during the iterations
    M,J = Y.shape
    N = red_dict.shape[1]
    eps = 1e-4
    if correct_im_size: # Correct regularisation for number of voxels
        L = L*np.log10(J)
    if norm_sig: # Normalise signals
        norm_Y = np.linalg.norm(Y,ord = 2,axis = 0)
        Y = Y/norm_Y
    w = np.ones(N) # Initialise weights
    t0 = time.clock()
    if C1 is None: # First iteration
        C1,r = lsqnonneg(Y,red_dict)
        print('matching time it 1: {0:.5f}s'.format(time.clock() - t0))
    else:
        print('Reused old first iteration solution')
    C = C1
    if prun == True:
        prun = 1
    
    k=0

    N1 = N
    S = None
    try: # Try-except is to stop iterations when it takes to long, making it possible to return the latest result
        for k in range(1,max_iter):
            if prun == k: # Pruning 
                prunn_comp =  np.sum(C,1)/J>1e-15 #Determine which components are small

                S = np.arange(len(prunn_comp))[prunn_comp] # Determine indices of large components
                C = C[prunn_comp] # Determine solution corresponding to pruned dict
                N1 = sum(prunn_comp) # Size of pruned dict
                if verbose: print('Prunned percentage {},rest:{}.'.format(100-N1/N*100,N1))
                w = w[prunn_comp]

            w = (np.linalg.norm(C,2,axis=1)+eps)**(1-p/2)
            
            w[w<eps] = eps # prevent 0-weighting
            C0 = C.copy()
            t0 = time.clock()
            C,rel_err,r = rewlsqnonneg(Y,red_dict,w,out_z=C0,L=L,return_r = True,S=S) # Perform real calculations
            if verbose: print('matching time: {0:.5f}s'.format(time.clock() - t0))
            rel = np.linalg.norm(C-C0,ord='fro')/np.linalg.norm(C,ord='fro') #Determine relative convergence
            if verbose:
                print('k: {} relative difference between iterations {},elements: {}'.format(k,rel,np.sum(np.sum(C,1)>1e-4)))
            if rel<tol or np.isnan(rel):# or np.sum(np.sum(C,1)>1e-4)<num_comp:
                if np.isnan(rel):
                    C = C0
                if verbose:
                    print('Stopped after iteration {}'.format(k))
                break
            
    except KeyboardInterrupt:
        C = C0
    if prun and k>=prun:
        w0 = w
        w = np.zeros(N)
        w[prunn_comp] = w0
        C0 = C
        C = np.zeros((N,J))
        C[prunn_comp] = C0
        
    Ycalc = (C.T @ red_dict.T ).T
    r = Y - Ycalc # Calculate residual

    if norm_sig:
        Y = Y*norm_Y
        C = C*norm_Y

    Sfull = np.argsort(-C,axis=0)[:num_comp].T
    Cr = np.empty_like(Sfull,dtype = float)
    for k,(s,Cc) in enumerate(zip(Sfull,C.T)): 
        Cr[k] = Cc[s]
        
    # Components, indices, relative error, used weights and result of first iteration are returned.
    rel_err = np.linalg.norm(r,axis=0)/np.linalg.norm(Y,axis=0)
    return Cr,Sfull,rel_err,C1,C
