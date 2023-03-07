import numpy as np
import torch
import torch.nn as nn

#Embedding for transformation of spiketimes to spiketrains
neurons_in=10
time_steps=30
device = 'cuda' if torch.cuda.is_available() else 'cpu'
one_hot_encoder = nn.Embedding(time_steps,time_steps).to(device)
one_hot_encoder.weight.data = torch.eye(time_steps,device=device)
one_hot_encoder.weight.requires_grad = False

def encoding(x,Xi,enc_max,enc_min,device):
    # (Batchsize, N_in)
    batchsize = x.shape[0]
    N_in      = x.shape[1]

    #Exponent of distance measure
    exp = 2

    #Reshape input and repeat it along Neurons per sample axis
    x_rep = x.reshape( (batchsize,N_in,1) )
    x_rep = x_rep.repeat(1,1,10)
    # (Batchsize, N_in, Neurons_per_Sample)

    #Reshape reference points and repeat it along Batch and N_in axis
    Xi = Xi.reshape( (1,1,Xi.shape[0]) )
    Xi = Xi.repeat(batchsize,N_in,1)
    # (Batchsize, N_in, Neurons_per_Sample)

    #Calculate distance
    d = torch.abs(x_rep-Xi)**exp                        # Calculate distance to all Xi's

    #Scale with log depending on maximal possible distance
    d_max = (enc_max-enc_min)**exp
    beta = time_steps/np.log(d_max+2)         # scaling factor for d

    #Calculate the spike times and clip them to max
    t = beta*torch.log(d+1)                        # time of spike 
    t = torch.clip(t.int(),0,time_steps-1)
    
    #Convert spiketimes to spiketrains
    spikes = one_hot_encoder(t)
    # (Batchsize, N_in, Neurons_per_Sample,Time)
    spikes = torch.permute(spikes,(3,0,1,2))
    # (Time,Batchsize, N_in, Neurons_per_Sample)

    return spikes

#import matplotlib.pyplot as plt
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#enc_max = 5                # upper limit of encoding
#enc_min = 1                 # lower limit of encoding
#neurons_in = 10
#x = torch.tensor([[1,2],
#                  [3,4],
#                  [5,6]])
#Xi = torch.linspace(enc_min,enc_max,neurons_in)   # equally distributed over [-emc_max,enc_max]
#spikes = encoding(x,Xi,enc_max,enc_min,device,neurons_in=neurons_in,time_steps=30)
##print(Xi)
#print(spikes)
#print(spikes.shape)
#
#if True:
#    plt.imshow(np.transpose(spikes[:,0,:]))
#    plt.xlabel('distance d')
#    plt.ylabel('neuron index')
#    plt.show()
#
#    d = np.linspace(0,9,900)
#    d_max = np.max(d)
#    t_max = 30
#    plt.plot(d,t_max/np.log(d_max+2)*np.log(d+1),'b',label='^1')
#    plt.plot(d,t_max/np.log((d_max+1)**2)*np.log((d**2)+1),'r--',label='^2')
#    plt.plot(d,t_max/np.log((d_max+1)**4)*np.log((d**4)+1),'g--',label='^4')
#    plt.xlabel('distance d')
#    plt.ylabel('spike latency tau')
#    plt.legend()
#    plt.grid(True)
#    plt.show()
#
#if False:
#    d_start = np.linspace(0,10,100)
#    beta = 1
#    A = 10
#    d = np.abs(A-d_start)
#    d[d<=beta]= beta+1e-8 
#
#    plt.plot(d_start,np.log(d/(d-beta)))
#    plt.grid(True)
#    plt.show()
