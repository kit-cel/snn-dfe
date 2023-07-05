import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import MODEM

import Equalizer            #Requires Equalizer.py
import Loops                #Requires Training.py

##########################################################################################################

NN_type = "SNN"
EQ_type = "DFE"

modulation = "PAM"
modulation_order = 2;
use_gray            = 'yes';                                #Usage of gray encoding 'yes' or 'no'

ebno_dB = np.array([18]);

BATCHSIZE_MIN = 200000
BATCHSIZE_MAX = 200000

EPOCHS          = 5;                                  #Number of epochs
BATCHES_PER_SNR = 2000 * np.ones(ebno_dB.size);
BATCHES_PER_SNR = BATCHES_PER_SNR.astype(int)

LR = 1e-3
GAMMA = 0.9

EQ_TAP_CNT = 41;
HIDDEN_SIZE = 80

LMQ  = True
#IMDD = False
IMDD = True

channel_type = "im_dd"

SPS         = 3

#Channel from paper
#BAUDRATE    = 100*10**9
#D           = -5
#L           = 5000
#wavelen     = 1270
#beta        = 0.2
#BIAS = 2.25

#Short range channel with common parameters
BAUDRATE    = np.array([50*10**9])
D           = -17
#L           = np.array([5000])
L           = np.linspace(1000,10000,10)
wavelen     = np.array([1550])
beta        = 0.2
BIAS = 0.25

##########################################################################################################

# use pytorch device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for training:",device)

##########################################################################################################

modem = MODEM.MODEM("PAM",modulation_order,use_gray,device,IMDD=IMDD,LMQ=LMQ);

##########################################################################################################

# initialize model and copy to target device
print(str(2**modulation_order)+"-"+modulation);
print("model = equalizer("+str(EQ_TAP_CNT)+",1)")
model = Equalizer.SNN_Perceptron_Equalizer( 8 * (EQ_TAP_CNT//2+1) + 2**modulation_order * (EQ_TAP_CNT//2),
                                            HIDDEN_SIZE,
                                            2**modulation_order,
                                            device=device, 
                                            dt = 1 * 10**(-3) );
model.to(device)

Init_Model_Path = str(2**modulation_order)+"-"+modulation+"_"+str(NN_type)+"_"+str(EQ_type)+"_Equalizer_"+channel_type+".pt";

torch.save(model,Init_Model_Path);

Loops.Train(Init_Model_Path, 
            NN_type, EQ_type,
            BAUDRATE, L, D, wavelen, SPS, beta, BIAS,
            ebno_dB, modem, EQ_TAP_CNT,
            EPOCHS, BATCHES_PER_SNR, BATCHSIZE_MIN, BATCHSIZE_MAX, LR, GAMMA);

