import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import MODEM

import Equalizer            #Requires Equalizer.py
import Loops                #Requires Training.py

##########################################################################################################

NN_type = "ANN"
EQ_type = "DFE"

modulation = "PAM"
modulation_order = 2;
use_gray            = 'yes';                                #Usage of gray encoding 'yes' or 'no'

#ebno_dB = np.array([18]);
ebno_dB = np.arange(12,22,1);                          #EbN0 values at which Symbols will be created
#op_point = None
op_point = 14

BATCHSIZE = 10000

EPOCHS = 1000;                                  #Number of epochs

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
#L           = np.linspace(1000,10000,10)
L           = np.array([5000])
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

Init_Model_Path = str(2**modulation_order)+"-"+modulation+"_"+str(NN_type)+"_"+str(EQ_type)+"_Equalizer_"+channel_type+".pt";

Loops.Eval(Init_Model_Path, 
            NN_type, EQ_type,
            BAUDRATE, L, D, wavelen, SPS, beta, BIAS,
            ebno_dB, modem, EQ_TAP_CNT,
            EPOCHS, BATCHSIZE,
            op_point);

