import torch
import torch.nn as nn

import norse
from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFCell, LIFRecurrentCell
from norse.torch.module.leaky_integrator import LICell

class ANN_Perceptron_Equalizer(nn.Module):
    def __init__(self, input_features, hidden_features, output_features,device):
        super(ANN_Perceptron_Equalizer, self).__init__()
        self.fc_1 = nn.Linear(input_features, hidden_features).to(device)
        self.activ = nn.ReLU().to(device)
        self.fc_2 = nn.Linear(hidden_features, output_features).to(device)
                                            
    def forward(self, x):
        out = self.fc_1(x)
        out = self.activ(out)
        out = self.fc_2(out)
        return out, 0
 
class SNN_Perceptron_Equalizer(nn.Module):
    def __init__(self, input_features, hidden_features, output_features,device, dt=0.001):
        super(SNN_Perceptron_Equalizer, self).__init__()
        self.device = device;

        self.input_features  =  input_features
        self.hidden_features =  hidden_features
        self.output_features =  output_features

        self.p1 = LIFParameters(
                    alpha       = torch.nn.Parameter(torch.full((hidden_features,), torch.as_tensor(100.0)).to(device)),
                    v_th        = torch.nn.Parameter(torch.full((hidden_features,), torch.as_tensor(  1.0)).to(device)),
                    v_leak      = torch.nn.Parameter(torch.tensor(0.0).to(device)),
                    v_reset     = torch.nn.Parameter(torch.full((hidden_features,), torch.as_tensor(  0.0)).to(device)),
                    tau_mem_inv = torch.nn.Parameter(torch.full((hidden_features,), torch.as_tensor(100.0)).to(device)),
                    tau_syn_inv = torch.nn.Parameter(torch.full((hidden_features,), torch.as_tensor(200.0)).to(device))
                    )

        self.input_layer    = torch.nn.Linear(self.input_features   , self.hidden_features,bias=False).to(device);
        self.LIFRec_layer   = LIFRecurrentCell(self.hidden_features , self.hidden_features,p=self.p1,dt=dt,autapses=False).to(device);
        self.output_layer   = torch.nn.Linear(self.hidden_features  , self.output_features,bias=True).to(device);

    def __decode_sum(self,x):
        x = torch.sum(x,0);
        return x

    def forward(self, x):
        s0 = None

        seq_length,batch_size,_ = x.shape

        self.LIFRec_spikes  = torch.zeros(x.shape[0], x.shape[1], self.hidden_features, device=self.device)

        out = torch.zeros(x.shape[0], x.shape[1], self.output_features, device=self.device)

        for ts in range(seq_length):
            z               =  self.input_layer(x[ts,:,:]);
            z,s0            =  self.LIFRec_layer(z,s0);
            self.LIFRec_spikes[ts,:,:]  = z;
            z               =  self.output_layer(z)
            
            out[ts][:][:]   =  z

        hidden_z = self.LIFRec_spikes;
        spikerate = torch.sum(hidden_z)/(hidden_z.shape[0]*hidden_z.shape[1]*hidden_z.shape[2])
        
        z = self.__decode_sum(out)
        return z, spikerate


class SNN_Georg_et_al_Equalizer(nn.Module):
    def __init__(self, input_features, hidden_features, output_features,device, dt=0.001):
        super(SNN_Georg_et_al_Equalizer, self).__init__()
        self.device = device;

        self.input_features  =  input_features
        self.hidden_features =  hidden_features
        self.output_features =  output_features

        self.p1 = LIFParameters(
                    alpha       = torch.nn.Parameter(torch.full((hidden_features,), torch.as_tensor(100.0)).to(device)),
                    v_th        = torch.nn.Parameter(torch.full((hidden_features,), torch.as_tensor(  1.0)).to(device)),
                    v_leak      = torch.nn.Parameter(torch.tensor(0.0).to(device)),
                    v_reset     = torch.nn.Parameter(torch.full((hidden_features,), torch.as_tensor(  0.0)).to(device)),
                    tau_mem_inv = torch.nn.Parameter(torch.full((hidden_features,), torch.as_tensor(1/0.003)).to(device)),
                    tau_syn_inv = torch.nn.Parameter(torch.full((hidden_features,), torch.as_tensor(1/0.003)).to(device))
                    )

        self.p2 = LIFParameters(
                    v_leak      = torch.nn.Parameter(torch.tensor(0.0).to(device)),
                    tau_mem_inv = torch.nn.Parameter(torch.full((output_features,), torch.as_tensor(1/0.003)).to(device)),
                    tau_syn_inv = torch.nn.Parameter(torch.full((output_features,), torch.as_tensor(1/0.003)).to(device))
                    )

        self.input_layer    = torch.nn.Linear(self.input_features   , self.hidden_features,bias=True).to(device);
        #self.LIFRec_layer   = LIFCell(self.hidden_features , self.hidden_features,p=self.p1,dt=dt,autapses=False).to(device);
        self.LIF_layer      = LIFCell(p=self.p1,dt=dt).to(device);
        self.output_layer   = torch.nn.Linear(self.hidden_features  , self.output_features,bias=True).to(device);
        self.LI_layer       = LICell(p=self.p2,dt=dt)

    def __decode_max(self,x):
        x, _ = torch.max(x, 0)
        return x

    def forward(self, x):
        s0 = None
        s1 = None

        seq_length,batch_size,_ = x.shape

        self.LIF_spikes  = torch.zeros(x.shape[0], x.shape[1], self.hidden_features, device=self.device)

        out = torch.zeros(x.shape[0], x.shape[1], self.output_features, device=self.device)

        for ts in range(seq_length):
            z               =  self.input_layer(x[ts,:,:]);
            z,s0            =  self.LIF_layer(z,s0);
            self.LIF_spikes[ts,:,:]  = z;
            z               =  self.output_layer(z)
            v,s1            =  self.LI_layer(z,s1)
            
            out[ts][:][:]   =  v

        hidden_z = self.LIF_spikes;
        spikerate = torch.sum(hidden_z)/(hidden_z.shape[0]*hidden_z.shape[1]*hidden_z.shape[2])
        
        out = self.__decode_max(out)
        return out, spikerate


