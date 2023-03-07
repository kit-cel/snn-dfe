import torch
import numpy as np

import rrc  # requires rrc.py to be accessible (e.g., in the same directory) Root Raised Cosine

class IM_DD_Channel:
    def __init__(self,SYMCNT,Baud_Rate,rate,USF,D,channel_length,wavelen,RRC_beta,EBN0,bias,device):
        self.__SYMCNT = SYMCNT;
        self.__R      = Baud_Rate;
        self.__rate   = rate;
        self.__dt     = 1/(USF*Baud_Rate);
        self.__USF    = USF;
        self.__D      = D;                                                      #In ps/nm/km
        self.__length = channel_length;
        self.__wavelen = wavelen;
        self.__rrc_beta = RRC_beta;
        self.__EBN0   = EBN0;
        self.__beta = -self.__wavelen**2/(2*np.pi*300000)*self.__D*10**(-27);
        self.__bias = bias
        self.__device = device
        self.__generate_H();
        self.__symbol_energy = None;
        self.__noise_energy  = None;

    def __generate_H(self):
        g = rrc.get_rrc_ir(self.__SYMCNT * self.__USF + 1, self.__USF, 1, self.__rrc_beta);  # RRC impulse response nonodd countnumber for symetric filter with clear middle
        g = g[:-1:]
        g = g/np.sqrt(np.sum(np.abs(g)**2))                                                  # Scale to unit energy
        g = np.fft.ifftshift(g)
        g = g/np.sqrt(self.__USF)
        self.__G = np.fft.fft(g)                                                             # Store for MF

        G_without_bias = np.copy(self.__G)                                                   # Copy MF
        bias_spec = self.__bias*np.ones(G_without_bias.size,dtype=complex)                                 # Calculate bias spectrum (numerical issue prevention)
        bias_spec = np.fft.fft(bias_spec)

        dw = (2 * np.pi * np.fft.fftfreq(self.__SYMCNT*self.__USF, self.__dt))               # Calculate frequencies
        C = np.exp( -1j * self.__beta / 2 * dw**2 * self.__length )                          # Calculate chromatic dispersion

        self.__H_without_bias = G_without_bias*C                                             # Store overall channel frequency response upto the photo diode
        self.__H_bias         = bias_spec*C

        self.__G = torch.from_numpy(self.__G).reshape((-1,1)).to(self.__device).flatten()
        self.__H_without_bias = torch.from_numpy(self.__H_without_bias).reshape((-1,1)).to(self.__device).flatten()
        self.__H_bias = torch.from_numpy(self.__H_bias).reshape((-1,1)).to(self.__device).flatten()

    def apply(self,symbols):
        if(symbols.shape[0] != self.__SYMCNT):
            print("Symbol size was defined as "+str(self.__SYMCNT)+"!\nPlease initialize new IM_DD_Channel object or insert the specified amount of data!")
            quit()

        us_symbols = torch.zeros((symbols.shape[0]*self.__USF,1),dtype=torch.complex128,device=self.__device).flatten()    #Allocate storage
        us_symbols[::self.__USF] = symbols                                                          #Upsample symbols

        us_symbols = torch.fft.fft(us_symbols)
        pre_sqrt = us_symbols * self.__H_without_bias + self.__H_bias  #Apply pulseforming,bias and chromatic dispersion
        pre_sqrt = torch.fft.ifft(pre_sqrt)                                            #Transform to time domain
        pre_sqrt = pre_sqrt * torch.sqrt(pre_sqrt.shape[0]/torch.sum(torch.abs(pre_sqrt)**2))

        pos_sqrt = torch.abs(pre_sqrt)**2                                           #Apply photodiode

        esn0_dB = self.__EBN0 + 10 * np.log10( self.__rate )                        #Convert Eb/N0 to Es/N0
        sigma = np.sqrt( (10**(-esn0_dB / 10)) );                                   #Calculate sigma
        noise = torch.randn(pos_sqrt.shape[0],device=self.__device);                #Create noise
        noise = sigma * noise;                                                      #Scale noise to correct SNR

        self.__symbol_energy = torch.sum(torch.abs(pos_sqrt)**2)-torch.mean(pos_sqrt)**2;
        self.__noise_energy  = torch.sum(torch.abs(noise)**2);

        pre_mf = pos_sqrt + noise;                                                  #Add noise to photodiode output

        pre_mf = torch.fft.fft(pre_mf)
        recv_symbols = pre_mf*self.__G                                              #Apply (missmatched) matched filter
        recv_symbols = torch.real(torch.fft.ifft(recv_symbols))

        recv_symbols = recv_symbols[::self.__USF]                                   #Downsample
        recv_symbols = recv_symbols * self.__USF

        return recv_symbols

    def give_energies(self):
        return self.__symbol_energy.clone(),self.__noise_energy.clone()



#################################################
#                    Test                       #
#################################################
#
#import quantizer            #requures quantizer.py
#
#if torch.cuda.is_available():
#    DEVICE = torch.device("cuda")
#else:
#    DEVICE = torch.device("cpu")
#print("Using " + str(DEVICE) + " for training.");
#
#
##SYMCNT      = 1000
#SYMCNT      = 10000
#BAUDRATE    = 100*10**9
#rate        = 2
#USF         = 4
##Channel from paper
#D           = -5
#length      = 5000
#wavelen     = 1270
#beta        = 0.2
#EBN0        = 20
#bias        = 2.25
#
#channel =  IM_DD_Channel(SYMCNT,BAUDRATE,rate,USF,D,length,wavelen,beta,EBN0,bias,DEVICE);
#
#symbols = torch.randint(4,size=(SYMCNT,1))-1.5
#symbols = symbols*2
#symbols = symbols.flatten()
#print(np.unique(symbols))
#
#recv_symbols = channel.apply(symbols)#-bias**2
#e_sym,e_noise = channel.give_energies();
#e_sym = e_sym.detach().cpu().numpy()
#e_noise = e_noise.detach().cpu().numpy()
#
#print("Mean: "+str(torch.mean(recv_symbols)))
#print("Symbols power: "+str(e_sym/symbols.shape[0]))
#print("Noise power: "+str(e_noise/symbols.shape[0]))
#print("SNR: "+str(10*np.log10(e_sym/e_noise)))
#
#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(recv_symbols.detach().cpu().numpy())
#
#recv_symbols = recv_symbols-torch.mean(recv_symbols)
#
#h = np.correlate(recv_symbols.detach().cpu().numpy(),recv_symbols.detach().cpu().numpy(),'full')/recv_symbols.shape[0]
#h = h/np.sqrt(np.sum(np.abs(h)**2))
#h = h[np.argmax(h)-6:np.argmax(h)+7:]                       #17 Tap of channel ir
##h = h[np.argmax(h)-200:np.argmax(h)+201:]
#
#plt.figure()
#classbits = 8
#counts = torch.zeros(2**classbits)
#dx,rx_quant = quantizer.midrise(recv_symbols,classbits,25);
#for n in range(2**classbits):
#    counts[n] = torch.count_nonzero(rx_quant==n)
##print(rx_quant)
##print(counts)
#plt.plot(np.linspace(-25,25,counts.shape[0]),counts.numpy(),label='Equalized')
#plt.legend()
#plt.title("Histogramm")
#
#plt.figure()
#plt.stem(h)
#plt.title("h")
#
#plt.show()
