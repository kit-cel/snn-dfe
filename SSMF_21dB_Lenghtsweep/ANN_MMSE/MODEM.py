import numpy as np
import torch

import bit                  #requires bit.py
import gray_coder           #requires gray_coder.py
import quantizer            #requures quantizer.py

class MODEM:

#####################################################################
#                   General part                                    #
#####################################################################

    def __init__(self, modulation, m, gray, device,IMDD=False,LMQ=False):
        self.change(modulation,m,gray,device,IMDD=IMDD,LMQ=LMQ);

    def change(self, modulation, m, gray, device,IMDD,LMQ):
        if(modulation == "QAM"):
            if(IMDD==True):
                print("QAM is infeasable with IMDD...")
                quit()
            if(LMQ==True):
                print("LMQ is not compatible with QAM...")
                quit()
            self.__modulation = modulation;
            self.__QAM_change(m);
        elif(modulation == "PSK"):
            if(IMDD==True):
                print("PSK is infeasable with IMDD...")
                quit()
            if(LMQ==True):
                print("LMQ is not compatible with PSK...")
                quit()
            self.__modulation = modulation;
            self.__PSK_change(m);
        elif(modulation == "PAM"):
            self.__modulation = modulation;
            self.__PAM_change(m,IMDD=IMDD,LMQ=LMQ);
        elif(modulation == "APSK"):
            if(IMDD==True):
                print("APSK is infeasable with IMDD...")
                quit()
            if(LMQ==True):
                print("LMQ is not compatible with APSK...")
                quit()
            self.__modulation = modulation;
            self.__APSK_change(m);
        else:
            print(str(modulation)+" is not implemented yet!")
            print("Only QAM,PSK,PAM,APSK available.")
            quit()

        if(gray=='yes'):
            self.__encode_gray = gray;
        elif(gray=='no'):
            self.__encode_gray = gray;
        else:
            print("Please enter for gray either yes or no");
            quit();

        self.__device = device

        if(modulation == "QAM"):
            self.__bitmap = bit.BIT_MAPPER(self.__mod_order//2,self.__device)
        elif(modulation == "APSK"):
            self.__bitmap = bit.BIT_MAPPER(self.__mod_order//2,self.__device)
        else:
            self.__bitmap = bit.BIT_MAPPER(self.__mod_order,self.__device)

    def modulate(self,bits):
        if(self.__modulation == "QAM"):
            return self.__QAM_modulate(bits);
        elif(self.__modulation == "PSK"):
            return self.__PSK_modulate(bits);
        elif(self.__modulation == "PAM"):
            return self.__PAM_modulate(bits);
        elif(self.__modulation == "APSK"):
            return self.__APSK_modulate(bits);
        else:
            print(str(modulation)+" is not implemented yet!")
            print("Only QAM,PSK,PAM,APSK available.")
            quit()

    def A_to_Symbols(self,A_i,A_q=0):
        if(self.__modulation == "QAM"):
            return self.__QAM_A_to_Symbols(A_i,A_q);
        elif(self.__modulation == "PSK"):
            return self.__PSK_A_to_Symbols(A_i);
        elif(self.__modulation == "PAM"):
            return self.__PAM_A_to_Symbols(A_i);
        elif(self.__modulation == "APSK"):
            return self.__APSK_A_to_Symbols(A_i,A_q);
        else:
            print(str(modulation)+" is not implemented yet!")
            print("Only QAM,PSK,PAM,APSK available.")
            quit()

    def A_to_bits(self,A_i,A_q=0):
        if(self.__modulation == "QAM"):
            return self.__QAM_A_to_bits(A_i,A_q);
        elif(self.__modulation == "PSK"):
            return self.__PSK_A_to_bits(A_i);
        elif(self.__modulation == "PAM"):
            return self.__PAM_A_to_bits(A_i);
        elif(self.__modulation == "APSK"):
            return self.__APSK_A_to_bits(A_i,A_q);
        else:
            print(str(modulation)+" is not implemented yet!")
            print("Only QAM,PSK,PAM,APSK available.")
            quit()

    def demodulate(self,symbols,optimize=True,name=None):
        if(self.__modulation == "QAM"):
            return self.__QAM_demodulate(symbols);
        elif(self.__modulation == "PSK"):
            return self.__PSK_demodulate(symbols);
        elif(self.__modulation == "PAM"):
            return self.__PAM_demodulate(symbols,optimize,name);
        elif(self.__modulation == "APSK"):
            return self.__APSK_demodulate(symbols);
        else:
            print(str(modulation)+" is not implemented yet!")
            print("Only QAM,PSK,PAM,APSK available.")
            quit()

    def give_mod_order(self):
        return self.__mod_order;
    def give_max_real(self):
        return self.__max_re;
    def give_max_imag(self):
        return self.__max_im;
    def give_mean_power(self):
        return self.__power;
    def give_DC(self):
        return self.__DC;
    def give_amplification(self):
        return self.__amp;
    def is_gray_encoding(self):
        return self.__encode_gray;
    def give_modulation(self):
        return self.__modulation;
    def give_IMDD(self):
        if(self.__modulation == "PAM"):
            return self.__IMDD;
        else:
            return False;
    def give_LMQ(self):
        if(self.__modulation == "PAM"):
            return self.__LMQ;
        else:
            return False;
    def give_device(self):
        return self.__device;

#####################################################################
#                   Bit to class / Class to bit                     #
#####################################################################

    def __bit_2_class(self,bits):
        if(self.__modulation == "QAM"):
            return self.__bit_2_class_complex(bits);
        elif(self.__modulation == "APSK"):
            return self.__bit_2_class_complex(bits);
        else:
            return self.__bit_2_class_real(bits);

    def __bit_2_class_real(self,bits):
        if(bits.shape[0]/self.__mod_order%1 != 0):
            print("Bits have to be a multiple of modulation order");
            print(str(bits.shape[0])+"/"+str(self.__mod_order)+"%1 ="+str(bits.shape[0]/self.__mod_order%1)+" != 0");
            quit();

        bits_i = torch.empty(size=(self.__mod_order,bits.shape[0]//self.__mod_order),device=self.__device);       

        for n in range(self.__mod_order):
            bits_i[n] = bits[n::self.__mod_order];#Fill mod order columns in every row, row equals symbol in binary
        
        bits_i = bits_i.T
 
        A_i = self.__bitmap.map(bits_i)
        A_i = A_i.flatten().int();                                           #Reshape
   
        if(self.__encode_gray=='yes'):        
            A_i = gray_coder.encode(A_i);                                   #Convert symbols to graycoded symbols
        else:
            A_i = A_i;                                                      #Dont convert to gray
        return A_i

    def __bit_2_class_complex(self,bits):
        if(bits.shape[0]/self.__mod_order%1 != 0):
            print("Bits have to be a multiple of modulation order");
            print(str(bits.shape[0])+"/"+str(self.__mod_order)+"%1 ="+str(bits.shape[0]/self.__mod_order%1)+" != 0");
            quit();

        #Array with symbols in columns and every row corresponds to a bit in the symbol
        bits_i = torch.empty(size=(self.__mod_order//2,bits.shape[0]//self.__mod_order),device=self.__device);       
        bits_q = torch.empty(size=(self.__mod_order//2,bits.shape[0]//self.__mod_order),device=self.__device);       
        for n in range(self.__mod_order//2):
            bits_i[n] = bits[                    n::self.__mod_order];  #Fill I Bits (every m'ths bit and diff start point)
            bits_q[n] = bits[self.__mod_order//2+n::self.__mod_order];  #Same as Q but in upper half of bit chunk of size m

        bits_i = bits_i.T
        bits_q = bits_q.T

        A_i = self.__bitmap.map(bits_i)
        A_q = self.__bitmap.map(bits_q)
        A_i = A_i.flatten().int();                                           #Reshape
        A_q = A_q.flatten().int();
        if(self.__encode_gray=='yes'):        
            A_i = gray_coder.encode(A_i);                                   #Convert symbols to graycoded symbols
            A_q = gray_coder.encode(A_q);
        else:
            A_i = A_i;                                                      #Dont convert to gray
            A_q = A_q;
        return A_i, A_q

    def __class_2_bit(self,A_i,A_q=0):
        if(self.__modulation == "QAM"):
            return self.__class_2_bit_complex(A_i,A_q);
        elif(self.__modulation == "APSK"):
            return self.__class_2_bit_complex(A_i,A_q);
        else:
            return self.__class_2_bit_real(A_i);
 
    def __class_2_bit_real(self,A_i):
        delta_x,A_i = quantizer.midrise(A_i,self.__mod_order,1);        #Quantize Re to classes

        if(self.__encode_gray=='yes'):        
            A_i = gray_coder.decode(A_i);                                 #Convert back from gray
        else:
            A_i = A_i;                                                    #No backconversion
        bits_i = self.__bitmap.demap(A_i)
        recv_bits = bits_i.flatten()
        return delta_x, recv_bits
   
    def __class_2_bit_complex(self,A_i,A_q):
        delta_x,A_i = quantizer.midrise(A_i,self.__mod_order//2,1);    #Quantize Re and Im to classes
        delta_x,A_q = quantizer.midrise(A_q,self.__mod_order//2,1);
        if(self.__encode_gray=='yes'):        
            A_i = gray_coder.decode(A_i);                                 #Convert back from gray
            A_q = gray_coder.decode(A_q);
        else:
            A_i = A_i;                                                    #No backconversion
            A_q = A_q;
        bits_i = self.__bitmap.demap(A_i)
        bits_q = self.__bitmap.demap(A_q)
        recv_bits = torch.hstack( (bits_i,bits_q) ).flatten()
        return delta_x, recv_bits

#####################################################################
#                   Store LMQ codebook                              #
#####################################################################

    def save_codebook(self,name):
        if(self.__LMQ == True):
            if(self.__Q != None):
                self.__Q.save_codebook(name)
            else:
                print("Execute a demodulation before you store the codebook!")
                quit()
        else:
            print("LMQ is deactivated! Cannot store the codebook!")
            quit()

#####################################################################
#                   QAM specifics                                   #
#####################################################################

    def __QAM_change(self,m):
        self.__power    = (2**m-1)/6;                               #Average Power of QAM without DC
        self.__max_re   = 1/np.sqrt(self.__power)*((2**(m/2)-1)/2); #Maximal real part of symbol
        self.__max_im   = 1/np.sqrt(self.__power)*((2**(m/2)-1)/2); #Maximal imag part of symbol
        self.__DC       = ((2**(m/2)-1)/2) + 1j*((2**(m/2)-1)/2);   #DC of QAM constellation without shift
        self.__amp = (1 - 1/np.sqrt(2**m))/self.__max_re;           #Calculate amp for sym in decision reg
        if(m%2 != 0):
            print("m has to be a muliple of 2");
            quit();
        else:
            self.__mod_order = m;

    def __QAM_modulate(self,bits):
        A_i,A_q = self.__bit_2_class(bits);
        symbols = self.__QAM_A_to_Symbols(A_i,A_q);                               #Create symbols
        A = 2**(self.__mod_order//2)*A_q+A_i                                #Convert re/im integer to class label
        return A, symbols

    def __QAM_A_to_Symbols(self,A_i,A_q):
        M = 2**(self.__mod_order//2);
        symbols = (A_i + 1j*A_q)                                            #Create QAM symbols
        symbols = symbols - self.__DC                                       #Remove DC
        k_scale = np.sqrt(1/self.__power)                                   #Calculate scaling factor for unit power
        symbols = k_scale * symbols;                                        #Scale to Unit Power
        return symbols

    def __QAM_A_to_bits(self,A):
        A_q = A // 2**(self.__mod_order//2)
        A_i = A - 2**(self.__mod_order//2)*A_q
        return self.__class_2_bit(A_i,A_q);

    def __QAM_demodulate(self,symbols):
        symbols = self.__amp*symbols                                        #Amp symbols so they lie in decision region
        A_i = torch.real(symbols);
        A_q = torch.imag(symbols);
        return self.__class_2_bit(A_i,A_q);

#####################################################################
#                   PSK specifics                                   #
#####################################################################

    def __PSK_change(self,m):
        self.__power    = 1;                                        #Average Power of PSK without DC
        self.__max_re   = 1;                                        #Maximal real part of symbol
        self.__max_im   = 1;                                        #Maximal imag part of symbol
        self.__DC       = 0;                                        #DC of PSK constellation without shift
        self.__amp      = 1;                                        #Calculate amp for sym in decision reg
        if(m < 1):
            print("Modulation order has to be larger than 0!");
            quit();
        elif(m > 16):
            print("Modulation order is only supported up to 16!");
            print("Due to numerical issues a 2^"+str(m)+" aka. "+str(2**m)+"-PSK is not supported!")
            quit();
        else:
            self.__mod_order = m;

    def __PSK_modulate(self,bits):
        A = self.__bit_2_class(bits);                               #Map bits to class
        symbols = self.__PSK_A_to_Symbols(A);                       #Create symbols 
        return A, symbols

    def __PSK_A_to_Symbols(self,A):
        M = 2**(self.__mod_order);
        phi = 2 * np.pi * (A/M) - np.pi; 

        phi = phi + 1e-6                                            #Due to different output of torch.exp and np.exp
        
        symbols = torch.exp(1j * phi);                              #Create M-PSK symbols
        symbols = symbols - self.__DC                               #Remove DC
                                                                    #Calculate power as mean of current power and stored power
        k_scale = np.sqrt(1/self.__power)                           #Calculate scaling factor for unit power
        symbols = k_scale * symbols;                                #Scale to Unit Power
        return symbols

    def __PSK_A_to_bits(self,A):
        M = 2**(self.__mod_order);
        A = (A - (M-1)/2)/((M-1)/2)                                         #Scale to -1 - +1
        return self.__class_2_bit(A);

    def __PSK_demodulate(self,symbols):
        symbols = self.__amp*symbols                                #Amp symbols so they lie in decision region
 
        phi = torch.angle(symbols);                                 #Calculate angle of symbols

        M = 2**(self.__mod_order);
        A_i = phi/(2*np.pi) * M;                                    #Angle to symbol class

        angle = 2 * np.pi / M;
        angle = angle / 2;
        neg_ang = torch.tensor([np.pi/(2*np.pi)*M],device=self.__device)
        A_i = torch.where(phi > (np.pi-angle) , neg_ang , A_i)              #If "overflow" of angle occurs, map the symbol to the most negative angle 
                                                                            #due to circulatity of angle
        N = 2**self.__mod_order;
        A_i = A_i * 2/N                                                     #Scale to -1 - +1

        return self.__class_2_bit(A_i);

#####################################################################
#                   PAM specifics                                   #
#####################################################################

    def __PAM_change(self,m,IMDD,LMQ):
        self.__LMQ = LMQ
        if(LMQ == True):
            self.__Q = None
        if(IMDD==False):
            self.__power    = (2**(2*m)-1)/3;                       #Average Power of PAM without DC
        else:
            self.__power    =  np.sqrt(1/2**m * np.sum(np.arange(0,2**m,1)**2));
        self.__max_re   = 1;                                        #Maximal real part of symbol
        self.__max_im   = 0;                                        #Maximal imag part of symbol
        self.__DC       = 0;                                        #DC of PAM constellation without shift
        if(IMDD==False):
            self.__amp      = np.sqrt(self.__power);                #Calculate amp for sym in decision reg
        else:
            self.__amp      = self.__power;
        self.__IMDD     = IMDD;

        if(m < 1):
            print("m shall be larger than 0!");
            quit();
        elif(m > 13):
            print("Modulation order is only supported up to 13!");
            print("Due to numerical issues a 2^"+str(m)+" aka. "+str(2**m)+"-PAM is not supported!")
            quit();
        else:
            self.__mod_order = m;

    def __PAM_modulate(self,bits):
        A = self.__bit_2_class(bits);                               #Map bits to class
        symbols = self.__PAM_A_to_Symbols(A);                       #Create symbols 
        return A, symbols

    def __PAM_A_to_Symbols(self,A):
        M = 2**(self.__mod_order);
        A = A.float()
        
        if(self.__IMDD == False):
            A = A + 1;
            symbols = 2*A - 1 - M;                                  #Create M-PAM symbols
        else:
            symbols = A;                                            #Create M-PAM symbols
    
        if(self.__IMDD == False):
            symbols = symbols - self.__DC                                   #Remove DC
        else:                                                               #Calculate power as mean of current power and stored power
            symbols = torch.sign(symbols)*torch.sqrt(torch.abs(symbols))

        k_scale = np.sqrt(1/self.__power)                                   #Calculate scaling factor for unit power
        symbols = k_scale * symbols;                                        #Scale to Unit Power
        return symbols

    def __PAM_A_to_bits(self,A):
        M = 2**(self.__mod_order);
        A = (A - (M-1)/2)/((M-1)/2)                                         #Scale to -1 - +1
        return self.__class_2_bit(A);

    def __PAM_demodulate(self,symbols,optimize,name):
        symbols = self.__amp*symbols                                        #Amp symbols so they lie in decision region

        M = 2**(self.__mod_order);

        if(self.__IMDD == False):
            A_i = (symbols+1+M)/2;                                         #Bring symbols back to A space
            A_i = A_i - 1;
        else:
            A_i = symbols

        if(self.__LMQ == True):
            self.__Q = quantizer.Lloyd_Max(self.__mod_order,self.__device)
            #self.__Q = quantizer.Lloyd_Max(self.__mod_order,self.__device,np.arange(0,M,1))
            if(optimize==True):
                _,codebook = self.__Q.optimize(A_i)
                delta_x = codebook
            else:
                self.__Q.load_codebook(name)
                delta_x = self.__Q.give_codebook()
            A_i,_ = self.__Q.quantize(A_i)

        A_i = (A_i - (M-1)/2)/((M-1)/2)                                     #Scale to -1 - +1

        return self.__class_2_bit(A_i);

#####################################################################
#                   APSK specifics                                  #
#####################################################################

    #########################################################
    #   Based on: QAM to circular isomorphic constellations #
    #             doi: 10.1109/ASMS-SPSC.2016.7601550.      #
    #########################################################

    def __APSK_change(self,m):
        if(m==2):
            self.__power    = (2**m-1)/6;                           #Average Power of APSK without DC
        elif(m==4):
            self.__power    = 1.400049242445002*(2**m-1)/6;         #Average Power of APSK without DC
        elif(m==6):
            self.__power    = 1.47622076599796*(2**m-1)/6;          #Average Power of APSK without DC
        elif(m==8):
            self.__power    = 1.4940962089400427*(2**m-1)/6;        #Average Power of APSK without DC 
        elif(m==10):
            self.__power    = 1.498502665232182*(2**m-1)/6;         #Average Power of APSK without DC  
        elif(m==12):
            self.__power    = 1.499786724286523*(2**m-1)/6;         #Average Power of APSK without DC   
        elif(m==14):
            self.__power    = 1.4997117188746585*(2**m-1)/6;        #Average Power of APSK without DC    
        else:
            self.__power    = 1.4997618593061517*(2**m-1)/6;        #Average Power of APSK without DC

        self.__max_re   = 1/np.sqrt(self.__power)*((2**(m/2)-1)/2); #Maximal real part of symbol
        self.__max_im   = 1/np.sqrt(self.__power)*((2**(m/2)-1)/2); #Maximal imag part of symbol
        self.__DC       = ((2**(m/2)-1)/2) + 1j*((2**(m/2)-1)/2);   #DC of APSK constellation without shift
        self.__amp      = (1 - 1/np.sqrt(2**m))/self.__max_re;      #Calculate amp for sym in decision reg
        if(m%2 != 0):
            print("m has to be a muliple of 2");
            quit();
        else:
            self.__mod_order = m;

    def __APSK_modulate(self,bits):
        A_i,A_q = self.__bit_2_class(bits);
        symbols = self.__APSK_A_to_Symbols(A_i,A_q);                               #Create symbols
        A = 2**(self.__mod_order//2)*A_q+A_i                                #Convert re/im integer to class label
        return A, symbols

    def __APSK_A_to_Symbols(self,A_i,A_q):
        M = 2**(self.__mod_order//2);
        symbols = (A_i + 1j*A_q)                                            #Create QAM symbols
        symbols = symbols - self.__DC                                       #Remove DC
                                                                            #Calculate power as mean of current power and stored power
        k_scale = np.sqrt(1/self.__power)                                   #Calculate scaling factor for unit power
        symbols = k_scale * symbols;                                        #Scale to Unit Power
        ##### QAM to circular isomorphic #####
        sqrt2 = torch.tensor([np.sqrt(2)],device=self.__device)
        reim_symbols = torch.hstack((torch.abs(torch.real(symbols.reshape((symbols.shape[0],1)))),torch.abs(torch.imag(symbols.reshape((symbols.shape[0],1))))))
        symbols = (sqrt2*torch.max(reim_symbols,1).values)/(torch.sqrt(torch.real(symbols)**2+torch.imag(symbols)**2)+10e-6)*symbols
        #####################################

        return symbols

    def __APSK_A_to_bits(self,A):
        A_q = A // 2**(self.__mod_order//2)
        A_i = A - 2**(self.__mod_order//2)*A_q
        return self.__class_2_bit(A_i,A_q);

    def __APSK_demodulate(self,symbols):
        ##### Inverse QAM to circular isomorphic #####
        sqrt2 = torch.tensor([np.sqrt(2)],device=self.__device)
        reim_symbols = torch.hstack((torch.abs(torch.real(symbols.reshape((symbols.shape[0],1)))),torch.abs(torch.imag(symbols.reshape((symbols.shape[0],1))))))
        symbols = (torch.sqrt(torch.real(symbols)**2+torch.imag(symbols)**2)+10e-6)/(sqrt2*torch.max(reim_symbols,1).values)*symbols
        ##############################################

        symbols = self.__amp*symbols                                        #Amp symbols so they lie in decision region
        A_i = torch.real(symbols);
        A_q = torch.imag(symbols);
        return self.__class_2_bit(A_i,A_q);

###################################################################################################
#                               Testing                                                           #                                                              
###################################################################################################
#
#import time
#
#if torch.cuda.is_available():
#    DEVICE = torch.device("cuda")
#else:
#    DEVICE = torch.device("cpu")
#print("Using " + str(DEVICE) + " for training.");
#
#import matplotlib.pyplot as plt
#
#
#
#k = 1
#k = 2
#k = 3
#k = 4
#k = 8
#k = 13
#k = 14;
#k = 16;
#k = 17;
#k = 3;
#k=20
#
#st = time.time()
#a = MODEM("QAM",k,'yes',DEVICE);
#a = MODEM("PSK",k,'yes',DEVICE);
#a = MODEM("PAM",k,'yes',DEVICE);
#a = MODEM("APSK",k,'yes',DEVICE);
#a = MODEM("PAM",k,'yes',DEVICE,IMDD=True,LMQ=False);
#a = MODEM("PAM",k,'yes',DEVICE,IMDD=False,LMQ=True);
#
#et = time.time()
#print("Initialization time: "+str(et-st))
#
#print(str(2**k)+"-APSK")
#
#print(a.give_mean_power());
#a.change(4,'no');
#a.change(4,'yes');
#print(a.give_mean_power());
#
#print(a.give_mod_order());
#print(a.give_max_real());
#print(a.give_max_imag());
#print(a.give_mean_power());
#print(a.is_gray_encoding());
#
#print("External access:")
#print(getattr(a,'__power'));
#setattr(a,'__power',5)
#print(a.__power)
#
#a = MODEM();
#
#
#a.change(4,'pups');
#print(a.give_mean_power());
#
#N = 2**(k+2);
#N = 256*k;
#N = 25600*k;
#N = 200000*k;
#N = 25600000*k;
#bits = torch.randint(0,2,size=(N,1)).flatten().to(DEVICE);
#print("Bits: "+str(bits));
##A,symbols = a.modulate(bits);
#st = time.time()
#A,symbols = a.modulate(bits);
#et = time.time()
#print("Modulation time: "+str(et-st))
#
#
#print("A: "+str(A));
##print("Symbols:" + str(symbols));
##print(symbols.shape)
#
##Symbol_energy  = np.sum(np.abs(symbols)**2);
##print("Actual symbol energy: "+str(Symbol_energy));
##avg_power = Symbol_energy / symbols.size;
##print("Actual symbol power: "+str(avg_power));
##
##recv_symbols = symbols + 0.2 * (torch.randn((symbols.shape[0],1)).flatten() + 1j * torch.randn((symbols.shape[0],1)).flatten()).to(DEVICE);
##recv_symbols = symbols;
##
##IMDD
#symbols = symbols - torch.min(symbols) + 0.1
##recv_symbols = symbols**2;
#recv_symbols = symbols**2 + 0.1 * (torch.randn((symbols.shape[0],1)).flatten() + 1j * torch.randn((symbols.shape[0],1)).flatten()).to(DEVICE);
#recv_symbols = recv_symbols - torch.mean(recv_symbols)
#
#Symbol_energy  = torch.sum(torch.abs(recv_symbols)**2);
#print("Actual symbol energy at rx: "+str(Symbol_energy));
#avg_power = Symbol_energy / symbols.shape[0];
#print("Actual symbol power at rx: "+str(avg_power));
##
##d_x,rx_bits = a.demodulate(recv_symbols);
#
#st = time.time()
#d_x,rx_bits = a.demodulate(recv_symbols);
#et = time.time()
#print("Demodulation time: "+str(et-st))
#
#print("Original bits: "+str(bits));
#print("Received bits: "+str(rx_bits));
#
#if(torch.sum((rx_bits-bits)**2) == 0):
#    print("Success!");
#else:
#    print(str(torch.sum((rx_bits-bits)**2))+" Errors occured!");
#    print(str(torch.sum((rx_bits-bits)**2)/bits.shape[0])+" %");
#
#symbols = symbols.cpu().numpy()
#recv_symbols = recv_symbols.cpu().numpy()
#
#plt.figure()
#plt.scatter(np.real(symbols),np.imag(symbols),label='Symbols',marker='x');
#plt.title("Constellation");
#plt.xlabel("Real");
#plt.ylabel("Imag");
#plt.legend();
#
#plt.figure()
#plt.stem(np.angle(recv_symbols),label='Angle symbols');
#plt.title("Angle");
#plt.xlabel("Index");
#plt.ylabel("Angle");
#plt.legend();
#
#plt.figure()
#plt.scatter(np.real(recv_symbols),np.imag(recv_symbols),label='Symbols',marker='x');
#plt.title("Recv constellation");
#plt.xlabel("Real");
#plt.ylabel("Imag");
#plt.legend();
#
#plt.show();

