import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from norse.torch import LIFParameters

import time

import Process_Screen           #Requires Process_Screen.py
import im_dd                    #Requires im_dd.py
import quantizer                #requires quantizer.py
import encoding                 #requires encoding.py

####################################################################################################################
#                                           Training                                                               #
####################################################################################################################

def ANN_MMSE_Symbols_to_Batch(recv_symbols       ,EQ_TAP_CNT,one_hot_encoder,device):
    rollsym = torch.hstack( (recv_symbols[-EQ_TAP_CNT//2+1::],recv_symbols) )
    rollsym = torch.hstack( (rollsym,recv_symbols[:EQ_TAP_CNT//2:]) )
    inputs = rollsym.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).float()
    
    return inputs

def ANN_DFE_Symbols_to_Batch(recv_symbols,symbols,EQ_TAP_CNT,one_hot_encoder,device):
    rollFB = torch.hstack( (symbols[-EQ_TAP_CNT//2+1::],symbols) )
    rollFB = torch.hstack( (rollFB,symbols[:EQ_TAP_CNT//2:]) )
    FB_in  = rollFB.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).float()

    rollsym = torch.hstack( (recv_symbols[-EQ_TAP_CNT//2+1::],recv_symbols) )
    rollsym = torch.hstack( (rollsym,recv_symbols[:EQ_TAP_CNT//2:]) )
    in_sym = rollsym.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).float()

    inputs = torch.hstack( (FB_in[::,0:EQ_TAP_CNT//2:1] , in_sym[::,EQ_TAP_CNT//2::1]) )

    return inputs

def SNN_MMSE_Symbols_to_Batch(recv_symbols       ,EQ_TAP_CNT,one_hot_encoder,device):
    #Feedforward
    enc_max = 4                # upper limit of encoding
    enc_min = -4               # lower limit of encoding
    neurons_in = 10
    Xi = torch.linspace(enc_min,enc_max,neurons_in).to(device)   # equally distributed over [-emc_max,enc_max]

    rollsym = torch.hstack( (recv_symbols[-EQ_TAP_CNT//2+1::],recv_symbols) )
    rollsym = torch.hstack( (rollsym,recv_symbols[:EQ_TAP_CNT//2:]) )
    in_sym  = rollsym.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).float()
    
    inputs  = encoding.encoding(in_sym,Xi,enc_max,enc_min,device)

    inputs  = inputs.reshape( (inputs.shape[0],inputs.shape[1],-1) )

    return inputs


def SNN_DFE_Symbols_to_Batch(recv_symbols,A      ,EQ_TAP_CNT,one_hot_encoder,device):
    rollA = torch.hstack( (A[-EQ_TAP_CNT//2+1::],A) )
    rollA = torch.hstack( (rollA,A[:EQ_TAP_CNT//2:]) )
    A_in  = rollA.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).int()
    A_in  = A_in[::,0:EQ_TAP_CNT//2:1]
    OneHot_A_in = one_hot_encoder(A_in)
    OneHot_A_in = OneHot_A_in.reshape( (OneHot_A_in.shape[0],-1) )

    rollsym = torch.hstack( (recv_symbols[-EQ_TAP_CNT//2+1::],recv_symbols) )
    rollsym = torch.hstack( (rollsym,recv_symbols[:EQ_TAP_CNT//2:]) )
    in_sym = rollsym.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).float()
    in_sym = in_sym[::,EQ_TAP_CNT//2::1]
    d_x,bin_re = quantizer.midtread_binary_unipolar(torch.abs(in_sym),8,4,device)
    sign = torch.permute(torch.sign(in_sym).repeat(8,1,1),(1,2,0))
    tern_in_sym = bin_re*sign
    tern_in_sym = tern_in_sym.reshape( (tern_in_sym.shape[0],-1) )

    inputs = torch.hstack( (OneHot_A_in, tern_in_sym) )

    inputs = inputs.reshape( (1,inputs.shape[0],-1) )
    inputs = torch.vstack( (inputs,torch.zeros( (9,inputs.shape[1],inputs.shape[2]), device=device)) )

    return inputs

def Train(Init_Model_Path, 
          NN_type, EQ_type,
          BAUDRATE, L, D, wavelen, SPS, beta, BIAS,
          ebno_dB, modem, EQ_TAP_CNT,
          EPOCHS, BATCHES_PER_SNR, BATCHSIZE_MIN, BATCHSIZE_MAX, LR, GAMMA):

    t_st = time.time()

    modulation_order = modem.give_mod_order();
    modulation       = modem.give_modulation();
    gray             = modem.is_gray_encoding();
    IMDD             = modem.give_IMDD();
    LMQ              = modem.give_LMQ();
    device           = modem.give_device();

    softmax = nn.Softmax(dim=1)
 
    if(NN_type == "ANN"):
        if(EQ_type == "DFE"):
            one_hot_encoder = None
        elif(EQ_type == "MMSE"):
            one_hot_encoder = None
        else:
            print(str(NN_type)+"-"+str(EQ_type)+" is not implemented yet...")
            quit()
    elif(NN_type=="SNN"):
        if(EQ_type == "DFE"):
            one_hot_encoder = nn.Embedding(2**modulation_order,2**modulation_order).to(device)
            one_hot_encoder.weight.data = torch.eye(2**modulation_order,device=device)
            one_hot_encoder.weight.requires_grad = False
        elif(EQ_type == "MMSE"):
            one_hot_encoder = None
        else:
            print(str(NN_type)+"-"+str(EQ_type)+" is not implemented yet...")
            quit()
    else:
        print(str(NN_type)+" is not implemented yet...")
        quit()
 
    monitor = Process_Screen.Screen(PATH = Init_Model_Path, 
                     BAUDRATE=BAUDRATE, L=L, D=D, wavelen=wavelen,
                     BAUDRATE_TOTAL=BAUDRATE, L_TOTAL=L, wavelen_TOTAL=wavelen,SNR_TOTAL=ebno_dB,
                     SPS=SPS,beta=beta, BIAS=BIAS,
                     modulation_order=modulation_order,modulation=modulation,gray=gray,IMDD=IMDD,LMQ=LMQ,
                     LR=LR,gamma=GAMMA,
                     NN_type=NN_type,EQ_type=EQ_type)

    loss_fn = torch.nn.CrossEntropyLoss()

    Symbol_energy = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
    Noise_energy  = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
    SER           = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
    BER           = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));

    monitor.EPOCHS=EPOCHS
    monitor.TOTAL_MAX=EPOCHS*np.sum(BATCHES_PER_SNR)*wavelen.size*BAUDRATE.size*L.size*ebno_dB.size

    for idx_Wl, Wl_i in enumerate(wavelen):
        monitor.wavelen = Wl_i;
        for idx_B, B_i in enumerate(BAUDRATE):
            monitor.B = B_i
            for idx_L, L_i in enumerate(L):
                monitor.L = L_i
                for idx,snr_i in enumerate(ebno_dB):
                    train_loss = [];

                    BATCHES = BATCHES_PER_SNR[idx]
                    monitor.BATCHES = BATCHES
                    monitor.SNR = snr_i

                    model = torch.load(Init_Model_Path, map_location=torch.device(device));  #initialize model and copy to target device

                    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
                    #sceduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

                    batch_size_per_epoch = np.linspace(BATCHSIZE_MIN,BATCHSIZE_MAX,num=EPOCHS).astype(int)  #Create list of batch sizes
                    batch_size_per_epoch = batch_size_per_epoch//2 * 2                   #Ensure all batch sizes are even

                    #if(NN_type=="SNN"):
                    #    alpha_list = np.linspace(1,100,EPOCHS).astype(float)

                    for epoch_cnt in range(EPOCHS):
                        e_st = time.time()
                        monitor.EPOCH = epoch_cnt+1;

                        sym_cnt = int(batch_size_per_epoch[epoch_cnt])                  #Number of symbols
                        N       = modulation_order*sym_cnt                              #Number of bits
                        monitor.Batchsize = sym_cnt

                        channel = im_dd.IM_DD_Channel(sym_cnt,B_i,modulation_order,SPS,D,L_i,Wl_i,beta,snr_i,BIAS,device);

                        #if(NN_type=="SNN"):
                        #    monitor.alpha=alpha_list[epoch_cnt];
                        #    if(EQ_type == "DFE"):
                        #        model.LIFRec_layer.p.alpha.data = torch.nn.Parameter(torch.full((model.hidden_features,), torch.as_tensor(alpha_list[epoch_cnt])).to(device))
                        #    elif(EQ_type == "MMSE"):
                        #        model.LIF_layer.p.alpha.data = torch.nn.Parameter(torch.full((model.hidden_features,), torch.as_tensor(alpha_list[epoch_cnt])).to(device))
                        #    else:
                        #        print(str(NN_type)+"-"+str(EQ_type)+" is not implemented yet...")
                        #        quit()

                        for batch_cnt in range(BATCHES):
                            monitor.BATCH = batch_cnt+1;
                            monitor.TOTAL_IDX=(epoch_cnt+1)*(batch_cnt+1)*(idx_Wl+1)*(idx_B+1)*(idx_L+1)*(idx+1)

                            b_st = time.time()

                            ##################################### Bits ################################################

                            bits = torch.randint(0,2,size=(N,1),device=device).flatten();

                            ##################################### Sender ################################################

                            A,symbols = modem.modulate(bits);

                            symbols = symbols * np.sqrt(5)

                            rate = modulation_order;

                            esn0_dB = snr_i + 10 * np.log10( rate )

                            #################################### Channel  ################################################

                            recv_symbols = channel.apply(symbols);
                            recv_symbols = recv_symbols - torch.mean(recv_symbols);
                            e_sym,e_noise = channel.give_energies();

                            Symbol_energy[idx_Wl][idx_B][idx_L][idx] += e_sym;
                            Noise_energy[idx_Wl][idx_B][idx_L][idx] += e_noise;

                            #################################### Create batch  ###########################################
                        
                            labels = A.to(torch.int64)

                            if(NN_type == "ANN"):
                                if(EQ_type == "DFE"):
                                    inputs = ANN_DFE_Symbols_to_Batch( recv_symbols,symbols,EQ_TAP_CNT,one_hot_encoder,device);
                                elif(EQ_type == "MMSE"):
                                    inputs = ANN_MMSE_Symbols_to_Batch(recv_symbols        ,EQ_TAP_CNT,one_hot_encoder,device);
                                else:
                                    print(str(NN_type)+"-"+str(EQ_type)+" is not implemented yet...")
                                    quit()
                            elif(NN_type == "SNN"):
                                if(EQ_type == "DFE"):
                                    inputs = SNN_DFE_Symbols_to_Batch(recv_symbols,A,EQ_TAP_CNT,one_hot_encoder,device);
                                elif(EQ_type == "MMSE"):
                                    inputs = SNN_MMSE_Symbols_to_Batch(recv_symbols        ,EQ_TAP_CNT,one_hot_encoder,device);
                                else:
                                    print(str(NN_type)+"-"+str(EQ_type)+" is not implemented yet...")
                                    quit()
                            else:
                                print(str(NN_type)+" is not implemented yet...")
                                quit()

                            #################################### Check intermediate performance  #########################

                            if(batch_cnt % np.ceil(BATCHES/10) == 0):
                                outputs, reg = model(inputs)
                                loss = loss_fn(outputs, labels)
                                A_eq = torch.argmax(softmax(outputs),axis=1).int();

                                d_x, recv_bits = modem.A_to_bits(A_eq);

                                classerr = torch.sum(torch.where(A_eq == A,0,1))
                                monitor.SER = classerr/A.shape[0]*100

                                test = torch.sum(torch.abs(recv_bits-bits))
                                monitor.BER = test/bits.shape[0]*100
                                monitor.LOSS = loss
                                monitor.REG = reg
                                monitor.show()
                                    
                            #################################### Update weights  ###########################################

                            outputs,reg = model(inputs)
                            loss = loss_fn(outputs, labels)
                            train_loss.append( loss.detach().cpu().numpy() );

                            # compute gradient
                            loss.backward()
                                            
                            # optimize
                            optimizer.step()
                                                        
                            # reset gradients
                            optimizer.zero_grad()

                            b_et = time.time()
                            monitor.Time_per_Batch = b_et-b_st;

                        e_et = time.time()
                        monitor.Time_per_Epoch = e_et-e_st;

                        SER[idx_Wl][idx_B][idx_L][idx] = classerr/A.shape[0]
                        BER[idx_Wl][idx_B][idx_L][idx] = test/bits.shape[0]
                        #sceduler.step();
                       
                        monitor.LR = LR
                        #monitor.LR = sceduler.get_last_lr();
                        del channel

                    torch.save(model.state_dict(),Init_Model_Path[:-3]+"_"+str(Wl_i)+"_nm_"+str(B_i)+"_Bd_"+str(L_i)+"_m_"+str(snr_i)+"_dB.sd");

                    BER[idx_Wl][idx_B][idx_L][idx] = BER[idx_Wl][idx_B][idx_L][idx]/EPOCHS
                    SER[idx_Wl][idx_B][idx_L][idx] = SER[idx_Wl][idx_B][idx_L][idx]/EPOCHS
                    del model
                    del inputs
                    del labels
                    del outputs

                    plt.figure(1)
                    plt.semilogy(train_loss);
                    plt.title("Training Loss");
                    plt.xlabel("Optimization Steps over all Epochs")
                    plt.ylabel("Loss")
                    mng = plt.get_current_fig_manager()
                    mng.full_screen_toggle()
                    plt.savefig("Training_Loss_"+str(NN_type)+"_"+str(EQ_type)+"_"+str(Wl_i)+"_nm_"+str(B_i)+"_Bd_"+str(L_i)+"_m_"+str(snr_i)+"_dB.svg", format="svg")
                    plt.clf()

    t_et = time.time()
    print()
    print("Time to train: "+str(t_et-t_st)+"\n")

    #Write results to file
    fp = open("Training"+str(NN_type)+"_"+str(EQ_type)+".txt", 'w')
    fp.write("EbN0: "+str(ebno_dB)+"\n")
    fp.write("BER: "+str(BER)+"\n")
    fp.write("SER: "+str(SER)+"\n")
    fp.write("Noisepower: "+str(-10*np.log10(Noise_energy/(SPS*np.sum(batch_size_per_epoch)*BATCHES)))+"\n")
    fp.write("SNR: "+str(10*np.log10(Symbol_energy/(Noise_energy)))+"\n")
    fp.write("Time to train: "+str(t_et-t_st)+"\n")
    fp.close()

####################################################################################################################
#                                           Evaluation                                                             #
####################################################################################################################
def Operate_ANN_MMSE(model,recv_symbols, EQ_TAP_CNT, monitor, device):
    softmax = nn.Softmax(dim=1)

    sym_cnt = recv_symbols.shape[0]
    EPOCHS  = recv_symbols.shape[1]

    A_eq = torch.zeros( (sym_cnt,EPOCHS), device=device);
    REG  = torch.zeros( (sym_cnt,EPOCHS), device=device);
    
    monitor.EPOCHS=EPOCHS
    
    for epoch_cnt in range(EPOCHS):
        rollsym = torch.hstack( (recv_symbols[-EQ_TAP_CNT//2+1::,epoch_cnt],recv_symbols[::,epoch_cnt]) )
        rollsym = torch.hstack( (rollsym,recv_symbols[:EQ_TAP_CNT//2:,epoch_cnt]) )
        inputs = rollsym.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).float()
 
        monitor.EPOCH = epoch_cnt+1;
        monitor.show()

        outputs,reg = model(inputs)
        A_eq[::,epoch_cnt] = torch.argmax(softmax(outputs),axis=1).int();

        REG[::,epoch_cnt] = reg

    REG = torch.sum(REG)/(sym_cnt*EPOCHS)

    return A_eq, REG


def Operate_ANN_DFE(model,recv_symbols,symbols, EQ_TAP_CNT, modem, monitor, device):
    softmax = nn.Softmax(dim=1)

    sym_cnt = symbols.shape[0]
    EPOCHS  = symbols.shape[1]

    REG = torch.zeros( (sym_cnt,EPOCHS), device=device);
    
    FB_in = torch.zeros( (sym_cnt,EPOCHS,EQ_TAP_CNT//2), device=device);
    FB_in[0] = symbols[-EQ_TAP_CNT//2+1::,::].T

    rollsym = torch.vstack( (recv_symbols[-EQ_TAP_CNT//2+1::,::],recv_symbols) )
    rollsym = torch.vstack( (rollsym,recv_symbols[:EQ_TAP_CNT//2:,::]) )
    in_sym = rollsym.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).float()

    A_eq = torch.zeros( (sym_cnt,EPOCHS), device=device);

    monitor.EPOCHS=sym_cnt

    for n in range(sym_cnt):

        if(n%100 == 0):
            monitor.EPOCH = n+1;
            monitor.show()

        inputs = torch.hstack( (FB_in[n] , in_sym[n,::,EQ_TAP_CNT//2::1]) )

        outputs,reg = model(inputs)
        A_eq[n] = torch.argmax(softmax(outputs),axis=1).int();

        REG[n,::] = reg

        if(n < sym_cnt-1):
            FB_in[n+1,::,0:-1:1] = FB_in[n,::,1::1]
            FB_in[n+1,::,-1]     = modem.A_to_Symbols(A_eq[n])*np.sqrt(5)

    REG = torch.sum(REG)/(sym_cnt*EPOCHS)

    return A_eq,REG

def Operate_SNN_MMSE(model,recv_symbols, EQ_TAP_CNT, monitor, device):
    softmax = nn.Softmax(dim=1)

    sym_cnt = recv_symbols.shape[0]
    EPOCHS  = recv_symbols.shape[1]

    A_eq = torch.zeros( (sym_cnt,EPOCHS), device=device);
    REG  = torch.zeros( (sym_cnt,EPOCHS), device=device);
    
    monitor.EPOCHS=EPOCHS
    
    for epoch_cnt in range(EPOCHS):
        #Feedforward
        enc_max = 4                # upper limit of encoding
        enc_min = -4               # lower limit of encoding
        neurons_in = 10
        Xi = torch.linspace(enc_min,enc_max,neurons_in).to(device)   # equally distributed over [-emc_max,enc_max]

        rollsym = torch.hstack( (recv_symbols[-EQ_TAP_CNT//2+1::,epoch_cnt],recv_symbols[::,epoch_cnt]) )
        rollsym = torch.hstack( (rollsym,recv_symbols[:EQ_TAP_CNT//2:,epoch_cnt]) )
        in_sym  = rollsym.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).float()
 
        inputs  = encoding.encoding(in_sym,Xi,enc_max,enc_min,device)

        inputs  = inputs.reshape( (inputs.shape[0],inputs.shape[1],-1) )

        monitor.EPOCH = epoch_cnt+1;
        monitor.show()

        outputs,reg = model(inputs)
        A_eq[::,epoch_cnt] = torch.argmax(softmax(outputs),axis=1).int();

        REG[::,epoch_cnt] = reg

    REG = torch.sum(REG)/(sym_cnt*EPOCHS)

    return A_eq, REG

def Operate_SNN_DFE(model,recv_symbols,A , EQ_TAP_CNT, one_hot_encoder, monitor, device):
    softmax = nn.Softmax(dim=1)

    sym_cnt = A.shape[0]
    EPOCHS  = A.shape[1]
    REG = torch.zeros( (sym_cnt,EPOCHS), device=device);

    A_in = torch.zeros( (sym_cnt,EPOCHS,EQ_TAP_CNT//2), device=device).int();
    A_in[0] = A[-EQ_TAP_CNT//2+1::,::].T

    rollsym = torch.vstack( (recv_symbols[-EQ_TAP_CNT//2+1::,::],recv_symbols) )
    rollsym = torch.vstack( (rollsym,recv_symbols[:EQ_TAP_CNT//2:,::]) )
    in_sym = rollsym.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).float()
    in_sym = in_sym[::,::,EQ_TAP_CNT//2::1]

    d_x,bin_re = quantizer.midtread_binary_unipolar(torch.abs(in_sym),8,4, device)
    sign = torch.permute(torch.sign(in_sym).repeat(8,1,1,1),(1,2,3,0))
    tern_in_sym = bin_re*sign
    tern_in_sym = tern_in_sym.reshape( (tern_in_sym.shape[0],tern_in_sym.shape[1],-1) )

    A_eq = torch.zeros( (sym_cnt,EPOCHS), device=device);
    
    monitor.EPOCHS=sym_cnt

    for n in range(sym_cnt):

        if(n%100 == 0):
            monitor.EPOCH = n+1;
            monitor.show()

        OneHot_A_in = one_hot_encoder(A_in[n,::]).reshape((A_in.shape[1],-1))

        inputs = torch.hstack( (OneHot_A_in, tern_in_sym[n,::]) )
        inputs = inputs.reshape( (1,inputs.shape[0],inputs.shape[1]) )
        inputs = torch.vstack( (inputs,torch.zeros( (9,inputs.shape[1],inputs.shape[2]), device=device)) )

        outputs, reg = model(inputs)
   
        REG[n,::] = reg

        A_eq[n] = torch.argmax(softmax(outputs),axis=1).int();

        if(n < sym_cnt-1):
            A_in[n+1,::,0:-1:1] = A_in[n,::,1::1]
            A_in[n+1,::,-1] = A_eq[n]

    REG = torch.sum(REG)/(sym_cnt*EPOCHS)

    return A_eq,REG

def Eval(Init_Model_Path, 
         NN_type, EQ_type,
         BAUDRATE, L, D, wavelen, SPS, beta, BIAS,
         ebno_dB, modem, EQ_TAP_CNT,
         EPOCHS, BATCHSIZE,
         op_point):
    with torch.no_grad():
        t_st = time.time()

        modulation_order = modem.give_mod_order();
        modulation       = modem.give_modulation();
        gray             = modem.is_gray_encoding();
        IMDD             = modem.give_IMDD();
        LMQ              = modem.give_LMQ();
        device           = modem.give_device();
     
        model = torch.load(Init_Model_Path, map_location=torch.device(device));  #initialize model and copy to target device
        model.device = device

        if(NN_type == "ANN"):
            if(EQ_type == "DFE"):
                one_hot_encoder = None
            elif(EQ_type == "MMSE"):
                one_hot_encoder = None
            else:
                print(str(NN_type)+"-"+str(EQ_type)+" is not implemented yet...")
                quit()
        elif(NN_type=="SNN"):
            if(EQ_type == "DFE"):
                one_hot_encoder = nn.Embedding(2**modulation_order,2**modulation_order).to(device)
                one_hot_encoder.weight.data = torch.eye(2**modulation_order,device=device)
                one_hot_encoder.weight.requires_grad = False
            elif(EQ_type == "MMSE"):
                one_hot_encoder = None
            else:
                print(str(NN_type)+"-"+str(EQ_type)+" is not implemented yet...")
                quit()
        else:
            print(str(NN_type)+" is not implemented yet...")
            quit()
     
        monitor = Process_Screen.Screen(PATH = Init_Model_Path, 
                         BAUDRATE=BAUDRATE, L=L, D=D, wavelen=wavelen,
                         BAUDRATE_TOTAL=BAUDRATE, L_TOTAL=L, wavelen_TOTAL=wavelen,SNR_TOTAL=ebno_dB,
                         SPS=SPS,beta=beta, BIAS=BIAS,
                         modulation_order=modulation_order,modulation=modulation,gray=gray,IMDD=IMDD,LMQ=LMQ,
                         NN_type=NN_type,EQ_type=EQ_type)

        loss_fn = torch.nn.CrossEntropyLoss()

        Symbol_energy = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        Noise_energy  = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        SER           = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        BER           = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        REG           = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        errors        = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        BER_Var       = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        BER_Varbuf    = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size,EPOCHS));

        monitor.TOTAL_MAX=EPOCHS*BATCHSIZE*wavelen.size*BAUDRATE.size*L.size*ebno_dB.size

        for idx_Wl, Wl_i in enumerate(wavelen):
            monitor.wavelen = Wl_i;
            for idx_B, B_i in enumerate(BAUDRATE):
                monitor.B = B_i
                for idx_L, L_i in enumerate(L):
                    monitor.L = L_i
                    for idx,snr_i in enumerate(ebno_dB):
                        monitor.BATCHES = 1
                        monitor.BATCH   = 1;
                        monitor.SNR     = snr_i

                        if(op_point == None):
                            path = Init_Model_Path[:-3]+"_"+str(Wl_i)+"_nm_"+str(B_i)+"_Bd_"+str(L_i)+"_m_"+str(snr_i)+"_dB.sd"
                        else:
                            path = Init_Model_Path[:-3]+"_"+str(Wl_i)+"_nm_"+str(B_i)+"_Bd_"+str(L_i)+"_m_"+str(op_point)+"_dB.sd"

                        monitor.Init_Model_Path = path
                        model.load_state_dict(torch.load(path, map_location=torch.device(device)));
     
                        batch_size_per_epoch = BATCHSIZE                                     #Number of symbols per batch
                        batch_size_per_epoch = batch_size_per_epoch//2 * 2                   #Ensure all batch sizes are even

                        e_st = time.time()

                        sym_cnt = int(batch_size_per_epoch)                             #Number of symbols
                        N       = modulation_order*sym_cnt                              #Number of bits
                        monitor.Batchsize = sym_cnt

                        monitor.TOTAL_IDX=(1+idx)*sym_cnt

                        channel = im_dd.IM_DD_Channel(sym_cnt,B_i,modulation_order,SPS,D,L_i,Wl_i,beta,snr_i,BIAS,device);

                        ##########################################################################
                        #   Create "iterations"-times parallel independent datasteams to create  #
                        #   to allow for parallel computation.                                   #
                        #   This is needed since due to the feedback an independent datasteam is #
                        #   not paralizable over the time domain, since the decisions of the     #
                        #   succeeding symbols depend on the previous decision.                  #
                        #   Allocate space to store the decisions which are fed back and         #
                        #   initialize them with the "preamble" aka previous symbols, since      #
                        #   the channel applies cyclic convolution.                              #
                        ##########################################################################

                        bits            = torch.zeros( (N      ,EPOCHS), device=device);
                        recv_bits       = torch.zeros( (N      ,EPOCHS), device=device);
                        A               = torch.zeros( (sym_cnt,EPOCHS), device=device);
                        A_eq            = torch.zeros( (sym_cnt,EPOCHS), device=device);
                        symbols         = torch.zeros( (sym_cnt,EPOCHS), device=device);
                        recv_symbols    = torch.zeros( (sym_cnt,EPOCHS), device=device);

                        for epoch_cnt in range(EPOCHS):
                            ##################################### Bits ################################################

                            bits[::,epoch_cnt] = torch.randint(0,2,size=(N,1),device=device).flatten();

                            ##################################### Sender ################################################

                            A[::,epoch_cnt],symbols[::,epoch_cnt] = modem.modulate(bits[::,epoch_cnt]);

                            symbols[::,epoch_cnt] = symbols[::,epoch_cnt] * np.sqrt(5)

                            rate = modulation_order;

                            esn0_dB = snr_i + 10 * np.log10( rate )

                            #################################### Channel  ################################################

                            recv_symbols[::,epoch_cnt] = channel.apply(symbols[::,epoch_cnt]);
                            recv_symbols[::,epoch_cnt] = recv_symbols[::,epoch_cnt] - torch.mean(recv_symbols[::,epoch_cnt]);
                            e_sym,e_noise = channel.give_energies();

                            Symbol_energy[idx_Wl][idx_B][idx_L][idx] += e_sym;
                            Noise_energy[idx_Wl][idx_B][idx_L][idx] += e_noise;

                        if(NN_type == "ANN"):
                            if(EQ_type == "DFE"):
                                A_eq, reg = Operate_ANN_DFE(model,recv_symbols,symbols, EQ_TAP_CNT, modem, monitor, device);
                            elif(EQ_type == "MMSE"):
                                A_eq, reg = Operate_ANN_MMSE(model,recv_symbols, EQ_TAP_CNT, monitor, device);
                            else:
                                print(str(NN_type)+"-"+str(EQ_type)+" is not implemented yet...")
                                quit()
                        elif(NN_type == "SNN"):
                            if(EQ_type == "DFE"):
                                A_eq, reg = Operate_SNN_DFE(model,recv_symbols,A , EQ_TAP_CNT, one_hot_encoder, monitor, device);
                            elif(EQ_type == "MMSE"):
                                A_eq, reg = Operate_SNN_MMSE(model,recv_symbols, EQ_TAP_CNT, monitor, device);
                            else:
                                print(str(NN_type)+"-"+str(EQ_type)+" is not implemented yet...")
                                quit()
                        else:
                            print(str(NN_type)+" is not implemented yet...")
                            quit()
                        
                        for epoch_cnt in range(EPOCHS):
                            delta_x, recv_bits[::,epoch_cnt] = modem.A_to_bits(A_eq[::,epoch_cnt]);

                        test                                    = (torch.sum(torch.abs(recv_bits-bits),axis=0)).detach().cpu()
                        BER[idx_Wl][idx_B][idx_L][idx]          = (torch.sum(test)/(bits.shape[0]*bits.shape[1])).numpy()
                        BER_Varbuf[idx_Wl][idx_B][idx_L][idx]   = (test/bits.shape[0]).numpy()
                        errors[idx_Wl][idx_B][idx_L][idx]       = (torch.sum(test)).numpy()
                        REG[idx_Wl][idx_B][idx_L][idx]          = reg.detach().cpu().numpy()

                        BER_Var[idx_Wl][idx_B][idx_L][idx] = 1/(EPOCHS-1)*np.sum(BER_Varbuf[idx_Wl][idx_B][idx_L][idx]**2)-EPOCHS/(EPOCHS-1)*BER[idx_Wl][idx_B][idx_L][idx]**2

                        monitor.REG = reg

                        classerr = torch.sum(torch.where(A_eq == A,0,1))
                        monitor.SER = classerr/(A.shape[0]*A.shape[1])*100
                        SER[idx_Wl][idx_B][idx_L][idx] = (classerr/(A.shape[0]*A.shape[1])).detach().cpu().numpy()

                        monitor.BER = BER[idx_Wl][idx_B][idx_L][idx] * 100
                            
                        e_et = time.time()
                        monitor.Time_per_Epoch = e_et-e_st;
                        
                        monitor.show()

                        del channel

        t_et = time.time()
        print()
        print("Time to eval: "+str(t_et-t_st)+"\n")

        #Write results to file
        if(op_point == None):
            fp = open("Evaluation_"+str(NN_type)+"_"+str(EQ_type)+".txt", 'w')
        else:
            fp = open("Evaluation_"+str(NN_type)+"_"+str(EQ_type)+"_Operation_Point_"+str(op_point)+"_dB.txt", 'w')
        fp.write("EbN0: "+str(ebno_dB)+"\n")
        fp.write("BER: "+str(BER)+"\n")
        fp.write("Errors: "+str(errors)+"\n")
        fp.write("SER: "+str(SER)+"\n")
        fp.write("Noisepower: "+str(-10*np.log10(Noise_energy/(SPS*BATCHSIZE*EPOCHS)))+"\n")
        fp.write("SNR: "+str(10*np.log10(Symbol_energy/(Noise_energy)))+"\n")
        fp.write("BER Variance: "+str(BER_Var)+"\n")
        fp.write("Regularization: "+str(REG)+"\n")
        fp.write("Time to eval: "+str(t_et-t_st)+"\n")
        fp.close()


