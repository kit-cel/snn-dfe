from os import system

class Screen:
    def __init__(self, PATH = None, 
                       BAUDRATE=None, L=None, D=None, wavelen=None,
                       BAUDRATE_TOTAL=None, L_TOTAL=None, wavelen_TOTAL=None,
                       SPS=None,beta=None, BIAS=None,
                       modulation_order=None,modulation=None,gray=None,IMDD=None,LMQ=None,
                       Total_idx=None,Total_max=None,
                       Epoch=None,Epochs=None,
                       Batch=None,Batches=None,
                       Batchsize=None,LR=None,gamma=None,
                       NN_type=None,EQ_type=None,
                       alpha=None,
                       SNR=None ,SNR_TOTAL=None,
                       BER=None,
                       SER=None,
                       Loss=None, Reg=None,
                       Time_per_Batch=None, Time_per_Epoch=None):
        self.Init_Model_Path    = PATH;
        self.BAUDRATE           = BAUDRATE;
        self.BAUDRATE_TOTAL     = BAUDRATE_TOTAL;
        self.L                  = L;
        self.L_TOTAL            = L_TOTAL;
        self.D                  = D;
        self.wavelen            = wavelen;
        self.wavelen_TOTAL      = wavelen_TOTAL;
        self.SPS                = SPS;
        self.beta               = beta;
        self.BIAS               = BIAS;
        self.SNR                = SNR; 
        self.SNR_TOTAL          = SNR_TOTAL; 
        self.modulation_order   = modulation_order
        self.modulation         = modulation
        self.gray               = gray
        self.IMDD               = IMDD
        self.LMQ                = LMQ
        self.TOTAL_IDX          = Total_idx
        self.TOTAL_MAX          = Total_max
        self.EPOCH              = Epoch
        self.EPOCHS             = Epochs
        self.BATCH              = Batch
        self.BATCHES            = Batches
        self.Batchsize          = Batchsize
        self.LR                 = LR
        self.gamma              = gamma
        self.NN_type            = NN_type
        self.EQ_type            = EQ_type
        self.alpha              = alpha
        self.SNR                = SNR
        self.BER                = BER
        self.SER                = SER
        self.LOSS               = Loss
        self.REG                = Reg
        self.Time_per_Batch     = Time_per_Batch
        self.Time_per_Epoch     = Time_per_Batch
    def show(self):
        _ = system('clear')

        print("################################################################################################################")
        print(self.Init_Model_Path)
        print("Baudrate          : "+str(self.BAUDRATE_TOTAL)+                        
              " Bd\t Lenght     : "+str(self.L_TOTAL) + " m")
        print("Wavelenght        : "+str(self.wavelen_TOTAL) +
              " \t\t SNR        : "+str(self.SNR_TOTAL))
        print("----------------------------------------------------------------------------------------------------------------")
        print("Baudrate          : "+str(self.BAUDRATE)+                        
              " Bd\t Lenght     : "+str(self.L) +
              " m\t Dispersion  : "+str(self.D) +
              "\t Wavelenght      : "+str(self.wavelen))
        print("Samples per Symbol: "+str(self.SPS)+                                  
              "\t\t\t beta       : "+str(self.beta)+
              "\t Bias        : "+str(self.BIAS))
        print("Modulation        : "+str(2**self.modulation_order)+"-"+str(self.modulation)+
              "\t\t Gray label.: "+str(self.gray)+
              "\t Squared Mod.: "+str(self.IMDD)+
              "\t Lloyd-Max-Demod.: "+str(self.LMQ))
        print("Total             : "+str(self.TOTAL_IDX)+" of "+str(self.TOTAL_MAX)+
              "\t\t Epochs     : "+str(self.EPOCH)+" of "+str(self.EPOCHS)+ 
              "\t Batch       : "+str(self.BATCH)+" of "+str(self.BATCHES))
        print("Batchsize         : "+str(self.Batchsize)+
              "\t\t Learn. rate: "+str(self.LR) +
              "\t Schedul. gma: "+str(self.gamma))
        print("Equalizer type    : "+str(self.NN_type)+"-"+str(self.EQ_type))
        if(self.NN_type == "SNN"):
            print("Alpha:            : "+str(self.alpha))
        print("SNR               : "+str(self.SNR)+" dB")
        print("BER               : "+str(self.BER)+" %")
        print("SER               : "+str(self.SER)+" %")
        print("Loss              : "+str(self.LOSS) +
              "\t Regul.     : "+str(self.REG))
        print("Time per Batch    : "+str(self.Time_per_Batch)+"\t Time per Epoch: "+str(self.Time_per_Epoch))
        print("################################################################################################################")


##########################################################################################
# <Path>
# <Baudrate> <L> <wavelen> <ebn0>
# <Baudrate> <L> <D> <wavelen> <SPS> <beta> <BIAS>
# <Order>-<Modulation> <Gray> <IMDD> <LMQ>
# Epoch: <E> of <E>    Batch: <B> of <B>
# <Batchsize>   <LR>    <Gamma>
# <ANN/SNN/...> 
# <Alpha>
# <BER>
# <SER>
# <Loss> / <Reg>
##########################################################################################
#
#from time import sleep,time
#import numpy as np
#
#sym_cnt = 100
#Init_Model_Path = "4-PAM_SNN_Equalizer_Channel-im_dd_18_dB.sd" 
#                  
#BAUDRATE = 100*10**9 
#L        = 5000
#D        = -17
#wavelen  = 1550
#SPS      = 3
#beta     = 0.2
#BIAS     = 0.25
#ebno_dB  = np.arange(12,22,1)
#modulation_order = 2
#modulation = 'PAM'
#gray = 'Yes'
#IMDD = 'True'
#LMQ  = 'True'
#EPOCHS = 10
#BATCHES_PER_SNR = np.linspace(20,200,10).astype(int)
#BATCHSIZE = (np.linspace(100,10000,10)//2)*2
#LR = 0.001
#GAMMA = 0.9
#NN_type = "SNN"
#NN_type = "ANN"
#EQ_TYPE = "DFE"
#alpha = np.linspace(1,100,10)
#
#monitor = Screen(PATH = Init_Model_Path, 
#                 BAUDRATE=BAUDRATE, L=L, D=D, wavelen=wavelen,
#                 BAUDRATE_TOTAL=BAUDRATE, L_TOTAL=L, wavelen_TOTAL=wavelen, SNR_TOTAL=ebno_dB,
#                 SPS=SPS,beta=beta, BIAS=BIAS,
#                 modulation_order=modulation_order,modulation=modulation,gray=gray,IMDD=IMDD,LMQ=LMQ,
#                 LR=LR,gamma=GAMMA,
#                 NN_type=NN_type,EQ_type=EQ_TYPE)
#monitor.show()
#sleep(1)
#
#monitor.EPOCHS=EPOCHS
#monitor.BATCHES=BATCHES_PER_SNR[3]
#monitor.TOTAL_MAX=monitor.EPOCHS*monitor.BATCHES
#
#for e in range(EPOCHS):
#    e_st = time()
#    monitor.EPOCH=e;
#    monitor.Batchsize=BATCHSIZE[e];
#    monitor.alpha=alpha[e];
#    for b in range(BATCHES_PER_SNR[3]):
#        b_st = time()
#        monitor.TOTAL_IDX=(e+1)*(b+1);
#        monitor.BATCH=b;
#        monitor.SNR=ebno_dB[3];
#        monitor.BER=np.random.randint(100);
#        monitor.SER=np.random.randint(100);
#        monitor.LOSS=np.random.randn();
#        monitor.REG=np.random.randn();
#        monitor.show();
#        sleep(0.1);
#        b_et = time()
#        monitor.Time_per_Batch = b_et-b_st;
#    e_et = time()
#    monitor.Time_per_Epoch = e_et-e_st;


