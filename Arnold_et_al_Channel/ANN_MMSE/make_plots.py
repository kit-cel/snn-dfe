import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

Torch_LMMSE_BER = np.array( [0.0212725, 0.01411275, 0.00906   , 0.00546375, 0.00323   , 
                             0.0018465, 0.001068  , 0.00054925, 0.0002945 , 0.000167  ] );

#Boecherer scale SPS3
BER = np.array( [0.021465, 0.014635, 0.009295, 0.005505, 0.00326167, 0.00189333, 0.00112, 0.00053667, 0.0003, 0.00015833] );

#TX Sym feedback
DFE_BER = np.array( [0.01898833, 0.01266333, 0.00809167, 0.00473, 0.00286667, 0.001685  , 0.00097167, 0.00052167, 0.00032   , 0.00017667] );


ANN_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM = np.array([1.70967504e-02, 1.07602496e-02, 6.14150008e-03, 3.22075002e-03,
                                                          1.46149995e-03, 5.97500009e-04, 2.01749994e-04, 5.67499992e-05,
                                                          9.99999975e-06, 1.75000002e-06]);
ANN_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM_NS = np.array([1.69992503e-02, 1.05147501e-02, 5.88029996e-03, 3.12720006e-03,
                                                             1.49259996e-03, 6.12650008e-04, 2.07649995e-04, 7.22499972e-05,
                                                             2.20500006e-05, 3.85000021e-06]);
ANN_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM_NS_14dB = np.array([1.74515508e-02, 1.05944499e-02, 5.90205006e-03, 2.98424996e-03,
                                                                  1.32990000e-03, 5.30150020e-04, 1.64800003e-04, 4.34499998e-05,
                                                                  7.90000013e-06, 1.15000000e-06]);


ANN_REF_BER  = np.array([2*1e-2,1.2*1e-2,7.0*1e-3,3.5*1e-3,2.0*1e-3,1*1e-3,5*1e-4,2.5*1e-4,1.5*1e-4])
ANN_REF2_BER = np.array([2*1e-2,1.1*1e-2,6.5*1e-3,3.2*1e-3,1.8*1e-3,8*1e-4,3*1e-4,1.3*1e-4,5.0*1e-5])
SNN_REF_BER  = np.array([2*1e-2,1.2*1e-2,7.0*1e-3,3.5*1e-3,2.0*1e-3,1*1e-3,5*1e-4,2.0*1e-4,8.5*1e-5])

BER_ref = np.array([2.8*10**(-2),1.8*10**(-2),10**(-2),6.5*10**(-3),3.5*10**(-3),
                    2*10**(-3),10**(-3),6*10**(-4),3*10**(-4),1.8*10**(-4)])
plt.figure()
plt.semilogy(sigma,Torch_LMMSE_BER,label='Torch MMSE Simulation',marker='o')
#plt.semilogy(sigma,BER,label='MMSE Simulation',marker='x')

plt.semilogy(sigma,DFE_BER,label='DFE',marker=10)

plt.semilogy(sigma,ANN_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM,label='ANN MMSE 5E 2000B LR1e-3 BS200000 1UspB TfI ADAM',marker='o')
plt.semilogy(sigma,ANN_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM_NS,label='ANN MMSE 5E 2000B LR1e-3 BS200000 1UspB TfI ADAM No Sceduler',marker='o')
plt.semilogy(sigma,ANN_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM_NS_14dB,label='ANN MMSE 5E 2000B LR1e-3 BS200000 1UspB TfI ADAM No Sceduler 17 dB Operationpoint',marker='o')

plt.semilogy(sigma,BER_ref,label='Reference LMMSE Arnold et.al. MMSE',marker='o')
plt.semilogy(sigma[:-1:],ANN_REF_BER,label='Reference LMMSE Arnold et.al. ANN1',marker='o')
plt.semilogy(sigma[:-1:],ANN_REF2_BER,label='Reference LMMSE Arnold et.al. ANN2',marker='o')
plt.semilogy(sigma[:-1:],SNN_REF_BER,label='Reference LMMSE Arnold et.al. SNN',marker='X')
plt.title("4-PAM BER/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("BER");
plt.grid(which='both')
plt.legend();

plt.show()

