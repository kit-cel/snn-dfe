import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

Torch_LMMSE_BER = np.array( [0.0212725, 0.01411275, 0.00906   , 0.00546375, 0.00323   , 
                             0.0018465, 0.001068  , 0.00054925, 0.0002945 , 0.000167  ] );

#TX Sym feedback
DFE_BER = np.array( [0.01898833, 0.01266333, 0.00809167, 0.00473, 0.00286667, 0.001685  , 0.00097167, 0.00052167, 0.00032   , 0.00017667] );


ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM = np.array( [1.60995007e-02, 9.46424976e-03, 5.12349978e-03, 2.60224994e-03,
                                                                   1.11925006e-03, 4.49999981e-04, 1.25249987e-04, 3.59999976e-05,
                                                                   8.25000025e-06, 7.49999981e-07] );
ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM_NS = np.array( [1.60438996e-02, 9.50669963e-03, 5.26979985e-03, 2.59819999e-03,
                                                                      1.18080003e-03, 4.48750012e-04, 1.54499998e-04, 3.79500016e-05, 
                                                                      8.69999985e-06, 1.75000002e-06] );
ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM_NS_14dB = np.array( [1.62663497e-02, 9.63644963e-03, 5.24469977e-03, 2.57815002e-03,
                                                                           1.11730001e-03, 4.17100004e-04, 1.30000000e-04, 3.18499988e-05,
                                                                           7.09999995e-06, 1.05000004e-06] );


ANN_REF_BER  = np.array([2*1e-2,1.2*1e-2,7.0*1e-3,3.5*1e-3,2.0*1e-3,1*1e-3,5*1e-4,2.5*1e-4,1.5*1e-4])
ANN_REF2_BER = np.array([2*1e-2,1.1*1e-2,6.5*1e-3,3.2*1e-3,1.8*1e-3,8*1e-4,3*1e-4,1.3*1e-4,5.0*1e-5])
SNN_REF_BER  = np.array([2*1e-2,1.2*1e-2,7.0*1e-3,3.5*1e-3,2.0*1e-3,1*1e-3,5*1e-4,2.0*1e-4,8.5*1e-5])

BER_ref = np.array([2.8*10**(-2),1.8*10**(-2),10**(-2),6.5*10**(-3),3.5*10**(-3),
                    2*10**(-3),10**(-3),6*10**(-4),3*10**(-4),1.8*10**(-4)])
plt.figure()
plt.semilogy(sigma,Torch_LMMSE_BER,label='Torch MMSE Simulation',marker='o')
plt.semilogy(sigma,DFE_BER,label='DFE',marker=10)

plt.semilogy(sigma,ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM,label='ANN DFE 5E 2000B LR1e-3 BS200000 1UspB SFB TfI ADAM',marker='X')
plt.semilogy(sigma,ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM_NS,label='ANN DFE 5E 2000B LR1e-3 BS200000 1UspB SFB TfI ADAM No Scheduler',marker='X')
plt.semilogy(sigma,ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM_NS_14dB,label='ANN DFE 5E 2000B LR1e-3 BS200000 1UspB SFB TfI ADAM No Scheduler 17dB Operationpoint',marker='X')

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

