import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

Torch_LMMSE_BER = np.array( [0.0212725, 0.01411275, 0.00906   , 0.00546375, 0.00323   , 
                             0.0018465, 0.001068  , 0.00054925, 0.0002945 , 0.000167  ] );

#TX Sym feedback
DFE_BER = np.array( [0.01898833, 0.01266333, 0.00809167, 0.00473, 0.00286667, 0.001685  , 0.00097167, 0.00052167, 0.00032   , 0.00017667] );

ANN_MMSE_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM = np.array([1.70967504e-02, 1.07602496e-02, 6.14150008e-03, 3.22075002e-03,
                                                              1.46149995e-03, 5.97500009e-04, 2.01749994e-04, 5.67499992e-05,
                                                              9.99999975e-06, 1.75000002e-06]);

ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM = np.array( [1.60995007e-02, 9.46424976e-03, 5.12349978e-03, 2.60224994e-03,
                                                                   1.11925006e-03, 4.49999981e-04, 1.25249987e-04, 3.59999976e-05,
                                                                   8.25000025e-06, 7.49999981e-07] );

SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM = np.array( [1.69750005e-02, 1.03799999e-02, 5.77999949e-03, 3.09500005e-03,
                                                                1.60499997e-03, 7.24999933e-04, 3.42499954e-04, 1.52499997e-04,
                                                                7.24999933e-05, 3.49999958e-05] );
SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_12dB = np.array( [1.72024995e-02, 1.04024999e-02, 5.95999993e-03, 3.02249994e-03,
                                                                     1.43999979e-03, 6.84999954e-04, 2.54999986e-04, 7.49999890e-05,
                                                                     2.49999983e-05, 9.99999975e-06] );
SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_14dB = np.array( [1.71199992e-02, 1.05949998e-02, 5.91249950e-03, 3.03499997e-03,
                                                                     1.41999992e-03, 5.34999976e-04, 2.19999976e-04, 7.24999991e-05,
                                                                     9.99999975e-06, 2.49999994e-06] );

SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_NVA = np.array( [1.57389995e-02, 9.27099958e-03, 4.85600019e-03, 2.46199989e-03,
                                                                    1.15100003e-03, 4.36000002e-04, 1.40999997e-04, 5.19999994e-05,    
                                                                    1.29999999e-05, 4.99999987e-06] );
#SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_NVA_14dB = np.array( [1.58389993e-02, 9.16299969e-03, 5.06900018e-03, 2.37199990e-03,
#                                                                         1.07999996e-03, 3.61999992e-04, 1.30000000e-04, 3.50000009e-05,
#                                                                         9.99999975e-06, 0.00000000e+00] );
SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_NVA_14dB = np.array( [1.57458000e-02, 9.16939974e-03, 4.93030017e-03, 2.37015006e-03,
                                                                         1.02590001e-03, 3.84699990e-04, 1.18199998e-04, 3.12500015e-05,
                                                                         6.14999999e-06, 1.35000005e-06] );


SNN_MMSE_BER_5E_2000B_BS200000_1US_TfI_ADAM_MAX = np.array( [1.67909991e-02, 1.01039996e-02, 5.91599988e-03, 2.99700000e-03,
                                                             1.54500001e-03, 6.53999974e-04, 2.84999987e-04, 1.25999999e-04,
                                                             6.80000012e-05, 4.09999993e-05] );
SNN_MMSE_BER_5E_2000B_BS200000_1US_TfI_ADAM_MAX_14dB = np.array( [1.70540009e-02, 1.05959997e-02, 5.76200010e-03, 3.10300011e-03,
                                                                  1.44300004e-03, 5.70999982e-04, 2.11999999e-04, 7.69999970e-05,
                                                                  2.09999998e-05, 6.00000021e-06] );

ANN_REF_BER  = np.array([2*1e-2,1.2*1e-2,7.0*1e-3,3.5*1e-3,2.0*1e-3,1*1e-3,5*1e-4,2.5*1e-4,1.5*1e-4])
ANN_REF2_BER = np.array([2*1e-2,1.1*1e-2,6.5*1e-3,3.2*1e-3,1.8*1e-3,8*1e-4,3*1e-4,1.3*1e-4,5.0*1e-5])
SNN_REF_BER  = np.array([2*1e-2,1.2*1e-2,7.0*1e-3,3.5*1e-3,2.0*1e-3,1*1e-3,5*1e-4,2.0*1e-4,8.5*1e-5])

BER_ref = np.array([2.8*10**(-2),1.8*10**(-2),10**(-2),6.5*10**(-3),3.5*10**(-3),
                    2*10**(-3),10**(-3),6*10**(-4),3*10**(-4),1.8*10**(-4)])
plt.figure()
plt.semilogy(sigma,Torch_LMMSE_BER,label='Torch MMSE Simulation',marker=3)

plt.semilogy(sigma,DFE_BER,label='DFE',marker=3)

plt.semilogy(sigma,ANN_MMSE_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM,label='ANN MMSE 5E 2000B LR1e-3 BS200000 1UspB TfI ADAM',marker=4)

plt.semilogy(sigma,ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM,label='ANN DFE 5E 2000B LR1e-3 BS200000 1UspB SFB TfI ADAM',marker=4)

plt.semilogy(sigma,SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM, label='Class feedback SNN DFE 5 Epochs 2000 Batch Batchsize 200000 1 Updatestep LR 1e-3 Train from Init ADAM SUM Decoding',marker='X');
#plt.semilogy(sigma,SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_12dB, label='Class feedback SNN DFE 5 Epochs 2000 Batch Batchsize 200000 1 Updatestep LR 1e-3 Train from Init ADAM SUM Decoding 15dB Operationpoint',marker='X');
plt.semilogy(sigma,SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_14dB, label='Class feedback SNN DFE 5 Epochs 2000 Batch Batchsize 200000 1 Updatestep LR 1e-3 Train from Init ADAM SUM Decoding 17dB Operationpoint',marker='X');

plt.semilogy(sigma,SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_NVA, label='Class feedback SNN DFE 5 Epochs 2000 Batch Batchsize 200000 1 Updatestep LR 1e-3 Train from Init ADAM SUM Decoding No variing alpha',marker='X');
plt.semilogy(sigma,SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_NVA_14dB, label='Class feedback SNN DFE 5 Epochs 2000 Batch Batchsize 200000 1 Updatestep LR 1e-3 Train from Init ADAM SUM Decoding No variing alpha 17dB Operationpoint',marker='X');

plt.semilogy(sigma,SNN_MMSE_BER_5E_2000B_BS200000_1US_TfI_ADAM_MAX, label='Arnold et.al SNN 5 Epochs 2000 Batch Batchsize 200000 1 Updatestep LR 1e-3 Train from Init ADAM MAX Decoding',marker=5);
plt.semilogy(sigma,SNN_MMSE_BER_5E_2000B_BS200000_1US_TfI_ADAM_MAX_14dB, label='Arnold et.al SNN 5 Epochs 2000 Batch Batchsize 200000 1 Updatestep LR 1e-3 Train from Init ADAM MAX Decoding 17dB Operationpoint',marker=5);

plt.semilogy(sigma,BER_ref,label='Reference LMMSE Arnold et.al. MMSE',marker='o')
plt.semilogy(sigma[:-1:],ANN_REF_BER,label='Reference LMMSE Arnold et.al. ANN1',marker='o')
plt.semilogy(sigma[:-1:],ANN_REF2_BER,label='Reference LMMSE Arnold et.al. ANN2',marker='o')
plt.semilogy(sigma[:-1:],SNN_REF_BER,label='Reference LMMSE Arnold et.al. SNN',marker='o')
plt.title("4-PAM BER/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("BER");
plt.grid(which='both')
plt.legend();

plt.show()

