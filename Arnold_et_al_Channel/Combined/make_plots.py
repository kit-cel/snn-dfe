import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

Torch_LMMSE_BER = np.array( [0.0212725, 0.01411275, 0.00906   , 0.00546375, 0.00323   , 
                             0.0018465, 0.001068  , 0.00054925, 0.0002945 , 0.000167  ] );

#############################################################################################################################################

#TX Sym feedback
DFE_BER = np.array( [0.01898833, 0.01266333, 0.00809167, 0.00473, 0.00286667, 0.001685  , 0.00097167, 0.00052167, 0.00032   , 0.00017667] );


#############################################################################################################################################

ANN_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM_NS = np.array([1.69992503e-02, 1.05147501e-02, 5.88029996e-03, 3.12720006e-03,
                                                             1.49259996e-03, 6.12650008e-04, 2.07649995e-04, 7.22499972e-05,
                                                             2.20500006e-05, 3.85000021e-06]);
ANN_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM_NS_14dB = np.array([1.74515508e-02, 1.05944499e-02, 5.90205006e-03, 2.98424996e-03,
                                                                  1.32990000e-03, 5.30150020e-04, 1.64800003e-04, 4.34499998e-05,
                                                                  7.90000013e-06, 1.15000000e-06]);

#############################################################################################################################################

ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM_NS = np.array( [1.60438996e-02, 9.50669963e-03, 5.26979985e-03, 2.59819999e-03,
                                                                      1.18080003e-03, 4.48750012e-04, 1.54499998e-04, 3.79500016e-05, 
                                                                      8.69999985e-06, 1.75000002e-06] );
ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM_NS_14dB = np.array( [1.62663497e-02, 9.63644963e-03, 5.24469977e-03, 2.57815002e-03,
                                                                           1.11730001e-03, 4.17100004e-04, 1.30000000e-04, 3.18499988e-05,
                                                                           7.09999995e-06, 1.05000004e-06] );


#############################################################################################################################################

SNN_MMSE_BER_5E_2000B_BS200000_1US_TfI_ADAM_MAX_NVA_NS = np.array( [1.58424508e-02, 9.37190000e-03, 5.14844991e-03, 2.54744990e-03,
                                                                    1.14419998e-03, 4.65399993e-04, 1.82549993e-04, 6.44500033e-05,
                                                                    2.29500001e-05, 1.18999997e-05] );
SNN_MMSE_BER_5E_2000B_BS200000_1US_TfI_ADAM_MAX_NVA_NS_14dB = np.array( [1.59850493e-02, 9.44800023e-03, 5.13949990e-03, 2.56250007e-03,
                                                                         1.12775003e-03, 4.43149998e-04, 1.43450001e-04, 4.13000016e-05,
                                                                         9.20000002e-06, 1.69999998e-06] );

#############################################################################################################################################

SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_NVA = np.array( [1.57389995e-02, 9.27099958e-03, 4.85600019e-03, 2.46199989e-03,
                                                                    1.15100003e-03, 4.36000002e-04, 1.40999997e-04, 5.19999994e-05,    
                                                                    1.29999999e-05, 4.99999987e-06] );
SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_NVA_14dB = np.array( [1.57458000e-02, 9.16939974e-03, 4.93030017e-03, 2.37015006e-03,
                                                                         1.02590001e-03, 3.84699990e-04, 1.18199998e-04, 3.12500015e-05,
                                                                         6.14999999e-06, 1.35000005e-06] );

#############################################################################################################################################
ANN_REF_BER  = np.array([2*1e-2,1.2*1e-2,7.0*1e-3,3.5*1e-3,2.0*1e-3,1*1e-3,5*1e-4,2.5*1e-4,1.5*1e-4])
ANN_REF2_BER = np.array([2*1e-2,1.1*1e-2,6.5*1e-3,3.2*1e-3,1.8*1e-3,8*1e-4,3*1e-4,1.3*1e-4,5.0*1e-5])
SNN_REF_BER  = np.array([2*1e-2,1.2*1e-2,7.0*1e-3,3.5*1e-3,2.0*1e-3,1*1e-3,5*1e-4,2.0*1e-4,8.5*1e-5])

BER_ref = np.array([2.8*10**(-2),1.8*10**(-2),10**(-2),6.5*10**(-3),3.5*10**(-3),
                    2*10**(-3),10**(-3),6*10**(-4),3*10**(-4),1.8*10**(-4)])

#############################################################################################################################################

plt.figure()
plt.semilogy(sigma,Torch_LMMSE_BER,label='Torch MMSE Simulation',marker=3)

plt.semilogy(sigma,DFE_BER,label='DFE',marker=4)

plt.semilogy(sigma,ANN_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM_NS,label='ANN MMSE 5E 2000B LR1e-3 BS200000 1UspB TfI ADAM No Sceduler',marker=5)
plt.semilogy(sigma,ANN_BER_5E_2000B_LR1e_3_BS200000_1US_TfI_ADAM_NS_14dB,label='ANN MMSE 5E 2000B LR1e-3 BS200000 1UspB TfI ADAM No Sceduler 17 dB Operationpoint',marker=5)

plt.semilogy(sigma,ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM_NS,label='ANN DFE 5E 2000B LR1e-3 BS200000 1UspB SFB TfI ADAM No Scheduler',marker=6)
plt.semilogy(sigma,ANN_DFE_BER_5E_2000B_LR1e_3_BS200000_1US_SFB_TfI_ADAM_NS_14dB,label='ANN DFE 5E 2000B LR1e-3 BS200000 1UspB SFB TfI ADAM No Scheduler 17dB Operationpoint',marker=6)

plt.semilogy(sigma,SNN_MMSE_BER_5E_2000B_BS200000_1US_TfI_ADAM_MAX_NVA_NS, label='Arnold et.al SNN 5 Epochs 2000 Batch Batchsize 200000 1 Updatestep LR 1e-3 Train from Init ADAM MAX Decoding No varring alpha No scheduler',marker=7);
plt.semilogy(sigma,SNN_MMSE_BER_5E_2000B_BS200000_1US_TfI_ADAM_MAX_NVA_NS_14dB, label='Arnold et.al SNN 5 Epochs 2000 Batch Batchsize 200000 1 Updatestep LR 1e-3 Train from Init ADAM MAX Decoding No varring alpha No scheduler 17dB Operationpoint',marker=7);

plt.semilogy(sigma,SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_NVA, label='Class feedback SNN DFE 5 Epochs 2000 Batch Batchsize 200000 1 Updatestep LR 1e-3 Train from Init ADAM SUM Decoding No variing alpha',marker='X');
plt.semilogy(sigma,SNN_DFE_BER_5E_2000B_BS200000_1US_CFB_TfI_ADAM_SUM_NVA_14dB, label='Class feedback SNN DFE 5 Epochs 2000 Batch Batchsize 200000 1 Updatestep LR 1e-3 Train from Init ADAM SUM Decoding No variing alpha 17dB Operationpoint',marker='X');


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

