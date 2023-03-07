import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

################################################################################################################################################

ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS = np.array([2.44067498e-02, 1.53946504e-02, 1.01993000e-02, 6.09630020e-03,
                                                                       3.77140008e-03, 1.93875004e-03, 1.09619997e-03, 5.30000019e-04,
                                                                       2.26849996e-04, 7.74500004e-05]);
ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS_14dB = np.array([2.63186991e-02, 1.69010498e-02, 1.01857996e-02, 5.61645022e-03,
                                                                            2.85420008e-03, 1.28864998e-03, 5.06200013e-04, 1.77199996e-04,
                                                                            5.79500011e-05, 1.48999998e-05]);

################################################################################################################################################

ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS = np.array([2.37052999e-02, 1.47519000e-02, 8.41154996e-03, 4.43710014e-03,
                                                                      2.03240011e-03, 9.09650000e-04, 3.32249998e-04, 1.07899999e-04,
                                                                      3.34500000e-05, 6.79999994e-06]);
ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS_14dB = np.array([2.43450496e-02, 1.49376504e-02, 8.44809972e-03, 4.27789986e-03,
                                                                           1.94244995e-03, 7.52250024e-04, 2.59599998e-04, 7.21499964e-05,
                                                                           1.80999996e-05, 3.65000005e-06]);

################################################################################################################################################

SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_TfI_ADAM_MAX_noalpha_nosceduler = np.array([2.22980995e-02, 1.34347500e-02, 7.99894985e-03, 4.36295010e-03,
                                                                                       2.25095008e-03, 1.06529996e-03, 5.27700002e-04, 2.49600009e-04,
                                                                                       1.31499997e-04, 6.79000004e-05]);
SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_TfI_ADAM_MAX_noalpha_nosceduler_14dB = np.array([2.22693495e-02, 1.37985498e-02, 7.99554959e-03, 4.28130012e-03,
                                                                                            2.07949989e-03, 9.58000019e-04, 4.12199995e-04, 1.72300002e-04,
                                                                                            7.62499985e-05, 3.70999987e-05]);

################################################################################################################################################

SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_CFB_TfI_ADAM_SUM_noalpha_nosceduler = np.array([2.06211992e-02, 1.21170497e-02, 6.49964996e-03, 3.04864999e-03,
                                                                                          1.40904996e-03, 6.29900023e-04, 2.21099996e-04, 1.04649997e-04,
                                                                                          5.07500008e-05, 2.46000000e-05]);
SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_CFB_TfI_ADAM_SUM_noalpha_nosceduler_14dB = np.array([2.12833006e-02, 1.24191996e-02, 6.47879997e-03, 3.00185010e-03,
                                                                                               1.22135004e-03, 4.27999999e-04, 1.26150000e-04, 3.11000003e-05,
                                                                                               5.89999991e-06, 1.95000007e-06]);

################################################################################################################################################

plt.figure()

plt.semilogy(sigma,ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS,label='ANN MMSE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No Scheduler',marker=5)
plt.semilogy(sigma,ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS_14dB,label='ANN MMSE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No Scheduler 17dB Operationpoint',marker=5)

plt.semilogy(sigma,ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS,label='ANN DFE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No Scheduler',marker=6)
plt.semilogy(sigma,ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS_14dB,label='ANN DFE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No Scheduler 17dB Operationpoint',marker=6)

plt.semilogy(sigma,SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_TfI_ADAM_MAX_noalpha_nosceduler, label='Arnold et al. SNN 5 Epochs 2000 Batch LR 1e-3 Batchsize 200000 1 Update per Batch  Train from Init ADAM MAX Decoding No Variing Alpha No Sceduler',marker=7)
plt.semilogy(sigma,SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_TfI_ADAM_MAX_noalpha_nosceduler_14dB, label='Arnold et al. SNN 5 Epochs 2000 Batch LR 1e-3 Batchsize 200000 1 Update per Batch Train from Init ADAM MAX Decoding No Variing Alpha No Sceduler 17dB Operationpoint',marker=7)

plt.semilogy(sigma,SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_CFB_TfI_ADAM_SUM_noalpha_nosceduler, label='Ternary DFE SNN 5 Epochs 2000 Batch LR 1e-3 Batchsize 200000 1 Update per Batch Classfeedback Train from Init ADAM SUM Decoding No Variing Alpha No Sceduler',marker='X')
plt.semilogy(sigma,SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_CFB_TfI_ADAM_SUM_noalpha_nosceduler_14dB, label='Ternary DFE SNN 5 Epochs 2000 Batch LR 1e-3 Batchsize 200000 1 Update per Batch Classfeedback Train from Init ADAM SUM Decoding No Variing Alpha No Sceduler 17dB Operationpoint',marker='X')

plt.title("4-PAM BER/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("BER");
plt.grid(which='both')
plt.legend();

plt.show()

