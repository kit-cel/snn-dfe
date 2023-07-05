import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_CFB_TfI_ADAM_SUM_noalpha_nosceduler = np.array([2.06211992e-02, 1.21170497e-02, 6.49964996e-03, 3.04864999e-03,
                                                                                          1.40904996e-03, 6.29900023e-04, 2.21099996e-04, 1.04649997e-04,
                                                                                          5.07500008e-05, 2.46000000e-05]);
SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_CFB_TfI_ADAM_SUM_noalpha_nosceduler_14dB = np.array([2.12833006e-02, 1.24191996e-02, 6.47879997e-03, 3.00185010e-03,
                                                                                               1.22135004e-03, 4.27999999e-04, 1.26150000e-04, 3.11000003e-05,
                                                                                               5.89999991e-06, 1.95000007e-06]);

plt.figure()

plt.semilogy(sigma,SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_CFB_TfI_ADAM_SUM_noalpha_nosceduler, label='Ternary DFE SNN 5 Epochs 2000 Batch LR 1e-3 Batchsize 200000 1 Update per Batch Classfeedback Train from Init ADAM SUM Decoding No Variing Alpha No Sceduler',marker='X')
plt.semilogy(sigma,SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_CFB_TfI_ADAM_SUM_noalpha_nosceduler_14dB, label='Ternary DFE SNN 5 Epochs 2000 Batch LR 1e-3 Batchsize 200000 1 Update per Batch Classfeedback Train from Init ADAM SUM Decoding No Variing Alpha No Sceduler 17dB Operationpoint',marker='X')

plt.title("4-PAM BER/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("BER");
plt.grid(which='both')
plt.legend();

plt.show()

