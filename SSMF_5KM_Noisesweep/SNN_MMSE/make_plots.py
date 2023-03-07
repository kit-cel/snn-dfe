import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_TfI_ADAM_MAX_noalpha_nosceduler = np.array([2.22980995e-02, 1.34347500e-02, 7.99894985e-03, 4.36295010e-03,
                                                                                       2.25095008e-03, 1.06529996e-03, 5.27700002e-04, 2.49600009e-04,
                                                                                       1.31499997e-04, 6.79000004e-05]);

SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_TfI_ADAM_MAX_noalpha_nosceduler_14dB = np.array([2.22693495e-02, 1.37985498e-02, 7.99554959e-03, 4.28130012e-03,
                                                                                            2.07949989e-03, 9.58000019e-04, 4.12199995e-04, 1.72300002e-04,
                                                                                            7.62499985e-05, 3.70999987e-05]);

plt.figure()

plt.semilogy(sigma,SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_TfI_ADAM_MAX_noalpha_nosceduler, label='Arnold et al. SNN 5 Epochs 2000 Batch LR 1e-3 Batchsize 200000 1 Update per Batch  Train from Init ADAM MAX Decoding No Variing Alpha No Sceduler',marker='X')
plt.semilogy(sigma,SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_TfI_ADAM_MAX_noalpha_nosceduler_14dB, label='Arnold et al. SNN 5 Epochs 2000 Batch LR 1e-3 Batchsize 200000 1 Update per Batch Train from Init ADAM MAX Decoding No Variing Alpha No Sceduler 17dB Operationpoint',marker='X')

plt.title("4-PAM BER/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("BER");
plt.grid(which='both')
plt.legend();

plt.show()

