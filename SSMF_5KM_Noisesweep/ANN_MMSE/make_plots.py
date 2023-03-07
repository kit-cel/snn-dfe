import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS = np.array([2.44067498e-02, 1.53946504e-02, 1.01993000e-02, 6.09630020e-03,
                                                                       3.77140008e-03, 1.93875004e-03, 1.09619997e-03, 5.30000019e-04,
                                                                       2.26849996e-04, 7.74500004e-05]);
ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS_14dB = np.array([2.63186991e-02, 1.69010498e-02, 1.01857996e-02, 5.61645022e-03,
                                                                            2.85420008e-03, 1.28864998e-03, 5.06200013e-04, 1.77199996e-04,
                                                                            5.79500011e-05, 1.48999998e-05]);

plt.figure()

plt.semilogy(sigma,ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS,label='ANN MMSE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No Scheduler',marker=5)
plt.semilogy(sigma,ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS_14dB,label='ANN MMSE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No Scheduler 17dB Operationpoint',marker=5)

plt.title("4-PAM BER/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("BER");
plt.grid(which='both')
plt.legend();

plt.show()

