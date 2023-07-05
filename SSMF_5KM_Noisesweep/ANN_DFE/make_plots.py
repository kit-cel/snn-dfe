import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM = np.array([2.34619990e-02, 1.49707496e-02, 9.00949985e-03, 4.58999984e-03,
                                                                   2.24024989e-03, 9.17249918e-04, 3.39749991e-04, 1.08749990e-04,
                                                                   3.10000003e-05, 7.75000008e-06]);
ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS = np.array([2.37052999e-02, 1.47519000e-02, 8.41154996e-03, 4.43710014e-03,
                                                                      2.03240011e-03, 9.09650000e-04, 3.32249998e-04, 1.07899999e-04,
                                                                      3.34500000e-05, 6.79999994e-06]);
ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS_14dB = np.array([2.43450496e-02, 1.49376504e-02, 8.44809972e-03, 4.27789986e-03,
                                                                           1.94244995e-03, 7.52250024e-04, 2.59599998e-04, 7.21499964e-05,
                                                                           1.80999996e-05, 3.65000005e-06]);

plt.figure()

plt.semilogy(sigma,ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM,label='ANN DFE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM',marker=6)
plt.semilogy(sigma,ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS,label='ANN DFE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No Scheduler',marker=6)
plt.semilogy(sigma,ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS_14dB,label='ANN DFE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No Scheduler 17dB Operationpoint',marker=6)

plt.title("4-PAM BER/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("BER");
plt.grid(which='both')
plt.legend();

plt.show()

