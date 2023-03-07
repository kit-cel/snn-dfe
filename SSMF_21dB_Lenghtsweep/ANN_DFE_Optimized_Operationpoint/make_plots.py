import numpy as np
import matplotlib.pyplot as plt

opt_lenght = np.linspace(1,6,6)
lenght = np.linspace(1,10,10)

ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM = np.array([5.50000004e-06, 7.99999998e-06, 1.82500007e-05, 7.65000004e-05,
                                                                    1.12975005e-03, 1.42015005e-02, 2.25112494e-02, 2.75689997e-02,
                                                                    3.32412496e-02, 2.95887496e-02]);
ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NVA = np.array([4.34999993e-06, 7.30000011e-06, 1.75999994e-05, 8.16000029e-05,
                                                                        1.14784995e-03, 1.25244996e-02, 2.09027492e-02, 2.38512997e-02,
                                                                        2.81604007e-02, 2.96260007e-02]);
ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NVA_Opt = np.array([2.7000001e-06, 4.69999986e-06, 1.08499999e-05, 4.80999988e-05,
                                                                            0.00046935, 0.0153555]);


########################################################################################################################################

SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_MAX = np.array([3.60000013e-05, 4.19999997e-05, 5.99999985e-05, 2.03000003e-04,
                                                                        1.15300005e-03, 1.71600003e-02, 3.74470018e-02, 5.29440008e-02,
                                                                        6.30510002e-02, 6.91780001e-02]);

#SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_MAX_NVA = np.array([1.89999992e-05, 2.40000008e-05, 4.99999987e-05, 7.40000032e-05,
#                                                                            5.26999997e-04, 1.10930000e-02, 2.76779998e-02, 3.42120007e-02,
#                                                                            3.89709994e-02, 5.37250005e-02]);
SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_MAX_NVA = np.array([1.23500004e-05, 1.52000002e-05, 3.18499988e-05, 6.70500012e-05,
                                                                            5.16799977e-04, 1.07749999e-02, 2.71668993e-02, 3.38226482e-02,
                                                                            3.89678515e-02, 5.41325994e-02]);

########################################################################################################################################

ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM = np.array([9.49999958e-06, 9.49999958e-06, 1.84999997e-05, 4.99999966e-05,
                                                                   3.01500014e-04, 1.37809992e-02, 4.66527462e-02, 4.47447509e-02,
                                                                   5.02719998e-02, 5.40462494e-02]);
ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS = np.array([6.49999993e-06, 8.09999983e-06, 1.72000000e-05, 5.42499984e-05,
                                                                      3.24649998e-04, 1.36859501e-02, 4.06547002e-02, 4.48600017e-02,
                                                                      5.11852987e-02, 5.26759513e-02]);

ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS_Opt = np.array([3.05000003e-06, 4.74999979e-06, 1.09000002e-05, 4.01500001e-05,
                                                                          0.00029565, 0.01682055]);

########################################################################################################################################

SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_SUM_1_NVA = np.array([2.70000000e-05, 2.30000005e-05, 3.19999999e-05, 7.59999966e-05,
                                                                       2.24999996e-04, 6.87699998e-03, 2.86410004e-02, 3.93720008e-02,
                                                                       3.82470004e-02, 4.27579992e-02]);
SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_SUM_2_NVA = np.array([1.80000006e-05, 2.49999994e-05, 3.19999999e-05, 7.10000022e-05,
                                                                       2.46000011e-04, 6.92699989e-03, 2.85250004e-02, 3.95289995e-02,
                                                                       3.74020003e-02, 4.26100008e-02]);
SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_SUM_NVA = SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_SUM_1_NVA + SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_SUM_2_NVA
SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_SUM_NVA = SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_SUM_NVA / 2 

########################################################################################################################################

SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_SUM = np.array([7.00000019e-05, 5.70000011e-05, 8.90000010e-05, 2.09999998e-04,
                                                                       7.99999980e-04, 1.34429997e-02, 4.31340002e-02, 4.91760001e-02,
                                                                       5.27950004e-02, 5.40909991e-02]);

########################################################################################################################################

plt.figure()
plt.semilogy(lenght,ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM,label='ANN MMSE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM',marker=3)
plt.semilogy(lenght,ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NVA,label='ANN MMSE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No scheduler',marker=3)
plt.semilogy(opt_lenght,ANN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NVA_Opt,label='ANN MMSE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No scheduler Optimizer Operationpoint',marker=3)

plt.semilogy(lenght,ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM,label='ANN DFE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM',marker=4)
plt.semilogy(lenght,ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS,label='ANN DFE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No scheduler',marker=4)
plt.semilogy(opt_lenght,ANN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_NS_Opt,label='ANN DFE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM No scheduler Optimized Operationpoint',marker=4)

plt.semilogy(lenght,SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_MAX_NVA,label='SNN MMSE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM MAX Decoding No variing alpha No scheduler',marker='o')
#plt.semilogy(lenght,SNN_MMSE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_MAX,label='SNN MMSE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM MAX Decoding',marker='o')

plt.semilogy(lenght,SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_SUM_NVA,label='SNN DFE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM SUM Decoding No variing alpha No scheduler',marker='X')
#plt.semilogy(lenght,SNN_DFE_BER_5E_2000B_LR1e_3_BS_200000_1US_SFB_TfI_ADAM_SUM,label='SNN DFE 5E 2000B LR 1e-3 BS 200000 1US SFB TfI ADAM SUM Decoding',marker='X')

plt.title("4-PAM BER/Length Noisepower -21dB plot im/dd channel MMSE");
plt.xlabel("Fiberlenght in km");
plt.ylabel("BER");
plt.grid(which='both')
plt.legend();

plt.show()

