import torch
import pickle

import bit                  #requires bit.py

def midrise(x,w,x_max):
    Delta_x = x_max / (2**(w-1));
    xh_uniform_midrise = torch.sign(x)*(torch.floor(torch.abs(x)/Delta_x)+0.5);  #Select class from +/- 2**w-1/2
    xh_uniform_midrise = xh_uniform_midrise + (2**w-1)/2;               #Move to classes from 0-2**w-1
    xh_uniform_midrise = torch.clip(xh_uniform_midrise,0,2**w-1);          #Clip to 2**w-1
    return Delta_x,xh_uniform_midrise.int()

def midtread_binary_unipolar(x,w,x_max,device):
    bitmap = bit.BIT_MAPPER(w,device)

    x = torch.clip(x,0,x_max)
    Delta_x = x_max / (2**(w));
    xh_uniform_midtread = torch.floor(torch.abs(x)/Delta_x+0.5);   #Select class from +/- 2**w-1/2
    xh_uniform_midtread = torch.clip(xh_uniform_midtread,0,2**w-1);        #Clip to 2**w-1
    xh_uniform_midtread = torch.ceil(xh_uniform_midtread).int();                 #Round up
    binary = bitmap.demap(xh_uniform_midtread)
    return Delta_x,binary

class Lloyd_Max():
    def __init__(self,w,device,codebook=[]):
        if(w > 0):
            self.__w = w
        else:
            print("Wordlenght shall be larger than 0");
            quit()

        self.__device = device

        if(len(codebook) > 0):
            if(len(codebook)==2**w):
                self.__codebook = torch.tensor(codebook,device=self.__device)
            else:
                print("2**w and the size of the codebook have to match!")
                quit()
        else:
            self.__codebook =  torch.linspace(-1.1,1.1,2**w).reshape((2**w,1)).to(self.__device)    #Quanizer operates in range of -1 to 1
                                                                                                    #initial codebook slightly larger so that
                                                                                                    #the LMQ can converge to the outer points

    def optimize(self,x):
        x = x / torch.max(x)                                                #Scale to -1 - +1
        #large number for inifity
        lm_inf = 10*torch.max(torch.abs(x))

        codebook = self.__codebook.clone().float()                          #Create copy to work with

        codebooks = [codebook]                                              #Store inital codebook

        iter = 0

        maxclass = -1                                                       #Initialize storage for class with the most hits
        zerocnt = 100                                                       #Initialize storage for the number of classes without hits

        while True:
            thresholds = [-lm_inf] + [0.5*(codebook[t]+codebook[t+1]) for t in range(codebook.shape[0]-1)] + [lm_inf]   #Calculate optimal decision thresholds

            old_cb = codebook.clone()                                                               #Copy codebook to old codebook so its not overwritten
            sbuf = torch.zeros((codebook.shape))                                                    #Create space to store the number of hits
            for k in range(codebook.shape[0]):                                                      #For all classes...
                # new codebook center
                # find those samples that are in the quantization interval (indicator function)
                samples = x[(x >= thresholds[k]) & (x < thresholds[k+1])]                           #   find all samples which belong to class k
                if len(samples)==0:                                                                 #   if its a class without hits...
                    if(zerocnt == 1):                                                               #       if only one class is left...
                        if( k < maxclass ):                                                         #           if the class with most hits is above class k
                            codebook[k::] = codebook[k::] + 0.1 * torch.abs(codebook[k::]);         #               increase all consecutive class centers by 10%
                        else:                                                                       #           else
                            codebook[0:k:] = codebook[0:k:] - 0.1 * torch.abs(codebook[0:k:]);      #               decrease all previous class centers by 10%
                    else:                                                                           #       if more classes are zeros...
                        if(k == 0):                                                                 #           first class is zero...
                            codebook[k] = codebook[k+1]                                             #               first class is now the second class plus some variation
                            codebook[k] -= codebook[k] * 0.1 * torch.randn(1).to(self.__device)     #               
                        elif(k == codebook.shape[0]-1):                                             #           last class is zero...
                            codebook[k] = codebook[k-1]                                             #               last class is now the pre last class plus some variation
                            codebook[k] += codebook[k] * 0.1 * torch.randn(1).to(self.__device)     #
                        else:                                                                       #           intermediate class is zero...
                            codebook[k] = (codebook[k+1] + codebook[k-1])/2                         #               class is now the average of the above and below class 
                            codebook[k] += codebook[k] * 0.1 * torch.randn(1).to(self.__device)     #               plus some variation
                else:                                                                               #   its a class with hits...
                    codebook[k] = torch.sum(samples)/samples.shape[0]                               #       new class center is the average of all samples beloning to the class

                sbuf[k] = len(samples)                                                              #store the number of hits in the hit buffer

            maxclass = torch.max(sbuf)                                                          #determine the class with the most hits
                                                                                                #(Since we have uniformly distributed symbols this class is responsible
                                                                                                # for the classes which dont get any hit...)
            zerocnt = codebook.shape[0]-torch.count_nonzero(sbuf)                               #Calculate the number of classes without hit

            iter += 1                                                                           #increase iteration counter

            codebooks.append(codebook.clone())                                                  #store codebook for codebook history

            if torch.max(torch.abs(torch.tensor([codebook[i] - old_cb[i] for i in range(codebook.shape[0])],device=self.__device))) < 1e-10:
                break                                                                           #If the classcenters haven't changed more than 1e-10 in the iteration
                                                                                                #abort the optimization
            if(iter > 2000):                                                                    #Ensure the optimization does not get stuck
                break                                                                           #
        self.__codebook = codebook

        # also return all intermediate codebooks for plotting evolution
        return codebooks,codebook

    def save_codebook(self,name):
        print("Start writing process...");
        with open(name+".pkl", 'wb') as picklefile:                                             #Store codebook
            pickle.dump(self.__codebook,picklefile);
        print("DONE!");

    def load_codebook(self,name):
        with open(name+".pkl", 'rb') as picklefile:                                             #Load codebook
            self.__codebook = pickle.load(picklefile);

    # carry out quantization based on codebook
    def quantize(self, x):
        x = x / torch.max(x)                                                                    #Scale to -1 - +1
        #large number for inifity
        lm_inf = 10*torch.max(torch.abs(x))

        thresholds = [-lm_inf] + [0.5*(self.__codebook[t]+self.__codebook[t+1]) for t in range(2**self.__w-1)] + [lm_inf]

        Ax = torch.zeros(x.shape,device=self.__device)
        xh = torch.zeros(x.shape,device=self.__device)

        for k in range(2**self.__w):
            # find those samples that are in the quantization interval (indicator function)
            idx = (x >= thresholds[k]) & (x < thresholds[k+1])
            Ax[idx] = k
            xh[idx] = self.__codebook[k]

        return Ax.int(),xh

    def give_codebook(self):
        return self.__codebook.clone()

###################################################################################################
#                               Testing                                                           #                                                              
###################################################################################################
#
#if torch.cuda.is_available():
#    DEVICE = torch.device("cuda")
#else:
#    DEVICE = torch.device("cpu")
#print("Using " + str(DEVICE) + " for training.");
#
#w = 2
#
#numbers = torch.linspace(-2,2,256)
#print(numbers)
#print(numbers.shape)
#quantized = midrise(numbers,8,2);
#print(quantized)
#
##numbers = torch.linspace(0,255,256)
#numbers = torch.randn(64,1)+1
#numbers = torch.hstack( (torch.randn(64,1)+0,numbers) )
#numbers = torch.hstack( (torch.randn(64,1)-1,numbers) )
#numbers = torch.hstack( (torch.randn(64,1)+3,numbers) )
#
#Q = Lloyd_Max(w,DEVICE)
##Q = Lloyd_Max(w,DEVICE,torch.linspace(-2,2,2**w))
##Q = Lloyd_Max(w,DEVICE,torch.arange(0,2**w,1))
#print(Q.give_codebook())
#_,codebook = Q.optimize(numbers)
#print(Q.give_codebook())
#A_i,x_i = Q.quantize(numbers)
#
#print(x_i)
#print(A_i)
#
#self.__Q.load_codebook(name)
#delta_x = self.__Q.give_codebook()

