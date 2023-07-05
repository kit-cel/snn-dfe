import torch

class BIT_MAPPER:
    ###### class global ######

    ####### Functions ########
    def __init__(self, w, device):
        self.change(w,device);

    def change(self,w,device):
        if(w <= 0):
            print("w has to be larger than 0");
            quit();
        else:
            self.__wordlenght = w;

        self.__device = device

        self.__map_mat = torch.linspace(0,self.__wordlenght-1,self.__wordlenght).to(self.__device)
        self.__map_mat = 2**self.__map_mat
        #self.__map_mat = self.__map_mat.T

    def give_wordlenght(self):
        return self.__wordlenght;
    def give_device(self):
        return self.__device;

    def map(self,bits):
        numbers = self.__map_mat @ torch.flip(bits,dims=[1]).T
        return numbers
 
    def demap(self,numbers):
        orig_shape = numbers.shape
        numbers = numbers.flatten()
        bit_list = [(numbers >> shift_ind) & 1 for shift_ind in range(self.__wordlenght)] # little endian
        bit_list.reverse() # big endian
        bits = torch.zeros((self.__wordlenght,numbers.shape[0]),device=self.__device)
        for n in range(len(bit_list)):
            bits[n] = bit_list[n]
        bits = bits.T
    
        bits = bits.unflatten(0,(orig_shape))

        return bits
 
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
#wc = 2
#num = torch.tensor([0,1,2,3,1,1,3,0,2]).to(DEVICE)
#print(num)
#
#bm = BIT_MAPPER(wc,DEVICE)
#
#bits = bm.demap(num)
#print(bits)
#
#num = bm.map(bits)
#print(num)
#
#bits = torch.randint(0,2,size=(10,1)).float().flatten()
#bits = torch.randint(0,2,size=(5,2)).float()
#print(bits.flatten())
#test = bits.flatten()
#
#print(bits)
#print(bits.reshape((-1,2)))
#print(bits.reshape((-1,2)).T)
#
#bits = bits.reshape((2,-1)).T
#bits = bits.reshape((-1,2))
#num = bm.map(bits).int()
#print(num)
#bits = bm.demap(num)
#print(bits.flatten())
#if(torch.sum(torch.abs(bits.flatten()-test))!=0):
#    print("Failure!")
#else:
#    print("Success")
#
#
#print(bm.give_wordlenght())
#print(bm.give_device())
#
#bm.change(wc+1,DEVICE);
#print(bm.give_wordlenght())
#print(bm.give_device())
#
#bm.change(-1,DEVICE);
#bm.change(0,DEVICE);

