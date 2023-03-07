import torch

def encode(x):
    x_gray = x ^ (x >> 1);
    return x_gray;

def decode(x):
    mask = x;
    while(mask.any()):
        mask = mask >> 1;
        x = x ^ mask;
    return x;




#x = torch.tensor([3]);
#x = torch.randint(16,size=(20,1));
#print(x);
#y = encode(x);
#print(y);
#z = decode(y);
#print(z);
#if(torch.sum(torch.abs(x-z))==0):
#    print("Success");
#else:
#    print("Graycoder doesn't work properly!");

