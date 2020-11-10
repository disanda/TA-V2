import torch
import torchvision
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

#---------去除指定层---------
#这里需要去除net的前三层

# import copy
# dict1 = copy.copy(G1_state_dict)
# print(dict1.keys())
# dict1 = dict(netG1.named_parameters())
# dict2 = dict(netG2.named_parameters())

# print(type(dict1))
# print(type(dict2))

# keys = []
# dict3 = {}

# for i,j in dict1.items():
# 	if i.startswith('fc'):
# 		print(i)
# 		continue
# 	keys.append(i)

# dict3 = {k:dict1[k] for k in keys}
# print(dict3.keys())


from pro_gan_pytorch import  Encoder , Networks as net
netG1 = torch.nn.DataParallel(net.Generator(depth=9,latent_size=512)).to(device)# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
#netG1.load_state_dict(torch.load('D:\\AI-Lab\\PGGAN-TA\\result\\Step2_Training_EL2_GL2\\models\\G_model_ep0.pth',map_location=device))
#netG1.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device))
#netG1.load_state_dict(torch.load('D:\\AI-Lab\\PGGAN-TA\\result\\Step2_G-allLoss-allLoss_wwm\\models\\G_model_ep3.pth',map_location=device))
netG1.load_state_dict(torch.load('D:\\AI-Lab\\PGGAN-TA\\result\\Step2_G_wwm_allLoss\\models\\G_model_ep8.pth',map_location=device))
netEn = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
#netEn.load_state_dict(torch.load('./pre-model/D2E_std_L2_ep9.pth',map_location=device))
netEn.load_state_dict(torch.load('./pre-model/D_all_Loss_ep19.pth',map_location=device))

#--------------操作 y-> z -> x-------------
loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
from PIL import Image
def image_loader(image_name):
 image = Image.open(image_name).convert('RGB')
 image = image.resize((1024,1024))
 image = loader(image).unsqueeze(0)
 return image.to(torch.float)

name = 'wwm.png'
im1=image_loader('./real-img/'+name).to(device)
print(im1.mean())
print(im1.std())
im2 = im1*2-1 #norm
print(im2.mean())
print(im2.std())


with torch.no_grad():
	z = netEn(im2,height=8,alpha=1)
	z = z.squeeze(2).squeeze(2)
#del netEn

# with torch.no_grad():
# 		x = (netG1(z,depth=8,alpha=1)+1)/2
# 		x = torch.cat((im1,x))
# torchvision.utils.save_image(x,'./wwm/id-%s_real_all-Loss.png'%(name), nrow=2)
# print('done')

#------------ 编辑------------
seed =0 #默认20
set_seed(seed)
dim= 24
size = 18
gap =200 #默认10
z = z.squeeze(0)
print(z.shape)
#z = torch.randn(512)
z = z.repeat(size,1) # [16，512]
#temp = torch.linspace(-gap, gap, size)
temp = torch.linspace(-gap, gap, size)
z[:,dim] = z[:,dim]+temp.to(device)

with torch.no_grad():
	x = (netG1(z,depth=8,alpha=1)+1)/2
torchvision.utils.save_image(x,'./wwm/G-allLoss_D-allLoss-dim%d_-gap%d_ep8.png'%(dim,gap), nrow=9)
