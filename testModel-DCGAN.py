import torch
import torchvision
import numpy as np
import random
import modules.network_dcgan as net


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

G = net.Generator(input_dim=128, output_channels = 3, image_size=256, scale=16).to(device)
G.load_state_dict(torch.load('/Users/apple/Desktop/preTrained_Model/celeba-dcgan-premodel/G_ep99_in128_out256_scale16.pth',map_location=device))

#D = net.Discriminator_SpectrualNorm(input_dim=128, input_channels = 3, image_size=256, scale=8).to(device)
#D.load_state_dict(torch.load('./premodel/celeba-dcgan/D_ep99_in128_out256_scale8.pth',map_location=device))

E = net.D2E(input_dim=128, input_channels = 3, image_size=256, scale=8).to(device)
E.load_state_dict(torch.load('/Users/apple/Desktop/preTrained_Model/celeba-dcgan-premodel/E_model_ep19.pth',map_location=device))

#--------------操作原 model--------------
# seed = 0 #默认20
# set_seed(seed)
# dim= 4
# size = 81
# gap =20 #默认10
# z = torch.randn(128)
# temp = torch.linspace(-gap, gap, size)
# z = z.repeat(size,1) # [16，512]
# z[:,dim] = z[:,dim]+temp
# z = z.view(-1,128,1,1)


# with torch.no_grad():
# 	G.eval()
# 	x = G(z)
# 	torchvision.utils.save_image(x,'./dim%d-seed%d--gap%d_G_Dim512.png'%(dim,seed,gap), nrow=9)
# 	print('done')



#z2 = z2.to(device)
# with torch.no_grad():
# 	#x = (netG(z2,depth=8,alpha=0.2)+1)/2
# 	#x = netG2(z2,depth=8,alpha=0.2)
# 	#z_512 = netEn(x,height=8,alpha=1)
# 	#z_512 = z_512.squeeze(2).squeeze(2)
# 	x = (netG1(z_512,depth=8,alpha=1)+1)/2
# torchvision.utils.save_image(x,'./infer-img-2/seed%d-dim%d_-gap%d_G_inDim512_Dz.png'%(seed,dim1,gap), nrow=9)
# #torchvision.utils.save_image(x,'./infer-img/seed%d-dim1-%d_dim2-%d-gap%d_myTrainedG_inDim1024.png'%(seed,dim1,dim2,gap), nrow=8)
# #torchvision.utils.save_image(x_2,'./infer-img/seed%d-dim%d_-gap%d_RcG_inDim512.png'%(seed,dim,gap), nrow=8)
# print('done')


#--------------text z_---------------
# x = torch.randn(1,3,1024,1024)
# z = torch.randn(8,512).to(device)
# # z = z.repeat(1,1) # [16，512]
# with torch.no_grad():
# 	#x = netG1(z,depth=8,alpha=1)
# 	z_512 = netEn(im1,height=8,alpha=1)
# 	z_512 = z_512.squeeze(2).squeeze(2)
# 	#img = (torch.cat((x,x2))+1)/2
# 	#torchvision.utils.save_image(img,'./infer-img-3/seed%d-dim%d_-gap%d_RcG_inDim512.png'%(seed,dim,gap), nrow=8)
# 	#x2=(x2+1)/2
# 	#torchvision.utils.save_image(x2,'./infer-img-3/name-%s-no%d.png'%('celeba1558',0), nrow=1)
# 	# print(z_512[:50])
# 	# print(z_512.mean())
# 	# print(z_512.std())
# 	#print(z-z_512)
# #编码后的z不一样，同样的z有共同的解

# size = 17
# gap =400 #默认10
# dim1= 308
# temp = torch.linspace(-gap, gap, size)
# z_512 = z_512.squeeze(0)
# z_512 = z_512.repeat(size,1) # [16，512]
# z_512[:,dim1] = temp
# with torch.no_grad():
# 	x2 = netG1(z_512,depth=8,alpha=1)
# 	x2=(x2+1)/2
# 	torchvision.utils.save_image(x2,'./infer-img-3/seed-%s-dim%d_-gap%d_D2E.png'%('celeba1558',dim1,gap), nrow=9)

#-------------load single image---------
loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

from PIL import Image
def image_loader(image_name):
 image = Image.open(image_name).convert('RGB')
 image = image.resize((256,256))
 image = loader(image).unsqueeze(0)
 return image.to(torch.float)

im1=image_loader('/Users/apple/Desktop/real-image/wwm.png')

print(im1.mean())
print(im1.std())

im2 = im1*2-1 

print(im2.mean())
print(im2.std())

with torch.no_grad():
	G.eval()
	#z = torch.randn(10,128,1,1)
	#im1 = G(z)
	z_ = E(im1)
	x = G(z_)
	print(x.shape)
	print(im1.shape)
	img = torch.cat((im1,x))
	torchvision.utils.save_image(img,'./id%s.png'%(str('cxx')), nrow=2)
	print('done')

# seed =0 #默认20
# set_seed(seed)
# dim= 9
# size = 18
# gap =20 #默认10
# with torch.no_grad():
# 	z = netEn(im1,height=8,alpha=1)
# z = z.squeeze(2).squeeze(2)
# z = z.squeeze(0)
# print(z.shape)
# #z = torch.randn(512)
# z = z.repeat(size,1) # [16，512]
# #temp = torch.linspace(-gap, gap, size)
# temp = torch.linspace(-gap, gap, size)
# z[:,dim] = temp

# with torch.no_grad():
# 	x = (netG1(z,depth=8,alpha=1)+1)/2
# torchvision.utils.save_image(x,'./infer-step2-stdL2/ep_0-dim%d_-gap%d_yc-3.png'%(dim,gap), nrow=9)
