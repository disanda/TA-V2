#这个版本只需要导入网络即可(不需要导入训练网络)，先已完成两个实验，第一个实验完成gt编码的比较，第二个实验完成G(z)的编码比较
#准备做 不同网络的比较，包括结构不同，weight不同的情况 (mnist中以上因素不同，区别不大)
#改进loss，拉近D(z)到原点的距离
import torch
import numpy as np
import os
import torchvision
from torch.autograd import Variable
import modules.network_dcgan as net

#device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------------path setting---------------
resultPath = "./result/DCGAN-Celeba-V2-Percp-MSE-realImg"
if not os.path.exists(resultPath):
    os.mkdir(resultPath)

resultPath1_1 = resultPath+"/imgs"
if not os.path.exists(resultPath1_1):
    os.mkdir(resultPath1_1)

resultPath1_2 = resultPath+"/models"
if not os.path.exists(resultPath1_2):
    os.mkdir(resultPath1_2)


#----------------test pre-model output-----------

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

G = net.Generator(input_dim=128, output_channels = 3, image_size=256, scale=16).to(device)
G.load_state_dict(torch.load('./premodel/celeba-dcgan/G_ep99_in128_out256_scale16.pth',map_location=device))
#D = net.Discriminator_SpectrualNorm(input_dim=128, input_channels = 3, image_size=256, scale=8).to(device)
#D.load_state_dict(torch.load('./premodel/celeba-dcgan/D_ep99_in128_out256_scale8.pth',map_location=device))

# E = net.D2E(input_dim=128, input_channels = 3, image_size=256, scale=8).to(device)
# toggle_grad(D1,False)
# toggle_grad(E,False)

# paraDict = dict(D.named_parameters()) #pre_model weight dict
# for i,j in E.named_parameters():
# 	if i in paraDict.keys():
# 		w = paraDict[i]
# 		j.copy_(w)
# 	else:
# 		print(i)
# toggle_grad(D2,True)
# del D1

E = net.D2E(input_dim=128, input_channels = 3, image_size=256, scale=8).to(device)
E.load_state_dict(torch.load('/_yucheng/TA-V2/result/DCGAN-Celeba-V1-Percp-MSE/models/E_model_ep19.pth',map_location=device))

# FC1 = net.FC_Map().to(device)
# FC2 = net.FC_Map().to(device)
# FC3 = net.FC_Map().to(device)
# import itertools
# fc_optimizer = torch.optim.Adam(itertools.chain(FC1.parameters(), FC2.parameters(),FC3.parameters()),lr=0.0001,betas=(0.6, 0.95),amsgrad=True)

#-------------load single image---------
loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

from PIL import Image
def image_loader(image_name):
 image = Image.open(image_name).convert('RGB')
 image = image.resize((256,256))
 image = loader(image).unsqueeze(0)
 return image.to(torch.float)

im=image_loader('./cxx.png').to(device)

#--------------training with generative image------------share weight: good result!------------step2:no share weight:
import lpips
optimizer = torch.optim.Adam(E.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8)
loss_l2 = torch.nn.MSELoss()
loss_kl = torch.nn.KLDivLoss() #衡量分布
loss_l1 = torch.nn.L1Loss() #稀疏
loss_percp = lpips.LPIPS(net='vgg').to(device)
loss_all=0
G.eval()
for epoch in range(20):
	for i in range(5001):
		z = torch.randn(63, 128).to(device)
		z = z.view(-1,128,1,1)
		with torch.no_grad():
			x = G(z)
		z2 = E(im)
		z2 = z2.view(1,128,1,1)
		z = torch.cat((z2,z))
		x = torch.cat((im,x))
		z_ = E(x.detach())
		with torch.no_grad():
			x_ = G(z_)
		optimizer.zero_grad()
		loss_1_1 = loss_l2(x,x_)
		loss_1_2 = loss_percp(x,x_).mean()#[-1,x] -> x
		loss_1 = loss_1_1 + loss_1_2
		loss_2 = loss_l2(z_.mean(),z.mean())
		loss_3 = loss_l2(z_.std(),z.std()) 
		loss_i = loss_1+0.01*loss_2+0.01*loss_3
		loss_i.backward()
		optimizer.step()
		loss_all +=loss_i.item()
		print('loss_all__:'+str(loss_all)+'--loss_i:'+str(loss_i.item())+'--loss_1_l2:'+str(loss_1_1.item())+'--loss_1_percp:'+str(loss_1_2.item()))
		print('loss_z_mean:'+str(loss_2.item())+'--loss_z_std:'+str(loss_3.item()))
		if i % 100 == 0:
			img = torch.cat((x[:8],x_[:8]))
			#img = torch.cat((x,x_))
			torchvision.utils.save_image(img, resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=8)
			with open(resultPath+'/Loss.txt', 'a+') as f:
				print('loss_all__:'+str(loss_all)+'--loss_i:'+str(loss_i.item())+'--loss_1_l2:'+str(loss_1_1.item())+'--loss_1_percp:'+str(loss_1_2.item()),file=f)
				print('loss_z_mean:'+str(loss_2.item())+'--loss_z_std:'+str(loss_3.item()),file=f)
			with open(resultPath+'/D_z.txt', 'a+') as f:
				print(str(epoch)+'-'+str(i)+'-'+'D_z:  '+str(z_[0,0:30].view(30))+'     D_z:    '+str(z_[0,30:60].view(30)),file=f)
				print('---------')
				print(str(epoch)+'-'+str(i)+'-'+'D_z_mean:  '+str(z_.mean())+'     D_z_std:    '+str(z_.std()),file=f)
				print('#########################')
			# with open(resultPath+'/fc_z.txt', 'a+') as f:
			# 	print(str(epoch)+'-'+str(i)+'-'+'fc_z1:  '+str(z_1[0,0:60].view(60)),file=f)
			# 	print('---------')
			# 	print(str(epoch)+'-'+str(i)+'-'+'fc_z2:  '+str(z_2[0,0:60].view(60)),file=f)
			# 	print('---------')
			# 	print(str(epoch)+'-'+str(i)+'-'+'fc_z3:  '+str(z_2[0,0:60].view(60)),file=f)
			# 	print('---------')
			# 	print(str(epoch)+'-'+str(i)+'-'+'fc_z:  '+str(z__[0,0:60].view(60)),file=f)
			# 	print('#########################')
	#if epoch%10==0 or epoch == 29:
	#torch.save(netG.state_dict(), resultPath1_2+'/G_model_ep%d.pth'%epoch)
	torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)



