import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import data_loader as dl
from PIL import Image
from model import BasicBlock, preEncoder, Encoder, Decoder
from torch.utils.data import DataLoader, Dataset, random_split

epochs = 30
D_Losses = []
G_Losses = []
count = 0
batch_size = 4
learning_rate = 0.0002
lmbda = 0

#data_path will different because this model is trained on different system
data_path = {'train' : "data/train/", 'test' : "data/test/"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform_img = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]
                  )

inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])


data = dl.ImageDataset(data_path, transform = transform_img)

train_size = int(0.9 * len(data))
test_size = len(data) - train_size

train_data, validation_data = random_split(data, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size = batch_size, shuffle=True)

pre_encoder = preEncoder(BasicBlock).to(device)
encoder = Encoder(BasicBlock).to(device)
decoder = Decoder(BasicBlock).to(device)
discriminator = Discriminator(BasicBlock).to(device)

ce_loss = nn.CrossEntropyLoss()
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

pre_encoder_optim = optim.Adam(pre_encoder.parameters(), lr = 0.0002, weight_decay=0)
encoder_optim = optim.Adam(encoder.parameters(), lr = 0.0002, weight_decay=0)
decoder_optim = optim.Adam(decoder.parameters(), lr = 0.0002, weight_decay=0)
discriminator_optim = optim.Adam(discriminator.parameters(), lr = learning_rate, weight_decay=lmbda)


pre_encoder = pre_encoder.train()
encoder = encoder.train()
decoder = decoder.train()
discriminator = discriminator.train()


for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        
        image, real_image = batch
        image = image.to(device)
        real_image = real_image.to(device)
        real_label = torch.ones([image.shape[0]]).long().to(device)
        fake_label = torch.zeros([image.shape[0]]).long().to(device)
        
        pre_encoder_optim.zero_grad()
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        
        x = pre_encoder(image)
        x = encoder(x)
        G_x = decoder(x)
        for i in range(2):
            discriminator_optim.zero_grad()
            D_x = discriminator(real_image)
            G_D_x = discriminator(G_x.detach())

            real_loss = ce_loss(D_x, real_label)
            fake_loss = ce_loss(G_D_x, fake_label)
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            discriminator_optim.step()
        
        G_D_x = discriminator(G_x)
        gen_real_loss = ce_loss(G_D_x, real_label)
        abs_loss = l1_loss(G_x, real_image)
        se_loss = mse_loss(G_x, real_image)
        gen_loss = gen_real_loss + abs_loss + se_loss
        gen_loss.backward()
        
        pre_encoder_optim.step()
        encoder_optim.step()
        decoder_optim.step()
        
        if count == 0:
            print("Epoch [{}/{}] after iteration {}: \n".format((epoch+1), epochs, count))
            imgs = [inv_normalize(image.detach()[0, :, :, :]).cpu().numpy().squeeze().transpose(1, 2, 0),
                    inv_normalize(G_x.detach()[0, :, :, :]).cpu().numpy().squeeze().transpose(1, 2, 0)]
            lab = ['Ground Truth', 'Generated']
            fig=plt.figure(figsize=(10, 10))
            for i in range(1, 3):
                fig.add_subplot(1, 2, i)
                plt.imshow(imgs[i-1])
                plt.title(lab[i-1])
            plt.show()
        count += 1

    D_Losses.append(disc_loss.item())
    G_Losses.append(abs_loss.item())
    print("Epoch [{}/{}] after iteration {}: \n".format((epoch+1), epochs, count))
    imgs = [inv_normalize(image.detach()[0, :, :, :]).cpu().numpy().squeeze().transpose(1, 2, 0),
            inv_normalize(G_x.detach()[0, :, :, :]).cpu().numpy().squeeze().transpose(1, 2, 0)]
    lab = ['Ground Truth', 'Generated']
    fig=plt.figure(figsize=(10, 10))
    for i in range(1, 3):
        fig.add_subplot(1, 2, i)
        plt.imshow(imgs[i-1])
        plt.title(lab[i-1])
    plt.show()
    
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_Losses,label="G")
plt.plot(D_Losses,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#for testing Images

count = 0
for i, batch in enumerate(validation_loader):
    image, real_image = batch
    x = pre_encoder(image.to(device))
    x = encoder(x)
    x = decoder(x)
    imgs = [inv_normalize(image.detach()[0, :, :, :]).cpu().numpy().squeeze().transpose(1, 2, 0),
            inv_normalize(x.detach()[0, :, :, :]).cpu().numpy().squeeze().transpose(1, 2, 0),
            inv_normalize(real_image[0, :, :, :]).cpu().numpy().squeeze().transpose(1, 2, 0)]
    lab = ['Ground Truth', 'Generated', 'Original Image']
    fig=plt.figure(figsize=(10, 10))
    for i in range(1, 4):
        fig.add_subplot(1, 3, i)
        plt.imshow(imgs[i-1], cmap = 'gray')
        plt.title(lab[i-1])
    plt.show()
    if count == 20:
        break
    count += 1
