
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import *
from EdDSA_dataset import *
from torch.autograd import Variable
import os

class Train:
    def __init__(self) :


        self.size_ = 1000 # Image size
        self.batch_size = 64  # Batch size

        # Model
        self.z_size = 100
        self.generator_layer_size = [256, 512, 1024]
        self.discriminator_layer_size = [1024, 512, 256]

        # Training
        self.epochs = 30  # Train epochs
        self.learning_rate = 1e-4
        self.class_num = 16
        self.device="cpu"

        self.log_path = "logs"


                    # Define generator
        self.generator = Generator(self.generator_layer_size, self.z_size, self.size_, self.class_num).to(self.device)
        # Define discriminator
        self.discriminator = Discriminator(self.discriminator_layer_size, self.size_, self.class_num).to(self.device)

        self.criterion = nn.BCELoss()


        # Optimizer
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

        
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def discriminator_train_step(self, real_traces, labels):
        self.d_optimizer.zero_grad()

        # train with real images
        real_validity = self.discriminator(real_traces, labels)
        real_loss = self.criterion(real_validity, Variable(torch.ones(self.batch_size)).to(self.device))
        
        # train with fake images
        z = Variable(torch.randn(self.batch_size, self.z_size)).to(self.device)
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, self.batch_size))).to(self.device)
        fake_traces = self.generator(z, fake_labels)
        fake_validity = self.discriminator(fake_traces, fake_labels)
        fake_loss = self.criterion(fake_validity, Variable(torch.zeros(self.batch_size)).to(self.device))
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()


        return d_loss.item()
    

    def generator_train_step(self,):
        self.g_optimizer.zero_grad()
        z = Variable(torch.randn(self.batch_size, self.z_size)).to(self.device)
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, self.batch_size))).to(self.device)
        fake_images = self.generator(z, fake_labels)
        validity = self.discriminator(fake_images, fake_labels)
        g_loss = self.criterion(validity, Variable(torch.ones(self.batch_size)).to(self.device))
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss.item()

    def train_model(self):
        # Data
        train_data_path = 'MachineLearningBasedSideChannelAttackonEdDSA/databaseEdDSA.h5' # Path of data
        valid_data_path = './data/Fashion MNIST/fashion-mnist_train.csv' # Path of data
        print('Train data path:', train_data_path)
        print('Valid data path:', valid_data_path)


        # transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=(0.5,), std=(0.5,))
        # ])

                    
        dataset = EdDSA(train_data_path, self.size_, transform=None)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


        max_iter = len(data_loader)



        for epoch in range(self.epochs):

            self.generator.train()
            self.discriminator.train()
            total_iters = 0
            
            print('Starting epoch {}...'.format(epoch+1))
            
            for i, (traces, labels) in enumerate(data_loader):
                total_iters += 1
                
                # Train data
                real_traces = Variable(traces).to(self.device)
                labels = Variable(labels).to(self.device)
                

                
                # Train discriminator
                d_loss = self.discriminator_train_step(real_traces, labels)
                
                # Train generator
                g_loss = self.generator_train_step()
            
                if i % 50 == 0:
                    print("Epoch: " + str(epoch + 1) + "/" + str(self.epochs)
                          + "\titer: " + str(i) + "/" + str(max_iter)
                        + "\ttotal_iters: " + str(total_iters)
                        + "\td_loss:" + str(round(d_loss, 4))
                        + "\tg_loss:" + str(round(g_loss, 4))
                        )

            if (epoch + 1) % 5 == 0:
                torch.save(self.generator.state_dict(), os.path.join(self.log_path, 'gen.pth'))
                torch.save(self.discriminator.state_dict(), os.path.join(self.log_path, 'dis.pth'))



            # # Set generator eval
            # generator.eval()
            
            # print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
            
            # # Building z 
            # z = Variable(torch.randn(self.class_num-1, self.z_size)).to(self.device)
            
            # # Labels 0 ~ 8
            # labels = Variable(torch.LongTensor(np.arange(class_num-1))).to(device)
            
            # # Generating images
            # sample_images = generator(z, labels).unsqueeze(1).data.cpu()
            
            # # Show images
            # grid = make_grid(sample_images, nrow=3, normalize=True).permute(1,2,0).numpy()
            # plt.imshow(grid)
            # plt.show()


if __name__=="__main__":
    t = Train()
    t.train_model()