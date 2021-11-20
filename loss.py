import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt

# Neural estimation of Renyi mutual information by Donsker-Vadrahan representation
# https://arxiv.org/pdf/2007.03814.pdf


class RenyiMutualInformation(nn.Module):

    def __init__(self, alpha, input_dim, output_dim, num_steps, lr):
        super(RenyiMutualInformation, self).__init__()

        self.alpha = alpha
        self.estimator = self.mlp([input_dim+output_dim, 4, 1])
        self.optimizer = torch.optim.Adam(
            [p for p in self.estimator.parameters() if p.requires_grad]
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
       

    def mlp(self, architecture):
        prev_neurones = architecture[0]
        net = []
        for neurones in architecture[1:]:
            net.append(nn.Linear(prev_neurones, neurones))
            net.append(nn.Tanh())
            prev_neurones = neurones

        return nn.Sequential(*net)            

    # input shape: x - [n_batch, dim_x], y - [n_batch, dim_y]
    def forward(self, x, y):
        
        losses = []

        for i in range(self.num_steps):

            pi = torch.randperm(x.shape[0])      

            f1 = (self.alpha - 1)*self.estimator(torch.cat((x, y), dim=1))
            f2 = self.alpha*self.estimator(torch.cat((x, y[pi]), dim=1))

            f1 = torch.log(torch.exp(f1).mean(dim=0))/(self.alpha - 1)
            f2 = torch.log(torch.exp(f2).mean(dim=0))/self.alpha  

            neg_loss = -(f1 - f2) # need to maximize, hence negative
            losses.append(-neg_loss)

            self.optimizer.zero_grad()
            neg_loss.backward()
            self.optimizer.step()
                        
        return -neg_loss, torch.tensor(losses)


       
if __name__ == "__main__":

    # plot 1, when x and y are independent
    x = torch.randn(128, 5)
    y = torch.randn(128, 5)
    criterion = RenyiMutualInformation(alpha=1.5, input_dim=x.shape[1], output_dim=y.shape[1], num_steps = 700, lr=0.01)
    out, losses = criterion(x,y)
    losses = losses.detach()
    plt.plot(losses)
    plt.show()

    # plot 2, when they are lineary dependent
    x = torch.randn(128, 5)
    y = 0.5*x + 1.0
    criterion = RenyiMutualInformation(alpha=1.5, input_dim=x.shape[1], output_dim=y.shape[1], num_steps = 700, lr=0.01)
    out, losses = criterion(x,y)
    losses = losses.detach()
    plt.plot(losses)
    plt.show()
