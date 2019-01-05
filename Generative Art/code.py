import torch
from torch.autograd import Variable
import torch.nn.functional as F

from time import time
import imageio

from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import seaborn as sns
%matplotlib inline

from google.colab import files


if torch.cuda.is_available():
  print('gpu available')


def plot_art(x, y, num_groups, num_points):
    plt.cla()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Gen Art', fontsize=30)
    ax.set_xlabel('x', fontsize=24)
    ax.set_ylabel('y', fontsize=24)
    ax.set_xlim(-1, width+1)
    ax.set_ylim(-1, width+1)
    
    cmap = sns.color_palette('husl', n_colors=num_groups)  # a list of RGB tuples
    for g in range(num_groups):
        ax.scatter(x.data.cpu().numpy()[0][g*num_points:(g+1)*num_points],
                   y.data.cpu().numpy()[0][g*num_points:(g+1)*num_points],
                   color = cmap[g],
                   alpha=0.9
                  )

    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def d_slice(dist, i, j, length):
    return torch.narrow(torch.narrow(dist, 0, i, length), 1, j, length)


class Net(torch.nn.Module):
    def __init__(self, n_points, n_ran):
        super(Net, self).__init__()
        self.n_ran = n_ran
        self.x = torch.nn.Parameter(n_ran*torch.rand(1, n_points))
        self.y = torch.nn.Parameter(n_ran*torch.rand(1, n_points))

    def forward(self):
        x = torch.clamp(self.x, 0, self.n_ran)
        y = torch.clamp(self.y, 0, self.n_ran)
        return x, y
    
    def dist_mat(self):
        d = self.x.shape[1]
        dist = torch.zeros(d, d)

        for i in range(d):
            for j in range(i+1, d):
                dist[i, j] = torch.sqrt( (self.x[0][i] - self.x[0][j])**2 + (self.y[0][i] - self.y[0][j])**2 )
        return dist


torch.manual_seed(101)    # reproducible

width = 10
num_points = 5
num_groups = 6
net = Net(num_points*num_groups, width).cuda()

opt = torch.optim.SGD(net.parameters(), lr=.001, momentum=0.9)

fig, ax = plt.subplots(figsize=(8,8))
my_images = []

for s in range(150):
    opt.zero_grad()
    
    x, y = net()
    z = 0
    dist = net.dist_mat()
    for i in range(0, num_points*num_groups, num_points):
        for j in range(i, num_points*num_groups, num_points):
            if i == j:
                # be close to it's own group
                z += num_points*num_groups*torch.sum( torch.abs(d_slice(dist=dist, i=i, j=j, length=num_points)-1) )
            else:
                # move away from other groups
                z -= (1/((num_points)))*torch.sum( d_slice(dist=dist, i=i, j=j, length=num_points) )
    
    z.backward() # Calculate gradients
    opt.step()
#     print(x)
    
    if s % 1 == 0:
        print('epoch = ' + str(s), end='\r' )
        image = plot_art(x, y, num_groups, num_points)
        my_images.append(image)

name = 'gen_art_11b_' + str(time()) + '.gif'
imageio.mimsave(name, my_images, fps=10)
files.download(name)
