import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import AE

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--latent_dim', type=int, default=128)

args = parser.parse_args()

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

# Load STL-10 dataset
train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

# Get all the data
train_data = train_dataset.data
test_data = test_dataset.data

# reshape the data to one dimension
train_data = train_data.reshape(train_data.shape[0], -1)
test_data = test_data.reshape(test_data.shape[0], -1)

# subtract the mean
mu, sigma = train_data.mean(axis=0), train_data.std(axis=0)
train_data = (train_data - mu) / sigma
test_data = (test_data - mu) / sigma

train_data = torch.from_numpy(train_data).float().cuda()
test_data = torch.from_numpy(test_data).float().cuda()

# perform PCA
U, S, V = torch.svd(train_data)
train_latent = torch.matmul(train_data, V[:, :args.latent_dim])
test_latent = torch.matmul(test_data, V[:, :args.latent_dim])

# reconstruct the data
train_recon = torch.matmul(train_latent, V[:, :args.latent_dim].T)
test_recon = torch.matmul(test_latent, V[:, :args.latent_dim].T)

# reshape the data back to 3D
train_recon = train_recon.reshape(train_recon.shape[0], 3, 96, 96)
test_recon = test_recon.reshape(test_recon.shape[0], 3, 96, 96)

# define a 3x3 gaussian filter
gaussian_filter = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32).cuda()
# repeat the filter for 3 channels
gaussian_filter = gaussian_filter.repeat(3, 1, 1)

# visualize the reconstructed images
fig, axes = plt.subplots(5, 10, figsize=(10, 5))
# for i in range(10):
#     axes[0][i].imshow(train_data[i].reshape(3, 96, 96).permute(1, 2, 0).cpu().numpy())
#     axes[0][i].axis('off')
#     axes[1][i].imshow(train_recon[i].permute(1, 2, 0).cpu().numpy())
#     axes[1][i].axis('off')

#     diff_image = (train_data[i].reshape(3, 96, 96) - train_recon[i])
#     mse_image = (diff_image ** 2)
#     pse_image = F.conv2d(diff_image, gaussian_filter.unsqueeze(0), padding=1, stride=[1,1]) / 48
#     pse_image = (pse_image ** 2)
#     # rescale the images
#     mse_image = (mse_image - mse_image.min()) / (mse_image.max() - mse_image.min())
#     pse_image = (pse_image - pse_image.min()) / (pse_image.max() - pse_image.min())

#     axes[2][i].imshow((train_data[i].reshape(3, 96, 96) - train_recon[i]).permute(1, 2, 0).cpu().numpy())
#     axes[2][i].axis('off')
#     axes[3][i].imshow(mse_image.permute(1, 2, 0).cpu().numpy())
#     axes[3][i].axis('off')
#     axes[4][i].imshow(pse_image.permute(1, 2, 0).cpu().numpy())
#     axes[4][i].axis('off')

# plt.savefig('pca_train.png')

fig, axes = plt.subplots(5, 10, figsize=(10, 5))

for i in range(10):
    axes[0][i].imshow(test_data[i].reshape(3, 96, 96).permute(1, 2, 0).cpu().numpy())
    axes[0][i].axis('off')
    axes[1][i].imshow(test_recon[i].permute(1, 2, 0).cpu().numpy())
    axes[1][i].axis('off')

    diff_image = torch.abs(test_data[i].reshape(3, 96, 96) - test_recon[i])
    diff_image2 = diff_image.mean(axis = 0, keepdim= True)
    mse_image = (diff_image ** 2)
    mse_image = mse_image.mean(axis = 0, keepdim= True)
    pse_image = F.conv2d(diff_image, gaussian_filter.unsqueeze(0), padding=1, stride=[1,1]) / 16
    pse_image = (pse_image ** 2)
    # rescale the images use log scale
    diff_image2 = torch.log(diff_image2)
    mse_image = torch.log(mse_image)
    pse_image = torch.log(pse_image)
    diff_image2 = (diff_image2 - diff_image2.min()) / (diff_image2.max() - diff_image2.min())
    mse_image = (mse_image - mse_image.min()) / (mse_image.max() - mse_image.min())
    pse_image = (pse_image - pse_image.min()) / (pse_image.max() - pse_image.min())

    axes[2][i].imshow(diff_image2.permute(1, 2, 0).cpu().numpy())
    axes[2][i].axis('off')
    axes[3][i].imshow(mse_image.permute(1, 2, 0).cpu().numpy())
    axes[3][i].axis('off')
    axes[4][i].imshow(pse_image.permute(1, 2, 0).cpu().numpy())
    axes[4][i].axis('off')

plt.savefig('pca_test.png')




