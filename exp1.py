import argparse
import torch
import torch.nn as nn
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

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
# Create an AE model
model = AE(latent_dim=128, conv_dims=[16, 32, 64, 128])
model = model.cuda()

# Define a loss function
criterion = nn.MSELoss()

# Define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Train the model
for epoch in range(args.epochs):
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, images)
        
        # Backward propagation
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, loss.item()))

    # Test the model
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            loss = 0
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                # Forward propagation
                outputs = model(images)
                
                # Calculate loss
                loss += criterion(outputs, images)
                
            print('Test loss: {:.4f}'.format(loss / len(test_loader)))

            # Save reconstructed images
            images = images[:16]
            outputs = model(images)

            images = images.permute(0, 2, 3, 1)
            outputs = outputs.permute(0, 2, 3, 1)
            images = images.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            plt.figure(figsize=(16, 4))
            for i in range(8):
                plt.subplot(2, 8, i + 1)
                plt.imshow(images[i])
                plt.axis('off')
                plt.subplot(2, 8, i + 1 + 8)
                plt.imshow(outputs[i])
                plt.axis('off')
            plt.savefig('reconstructed_images_epoch_{}.png'.format(epoch + 1))

# Save the model checkpoints
torch.save(model.state_dict(), 'AE_mse.ckpt')


