import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

from model import CNN

SAVE_MODEL_PATH = "checkpoint/model.pth"

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor()
)

loaders = {
    'train': torch.utils.data.DataLoader(train_data,
                                        batch_size=100,
                                        shuffle=True,
                                        num_workers=1),
    'test': torch.utils.data.DataLoader(test_data,
                                       batch_size=100,
                                       shuffle=True,
                                       num_workers=1),
}

# Loss function and Optimizer
loss_func = nn.CrossEntropyLoss()
cnn = CNN()
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)

# Number of Epochs
num_epochs = 10

def train(num_epochs, cnn, loaders):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    cnn = cnn.to(device)
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
    
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
        
            pass
    
    
        pass
    torch.save(cnn.state_dict(), SAVE_MODEL_PATH)



def test():
    # Test the model
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    
    pass


if __name__ == "__main__":
    train(num_epochs, cnn, loaders)
    test()