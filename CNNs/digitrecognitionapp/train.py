import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

from model import Model

SAVE_MODEL_PATH = "best_accuracy.pth"

def train():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
    train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train, batch_size=64)
    test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test, batch_size=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    model = Model().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training
    num_epochs = 30
    best_eval_acc = 0
    validation_interval = 200
    start = time()

    for epoch in range(num_epochs):
        avg_loss = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad() # zero the parameter gradients
            preds = model(imgs) # forward pass
            loss = criterion(preds, labels) # calculate loss
            loss.backward() # backpropagation
            optimizer.step() # optimization
            
            # print statistics
            avg_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i+1}], Loss: {avg_loss / 100:.3f}")
                avg_loss = 0
            
            # validation
            if (i+1) % validation_interval == 0: # perform validation every 200 batch
                model.eval()
                correct, total = 0, 0
                for i, (imgs, labels) in enumerate(test_loader):
                    imgs, labels = imgs.to(device), labels.to(device)
                    with torch.no_grad():
                        preds = model(imgs)
                    preds = torch.argmax(preds, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.shape[0]
                accuracy = correct / total
                print(f"eval acc: {accuracy:.4f}")

                # save model
                if accuracy > best_eval_acc:
                    best_eval_acc = accuracy
                    torch.save(model.state_dict(), SAVE_MODEL_PATH)
        print(f"{epoch + 1}/{num_epochs} finished. Elapsed time: {time() - start:.1f} secs")
    print(f"Training finished. Best evaluation accuracy: {best_eval_acc:.4f}")

if __name__ == "__main__":
    train()
