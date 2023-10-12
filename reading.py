
from torchvision import datasets 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
    

# ----------------- Used Kaggle to download MNIST dataset, took about 3 minutes to write this code -----------------#

training_data = datasets.MNIST(root="data", 
                               download=True, 
                               train=True, 
                               transform=ToTensor(),
                               )

test_data = datasets.MNIST(root="data",
                           train=False,  #Downloads different dataset for train value... Prevent overfitting.
                           download=True,  
                           transform=ToTensor(),
                            )

losses = []
batch_numbers = []

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input size: 28x28, Output size: 128
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Output size: 10 (for 10 classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = Net()


criteria = nn.CrossEntropyLoss()  #Determine the criteria of right score, softmax activation to normalize the data so that probabilities add to 1, measure real val vs generated val
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, dataloader, criteria, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100}')
                losses.append(running_loss / 100)
                running_loss = 0.0

# Train the model
train(model, train_dataloader, criteria, optimizer)

# Test the model
correct = 0
total = 0
with torch.no_grad():   #No need to calculate gradient of anything, we are only evaluating the models accuracy here
    for data in test_dataloader:  #Accessing data
        images, labels = data   
        outputs = model(images)  #Check the models effectiveness on new data, checking for overfitting 
        _, predicted = torch.max(outputs.data, 1)   #Get the index of the class with the highest probability 
        total += labels.size(0) #Total number of tests
        correct += (predicted == labels).sum().item()  #Calculates how many it got right

print(f'Accuracy on the test set: {100 * correct / total}%')   #For us to see

# Plotting the loss
plt.plot(losses)
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()