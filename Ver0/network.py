"""
This script is used to create and train the LSTM neural network model. Saves the trained model as "trained_model.pth".
"""
import torch
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import random_split
matplotlib.use('macosx') #For the plots show correctly in pycharm

""" ------------------------- Preparing the Data -------------------------"""

# Loading the dataset
dataset = torch.load('dataset.pth')

# Loading the vocabulary
vocab_list = torch.load('vocab_list.pth')
token2idx = {token: i for i, token in enumerate(vocab_list)} #Converting word to numerical index
idx2token = {i: token for i, token in enumerate(vocab_list)} #Numerical index to word list, to be used in generating music

vocab_size = len(vocab_list)

# 80/20 split for training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Splitting the dataset into train and validation. Using a seed for reproducibility
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(14))


from torch.utils.data import DataLoader
# Create a DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #Shuffled
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)  #Not Shuffled

""" ------------------------- Creating the Model -------------------------"""

import torch.nn as nn

# Parameters
embedding_dim = 8   # Size of the embedding vectors (8 is best)
hidden_dim = 64     # cell state vector size (64 is best)
num_layers = 2       # number of LSTM layers (2 is best)
learning_rate = 0.001 # learning rate (0.001 is best)

# The class for the LSTM model:
class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Maps tokens to vectors
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Maps hidden states to vocabulary size

    def forward(self, x, h0 = None, c0 = None):
        batch_size = x.shape[0]  # Get batch size dynamically
        if h0 is None or c0 is None: #Initialize hidden states as zeros in the beginning of batches
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        embedded = self.embedding(x)  # Embedding layer
        out, (h, c) = self.lstm(embedded, (h0,c0))  # LSTM layer
        logits = self.fc(out)  # Fully connected layer
        return logits, (h, c)

# Making sure that the model is using the right device:
if torch.backends.mps.is_available():
    device = torch.device("mps") #mps for macbook
elif torch.cuda.is_available():
    device = torch.device("cuda") #cuda for google colab
else:
    device = torch.device("cpu")
print("Using device:", device)

# Initializing the model and moving to device
model = MusicLSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers
).to(device)

# Loss function and optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_perplexities = []
val_perplexities = []

for epoch in range(num_epochs):
    # TRAINING
    model.train()
    train_loss = 0.0  # Training loss
    correct_train_preds = 0.0  # Counts correct predictions
    total_train_preds = 0.0  # Counts total predictions



    for x_batch, y_batch in train_loader: #looping over batches
        h, c = None, None
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()  # Resets gradients
        output, (h,c) = model(x_batch, h, c)  # Forward pass

        output = output.view(-1, vocab_size)  # Reshape for loss calculation
        y_batch = y_batch.view(-1)

        loss_val = loss(output, y_batch)  # Compute loss
        loss_val.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        train_loss += loss_val.item()  # loss, from tensor to float

        # Computing accuracy
        preds = output.argmax(dim=1)  # Finding the prediction with highest probability
        correct_train_preds += (preds == y_batch).sum().item() #If the prediction is correct, prediction is equal to the y value
        total_train_preds += y_batch.shape[0]

        h = h.detach() # So that backpropagation is not over time steps
        c = c.detach()

    #Computing losses
    avg_train_loss = train_loss / len(train_loader)  # Average training loss
    train_perplexity = math.exp(avg_train_loss)  # Convert average loss to perplexity
    train_accuracy = correct_train_preds / total_train_preds  # Training accuracy
    train_perplexities.append(train_perplexity)
    train_accuracies.append(train_accuracy)

    # VALIDATION
    model.eval()
    val_loss = 0.0  # Validation loss
    correct_val_preds = 0  # Count correct predictions
    total_val_preds = 0  # Count total predictions

    with torch.no_grad(): # No need to do backprobagation since the gradients are already calculated.
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            h, c = None, None

            val_output, (h,c) = model(x_val, h, c)
            val_output = val_output.view(-1, vocab_size)
            y_val = y_val.view(-1)

            val_loss += loss(val_output, y_val).item()  # loss, from tensor to float

            # Compute accuracy
            preds = val_output.argmax(dim=1)  # Get predicted classes
            correct_val_preds += (preds == y_val).sum().item()
            total_val_preds += y_val.size(0)

    #Computing losses
    avg_val_loss = val_loss / len(val_loader)  # Average validation loss
    val_perplexity = math.exp(avg_val_loss)  # Convert average loss to perplexity
    val_accuracy = correct_val_preds / total_val_preds  # Validation accuracy
    val_perplexities.append(val_perplexity)
    val_accuracies.append(val_accuracy)

    # Printing results for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}]"
          f"  -  Train Perplexity: {train_perplexity:.4f}, Train Accuracy: {train_accuracy:.4f}"
          f" --- Val Perplexity: {val_perplexity:.4f}, Val Accuracy: {val_accuracy:.4f}")

#letting me know that the model has finished running (works on mac)
import os
os.system('say "Your model has finished running"')

""" ------------------------- Plotting the Measurements -------------------------"""
# plotting perplexity
plt.figure(figsize=(10, 5))
plt.plot(train_perplexities, label='Training Perplexity')
plt.plot(val_perplexities, label='Validation Perplexity')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Training and Validation Perplexity')
plt.legend()
plt.show()

# plotting accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


#Saving the model:
torch.save(model, "trained_model.pth")



