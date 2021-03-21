import torch.nn as nn
import torch
import time
import math

def timeSince(since):
            now = time.time()
            s = now - since
            m = math.floor(s / 60)
            s -= m * 60
            return '%dm %ds' % (m, s)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=20, layer=1):
        super(RNN, self).__init__()

        # Save dimensions
        self.hidden_size = hidden_size
        self.layer = layer

        # RNN layer
        self.RecurrentOne = nn.RNN(input_size, hidden_size, layer, batch_first=True)
        torch.nn.init.normal_(self.RecurrentOne.weight_hh_l0, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.RecurrentOne.weight_ih_l0, mean=0.0, std=0.01)
          
        # Output layer
        linearOut = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.normal_(linearOut.weight, mean=0.0, std=0.01)
        self.outputLayer = torch.nn.Sequential(
            linearOut,
            torch.nn.Sigmoid()
            )

    def forward(self, xReview):
        
        hidden0 = torch.zeros(self.layer, 1, self.hidden_size)
        
        X, hidden = self.RecurrentOne(xReview, hidden0)

        X = self.outputLayer(X)
        
        # X = X.view(batch_size, seq_len, self.nb_tags)
        Y_hat = X
        return Y_hat

    def train_review(self, xReview, sentiment, learning_rate):

        self.zero_grad()
        lossFunction = torch.nn.MSELoss(reduction='mean')

        output = self(xReview)

        loss = lossFunction(output, sentiment)
        loss.backward(retain_graph=True)

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in self.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        return output, loss.item()

    def train(self, xTrain, yTrain, learning_rate=0.1, epochs = 20):
        torch.autograd.set_detect_anomaly(True)
        # Metrics
        print_every = 5000
        plot_every = 1000

        # Keep track of losses for plotting
        current_loss = 0
        all_losses = []
        start = time.time()

        # Training
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        lossFunction = torch.nn.MSELoss(reduction='mean')
        xTrain = xTrain.unsqueeze(0)
        yTrain = yTrain.unsqueeze(1)
        yTrain = yTrain.unsqueeze(0)

        for iter in range(1, epochs + 1):
            print("Training Epoch : " + str(iter))
            y_hat = self(xTrain)

            trainLoss = lossFunction(y_hat, yTrain)

            # Reset the gradients in the network to zero
            optimizer.zero_grad()

            # Backprop the errors from the loss on this iteration
            trainLoss.backward(retain_graph=True)

            # Do a weight update step
            optimizer.step()

            loss = trainLoss.item()

            print(loss)

            # for index, xReview in enumerate(xTrain):
            #     print("Training Epoch : " + str(index + 1))
            #     sentiment = yTrain[index]
            #     output, loss = self(xReview, sentiment, learning_rate)
            #     current_loss += loss