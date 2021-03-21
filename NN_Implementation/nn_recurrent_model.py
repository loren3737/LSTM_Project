import torch.nn as nn
import torch
import time
import math
import nn_tools

def timeSince(since):
            now = time.time()
            s = now - since
            m = math.floor(s / 60)
            s -= m * 60
            return '%dm %ds' % (m, s)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=40, output_size=2):
        super(RNN, self).__init__()

        # Save dimensions
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(hidden_size, output_size)
        # torch.nn.init.normal_(self.i2h.weight, mean=0.0, std=0.01)
        # torch.nn.init.normal_(self.i2o.weight, mean=0.0, std=0.01)

        self.i2h_node = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o_node = nn.Linear(hidden_size, output_size)
        torch.nn.init.normal_(self.i2h_node.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.i2o_node.weight, mean=0.0, std=0.01)

        self.i2h = torch.nn.Sequential(
           self.i2h_node,
           torch.nn.Tanh()
           )

        self.i2o = torch.nn.Sequential(
           self.i2o_node,
        #    torch.nn.Tanh()
           )

        # self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputX, hidden):
        combined = torch.cat((inputX, hidden), 1)
        hidden = self.i2h(combined)
        if math.isnan(hidden[0][0]):
                phoo = 1
        output = self.i2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    
    def train(self, xTrain, sentiment, learning_rate = 0.001, epochs = 1):

        sentiment = torch.Tensor(sentiment)
        sentiment = sentiment.type(torch.LongTensor)

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        training_loss = []
       
        for iteration in range(1, epochs + 1):
            avg_loss = 0
            avg_count = 0
            for i, review in enumerate(xTrain):
                if i == 4845:
                    phoo = 2

                hidden = self.initHidden()

                self.zero_grad()

                for j, word in enumerate(review):
                    word_tensor = torch.Tensor(word).unsqueeze(0)
                    output, hidden = self.forward(word_tensor, hidden)

                loss = criterion(output, sentiment[i])
                loss.backward()
                
                # Print loss for the review
                loss_value = loss.item()
                training_loss.append(loss_value)
                avg_loss += loss_value
                avg_count += 1
                # print(loss_value)

                optimizer.step()

                # Add parameters' gradients to their values, multiplied by learning rate
                # for p in self.parameters():
                #     p.data.add_(p.grad.data, alpha=-learning_rate)

            print(avg_loss/avg_count)
            
        print("Loss over iteration")
        print(training_loss)

        return

    def train_dev(self, xTrain, sentiment, xDev, yDev, learning_rate = 0.001, epochs = 1):

        sentiment = torch.Tensor(sentiment)
        sentiment = sentiment.type(torch.LongTensor)

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()
        dev_acc_list = []

        training_loss = []
       
        for iteration in range(1, epochs + 1):
            avg_loss = 0
            avg_count = 0
            for i, review in enumerate(xTrain):
                if i == 4845:
                    phoo = 2

                hidden = self.initHidden()

                self.zero_grad()

                for j, word in enumerate(review):
                    word_tensor = torch.Tensor(word).unsqueeze(0)
                    output, hidden = self.forward(word_tensor, hidden)

                loss = criterion(output, sentiment[i])
                loss.backward()
                
                # Print loss for the review
                loss_value = loss.item()
                training_loss.append(loss_value)
                avg_loss += loss_value
                avg_count += 1
                # print(loss_value)

                optimizer.step()

                # Add parameters' gradients to their values, multiplied by learning rate
                # for p in self.parameters():
                #     p.data.add_(p.grad.data, alpha=-learning_rate)

            print(avg_loss/avg_count)
            yPredict = self.predict(xDev)
            dev_acc = nn_tools.Accuracy(yDev, yPredict)
            dev_acc_list.append(dev_acc)

            
        print("Loss over iteration")
        print(training_loss)

        print("Dev Accuracy")
        print(dev_acc_list)
        
        return

    def predict(self, xData, threshold=0.5):
        yPredict = []
        for review in xData:   
                hidden = self.initHidden()

                for j, word in enumerate(review):
                    word_tensor = torch.Tensor(word).unsqueeze(0)
                    output, hidden = self.forward(word_tensor, hidden)
                
                if output[0][1] > threshold:
                    yPredict.append(1)
                else:
                    yPredict.append(0)

        return yPredict

                
        



