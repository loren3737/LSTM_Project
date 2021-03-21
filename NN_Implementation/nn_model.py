import torch
import math
import nn_tools

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_nodes=300, layer1=20, layer2=10, layer3=4):
        super(NeuralNetwork, self).__init__()

        # Tracked information
        self.input_nodes = input_nodes

        # Layer 1
        linear1 = torch.nn.Linear(self.input_nodes, layer1)
        torch.nn.init.normal_(linear1.weight, mean=0.0, std=0.01)
        self.fullyConnectedOne = torch.nn.Sequential(
           linear1,
           torch.nn.Sigmoid()
           )
        
        # Layer 2
        linear2 = torch.nn.Linear(layer1, layer2)
        torch.nn.init.normal_(linear2.weight, mean=0.0, std=0.01)
        self.fullyConnectedTwo = torch.nn.Sequential(
           linear2,
           torch.nn.Sigmoid()
           )

        # Layer 3
        linear3 = torch.nn.Linear(layer2, layer3)
        torch.nn.init.normal_(linear3.weight, mean=0.0, std=0.01)
        self.fullyConnectedThree = torch.nn.Sequential(
           linear3,
           torch.nn.Sigmoid()
           )

        # Output layer
        linearOut = torch.nn.Linear(layer3, 1)
        torch.nn.init.normal_(linearOut.weight, mean=0.0, std=0.01)
        self.outputLayer = torch.nn.Sequential(
            linearOut,
            torch.nn.Sigmoid()
            )

    def forward(self, x):
        # Adjust input data
        out = x
        
        # Apply the layers created at initialization time in order
        out = self.fullyConnectedOne(out)
        out = self.fullyConnectedTwo(out)
        out = self.fullyConnectedThree(out)
        out = self.outputLayer(out)

        return out

    def predict(self, x, threshold=0.5):
        # Get the ouput propagations
        yValidatePredicted = self.forward(x)
        return [ 1 if pred > threshold else 0 for pred in yValidatePredicted ]
    
    def train_model_persample(self, xTrain, yTrain, max_epoch = 51, convergence = 0.000001, learning_rate = 1, min_epochs = 50):
        
        #Set training mode
        self.train(mode=False)    

        # Setup training values
        converged = False
        epoch = 1
        lastLoss = None
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        lossFunction = torch.nn.MSELoss(reduction='mean')
        loss_across_epoch = []
    
        while not converged and epoch < max_epoch:
            # Do the forward pass
            for i in range(len(xTrain)):            
                # sample = xTrain[i].unsqueeze(0)
                sample = xTrain[i].unsqueeze(0)
                yTrainPredicted = self(sample)
                trainLoss = lossFunction(yTrainPredicted, yTrain[i])

                # Reset the gradients in the network to zero
                optimizer.zero_grad()

                # Backprop the errors from the loss on this iteration
                trainLoss.backward()

                # Do a weight update step
                optimizer.step()

            loss = trainLoss.item()

            print("Current Loss: " + str(loss))
            print(loss)
            loss_across_epoch.append(loss)

            if lastLoss is None:
                lastLoss = loss
            else:
                if abs(lastLoss - loss) < convergence and epoch > min_epochs:
                    converged = True
                
            epoch = epoch + 1

        print("Total Epochs: " + str(epoch))
        self.train(mode=True)

        print("TRAINING LOSS")
        print(loss_across_epoch)

    def train_model_persample_dev(self, xTrain, yTrain, xDev, yDev, max_epoch = 51, convergence = 0.000001, learning_rate = 1, min_epochs = 50):
        
        #Set training mode
        self.train(mode=False)    

        # Setup training values
        converged = False
        epoch = 1
        lastLoss = None
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        lossFunction = torch.nn.MSELoss(reduction='mean')
        loss_across_epoch = []
        dev_acc_list = []

        while not converged and epoch < max_epoch:
            # Do the forward pass
            for i in range(len(xTrain)):            
                # sample = xTrain[i].unsqueeze(0)
                sample = xTrain[i].unsqueeze(0)
                yTrainPredicted = self(sample)
                trainLoss = lossFunction(yTrainPredicted, yTrain[i])

                # Reset the gradients in the network to zero
                optimizer.zero_grad()

                # Backprop the errors from the loss on this iteration
                trainLoss.backward()

                # Do a weight update step
                optimizer.step()

            loss = trainLoss.item()

            print("Current Loss: " + str(loss))
            print(loss)
            loss_across_epoch.append(loss)

            if lastLoss is None:
                lastLoss = loss
            else:
                if abs(lastLoss - loss) < convergence and epoch > min_epochs:
                    converged = True
                
            epoch = epoch + 1

            yPredict = self.predict(xDev)
            dev_accuracy = nn_tools.Accuracy(yDev, yPredict)
            dev_acc_list.append(dev_accuracy)


        print("Total Epochs: " + str(epoch))
        self.train(mode=True)

        print("TRAINING LOSS")
        print(loss_across_epoch)

        print("Dev accuracy")
        print(dev_acc_list)
        
