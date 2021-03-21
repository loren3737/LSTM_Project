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