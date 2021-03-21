import torch

def train(model, xTrain, yTrain, learning_rate=1, max_epoch=20, min_epochs=0, convergence=0):
    # We have two different types of end conditions
    # Runnning until convergence is hit or until we reach max_epochs
    if max_epoch is None and convergence is None:
        raise Exception("Either max_epoch or convergence must be set for training.")

    #Set training mode
    model.train(mode=False) 

    # Setup training values
    converged = False
    epoch = 1
    last_loss = None
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lossFunction = torch.nn.MSELoss(reduction='mean')
    loss_over_epoch = []

    while not converged and epoch < max_epoch:
        # Do the forward pass
        epoch_loss = 0
        for i, sample in enumerate(xTrain):
                
            sample = sample.unsqueeze(0)
            yTrainPredicted = model(sample)
            sample_loss = lossFunction(yTrainPredicted, yTrain[i].unsqueeze(0))
            epoch_loss += abs(sample_loss.item())

            # Reset the gradients in the network to zero
            optimizer.zero_grad()

            # Backprop the errors from the loss on this iteration
            sample_loss.backward()

            # Do a weight update step
            optimizer.step()

        epoch_loss /= len(xTrain)

        print("Epoch Average Loss: " + str(epoch_loss))
        loss_over_epoch.append(epoch_loss)

        if last_loss is None:
            last_loss = epoch_loss
        else:
            if abs(last_loss - epoch_loss) < convergence and epoch > min_epochs:
                converged = True
            
        epoch += 1

    print()
    print("Finish Training in Epochs: " + str(epoch))
    print("Loss over each step")
    print(loss_over_epoch)

    model.train(mode=True)

def predict(model, xData, threshold = 0.5):
    yValidatePredicted = model(xData)
    return [ 1 if pred > threshold else 0 for pred in yValidatePredicted ]

    