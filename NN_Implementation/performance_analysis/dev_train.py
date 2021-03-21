import data_plotter
# dev_train = [0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.4904, 0.7728, 0.8224, 0.8248, 0.826, 0.8284, 0.8276, 0.8272, 0.8268, 0.8268, 0.8284, 0.8284, 0.826, 0.8264, 0.8272, 0.8284, 0.8288, 0.8284, 0.8296, 0.8292, 0.83, 0.8296, 0.83, 0.8312]
# iterations = range(len(dev_train))
# data_plotter.plot(yData=[dev_train], Labels=["Dev Set"], xData=[iterations], plotTitle="Feedforward NN Learning", xTitle="Epochs", yTitle="Dev Set Accuracy")


dev_train = [0.4904, 0.7356, 0.8, 0.8232, 0.8276, 0.8296, 0.8284, 0.8292, 0.8384, 0.84, 0.8388, 0.8352, 0.8396, 0.8332, 0.842, 0.8372, 0.8444, 0.844, 0.82, 0.8424, 0.8472, 0.822, 0.8404, 0.8432, 0.8444, 0.8484, 0.8464, 0.8304, 0.8488, 0.844]
iterations = range(len(dev_train))
data_plotter.plot(yData=[dev_train], Labels=["Dev Set"], xData=[iterations], plotTitle="Recurrent NN Learning", xTitle="Epochs", yTitle="Dev Set Accuracy")
