import matplotlib
import numpy
import matplotlib.pyplot as plt

def plot(yData, Labels, xData, plotTitle, xTitle, yTitle, yTopLimit=None):

    if len(yData) != len(Labels):
        print("Error: Data and Label mismatch")

    for i in range(len(yData)):
        if len(yData[i]) != len(xData[i]):
            print("Error: yData and xData mismatch")

    plt.clf()
    
    fig, ax = plt.subplots()
    ax.grid()
    ax.set(title=plotTitle)      
    ax.set(xlabel=xTitle, ylabel=yTitle)

    if yTopLimit:
        ax.set_ylim(top=yTopLimit)

    for i in range(len(yData)):
        ax.plot(xData[i], yData[i], label = Labels[i])
            
    ax.legend()

    plt.show()

def plot_confusion_matrix(cm, target_names, title='Confusion matrix'):
    
    # Confusion Matrix Plotter
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def __SetUpChart(chartTitle=None, xAxisTitle=None, yAxisTitle=None):
   if chartTitle == None or xAxisTitle == None or yAxisTitle == None:
      raise UserWarning("Label your chart -- title and axes!")
   
   plt.clf()
   fig, ax = plt.subplots()
 
   ax.grid()
      
   ax.set(title=chartTitle)      
   ax.set(xlabel=xAxisTitle, ylabel=yAxisTitle)
   
   return fig, ax

def __CompleteChart(fig, ax, outputDirectory=None, fileName=None, yTopLimit=None, yBotLimit=None, invertYAxis=False):
   if yBotLimit != None:
      ax.set_ylim(bottom=yBotLimit)
   else:
      ax.set_ylim(bottom=0)
   
   if yTopLimit != None:
      ax.set_ylim(top=yTopLimit)

   ax.legend()
      
   if outputDirectory != None:
      filePath = "%s\\%s" % (outputDirectory, fileName)
      fig.savefig(filePath)

   else:
      plt.show()
      
   if invertYAxis:
      ax.invert_yaxis()
    
   matplotlib.pyplot.close(fig)

def __GetLineStyle(index, useLines):
   if useLines == False:
      return 'None'
   
   styles = ['-', ':', '-.', '--']
   
   return styles[index % len(styles)]

def __GetLineColor(index):
   colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
   
   return colors[index % len(colors)]

def __GetMarker(index, useMarkers=False):
   markers = ['x', 'o', '+', '*', 's']
   
   if useMarkers:
      return markers[index % len(markers)]
   else:
      return None

def plot_rocs(seriesFalsePositiveRates, seriesFalseNegativeRates, seriesLabels, useLines=True, chartTitle=None, xAxisTitle=None, yAxisTitle=None, outputDirectory=None, fileName=None):
   if len(seriesFalsePositiveRates) != len(seriesFalseNegativeRates):
      raise UserWarning("Mismatched number of seriesFalsePositiveRates and seriesFalseNegativeRates")
   
   if len(seriesFalsePositiveRates) != len(seriesLabels):
      raise UserWarning("Mismatched number of seriesFalsePositiveRates and seriesLabels")
   
   for i in range(len(seriesFalsePositiveRates)):
      if len(seriesFalsePositiveRates[i]) != len(seriesFalseNegativeRates[i]):
         raise UserWarning("Number of Y points in series %d does not match the number of X points" % (i))
      
   fig, ax =__SetUpChart(chartTitle, xAxisTitle, yAxisTitle)

   for i in range(len(seriesFalsePositiveRates)):
      ax.plot(seriesFalseNegativeRates[i], seriesFalsePositiveRates[i], label = seriesLabels[i], marker ='', color = __GetLineColor(i), linestyle=__GetLineStyle(i, useLines))

   ax.set_xlim(left=0.0)
   ax.set_xlim(right=1.0)
              
   __CompleteChart(fig, ax, outputDirectory, fileName, 0.0, 1.0, invertYAxis=True)