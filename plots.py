import seaborn as sns
import sklearn.metrics
import utils
import matplotlib.pyplot as plt
import math

def confusion_matrix(dataset, y_pred, title, test = True, class_names = utils.CLASS_NAMES):
    if test:
        data = dataset.y_test
    else:
        data = dataset.y_train
    
    sns.heatmap(sklearn.metrics.confusion_matrix(data[0:y_pred.shape[0],].argmax(1), y_pred.argmax(1), normalize = 'true'),
                annot = True,
                xticklabels = class_names,
                yticklabels = class_names).set(title = title)
    plt.show()

def heatmatrix(matrix, title):
    sns.heatmap(matrix,
                annot = True,
                xticklabels = utils.CLASS_NAMES,
                yticklabels = utils.CLASS_NAMES).set(title = title)
    plt.show()

def compare_images(images_a, images_b, rows):
    fig = plt.figure(figsize=(32, 24))

    columns = 10

    for i in range(int(rows * columns / 2)):
        if i >= images_a.shape[0]:
            plt.show()
            return

        fig.add_subplot(rows, columns, i*2+1)
        plt.imshow(images_a[i,])
        plt.axis('off')

        fig.add_subplot(rows, columns, i*2+2)
        plt.imshow(images_b[i,])
        plt.axis('off')

    plt.show()