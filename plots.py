import seaborn as sns
import sklearn.metrics
import utils
import matplotlib.pyplot as plt

def confusion_matrix(dataset, y_test_pred, title):
    sns.heatmap(sklearn.metrics.confusion_matrix(dataset.y_test[0:y_test_pred.shape[0],].argmax(1), y_test_pred.argmax(1), normalize = 'true'),
                annot = True,
                xticklabels = utils.CLASS_NAMES,
                yticklabels = utils.CLASS_NAMES).set(title = title)
    plt.show()