import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(conf_matrix):
    
        plt.figure(figsize=(8,6))  # figsize specifies the width and height of the figure
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')  # annot adds annotations, cmap sets the color map
        plt.xlabel('Predicted Labels')  # xlabel sets the label for the x-axis
        plt.ylabel('True Labels')  # ylabel sets the label for the y-axis
        plt.title('Confusion Matrix')
        plt.show()
    