import matplotlib as plt


def plotLosses(train_loss, val_loss=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if val_loss:
        ax.plot(range(len(val_loss)), val_loss, label='Validation Loss')

    ax.plot(range(len(train_loss)), train_loss, label='Training loss')

    plt.legend()
    plt.title("Train/Validation Loss Plot")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
