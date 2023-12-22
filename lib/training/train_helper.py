from matplotlib import pyplot as plt


def plot_result(title, val_list, train_list, type_data='Loss'):
    fig = plt.figure(figsize=(10, 10), dpi=500)
    plt.title(f'{title}')
    plt.plot(val_list, label="val")
    plt.plot(train_list, label="train")
    plt.xlabel("iterations")
    plt.ylabel(f'{type_data}')
    if type_data == 'Loss':
        plt.ylim(0, 3)
    else:
        plt.ylim(40, 100)
    plt.legend()
    fig.savefig(f'{title}_{type_data}.png', bbox_inches='tight')
    plt.close(fig)