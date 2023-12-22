import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from lib.utils.utils import AverageMeter, accuracy
from lib.utils.EarlyStopping import EarlyStopping
from lib.training.train_helper import plot_result
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from GPUtil import showUtilization as gpu_usage
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
        model,
        loader,
        optimizer,
        epoch_idx: int,
        lr_scheduler=None,
):
    losses_m = AverageMeter()
    acc_m = AverageMeter()

    model.train()
    print('Start training epoch: ', epoch_idx)
    for batch_idx, data in enumerate(tqdm(loader)):

        images, target = data
        images, target = images.to(device), target.to(device)
        target = target.flatten()

        output = model(images)

        loss = nn.CrossEntropyLoss()(output, target)

        losses_m.update(loss.item(), images.size(0))
        acc1 = accuracy(output, target, topk=(1,))
        acc_m.update(acc1[0].item(), output.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

    print(optimizer.param_groups[0]['lr'])

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    metrics = OrderedDict([('loss', losses_m.avg), ('Acc', acc_m.avg)])
    if lr_scheduler is not None:
        lr_scheduler.step()

    return metrics


def validate(model, loader):
    losses_m = AverageMeter()
    acc_m = AverageMeter()

    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            images, target = data
            images, target = images.to(device), target.to(device)
            target = target.flatten()

            output = model(images)

            loss = nn.CrossEntropyLoss()(output, target)
            acc1 = accuracy(output, target, topk=(1,))
            # reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(loss.item(), images.size(0))
            acc_m.update(acc1[0].item(), output.size(0))

    metrics = OrderedDict([('loss', losses_m.avg), ('Acc', acc_m.avg)])

    return metrics


def train(model,
          train_loader,
          val_loader,
          epoch_size=300,
          lr_scheduler=True,
          learning_rate=1e-7, optimizer_setup='Adam', w_decay=1e-7,
          patience=20, save_last=True,
          name='save', fold=0,
          ):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Training using:', device)
    model = torch.nn.DataParallel(model)
    model.to(device)

    if optimizer_setup == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    elif optimizer_setup == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay)

    min_valid_loss = np.inf
    max_acc = 0
    highest_val_epoch = 0
    train_acc = []
    train_losses = []
    val_acc = []
    val_losses = []

    if lr_scheduler:
        print('Applied lr_scheduler')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    else:
        scheduler = None

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print('Start Training Process:...')

    for epoch in range(epoch_size):

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch_idx=epoch + 1,
            lr_scheduler=scheduler,
        )

        eval_metrics = validate(model, val_loader)

        train_acc.append(train_metrics['Acc'])
        train_losses.append(train_metrics['loss'])
        val_acc.append(eval_metrics['Acc'])
        val_losses.append(eval_metrics['loss'])

        if save_last:
            torch.save(model.module.state_dict(),
                       f'/media/tedi/Elements/YJ_GM_Project/WEIGHT/{name}/Fold{fold}/{name}_{fold}_last.pth')
        print(f'Epoch {epoch + 1}:  Train: {train_metrics}-----Val: {eval_metrics}')

        if min_valid_loss > eval_metrics['loss']:
            print(f'Validation Loss Decreased. \t Saving The Model')
            min_valid_loss = eval_metrics['loss']
            # Saving State Dict
            torch.save(model.module.state_dict(),
                       f'/media/tedi/Elements/YJ_GM_Project/WEIGHT/{name}/Fold{fold}/{name}_{fold}_best_loss.pth')

        if max_acc < eval_metrics['Acc']:
            print(f'Validation Acc Increased. \t Saving The Model')
            max_acc = eval_metrics['Acc']
            highest_val_epoch = epoch + 1
            # Saving State Dict
            torch.save(model.module.state_dict(),
                       f'/media/tedi/Elements/YJ_GM_Project/WEIGHT/{name}/Fold{fold}/{name}_{fold}_best_acc.pth')

        early_stopping(eval_metrics['loss'], model)
        if early_stopping.early_stop:
            print(f'Early stopping at: {epoch - 9}')
            print(f'Highest validation accuracy: {max_acc} at epoch {highest_val_epoch}')
            plot_result(f'/media/tedi/Elements/YJ_GM_Project/WEIGHT/{name}/Fold{fold}/{name}_{fold}__Loss', val_losses,
                        train_losses, type_data='Loss')
            plot_result(f'/media/tedi/Elements/YJ_GM_Project/WEIGHT/{name}/Fold{fold}/{name}_{fold}__Acc', val_acc,
                        train_acc,
                        type_data='Accuracy')
            break


def test(
        model,
        test_loader,
        output_size
):
    y_pred = []
    y_true = []
    prob = []

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()
    with torch.no_grad():
        print('Start Testing:...')
        for batch_idx, data in enumerate(test_loader):
            images, target = data
            images, target = images.to(device), target.to(device)
            target = target.flatten()

            output = model(images)

            prob.extend(output)
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)
            labels = target.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    conf_mat = confusion_matrix(y_true, y_pred)
    y_true_1 = torch.LongTensor(y_true)
    y_true_2 = F.one_hot(y_true_1, num_classes=output_size)
    prob_1 = torch.FloatTensor(prob)
    print('Testing has finished.')
    return prob_1, y_true_2, conf_mat
