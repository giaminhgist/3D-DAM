from lib.dataloader.df_reader import df_reader
from lib.dataloader.MRIDataloader import MRIDataset
from lib.training import train
from lib.model.create_model import create_model
from lib.utils.utils import ParseKwargs
from torch.utils.data import DataLoader
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--experiment_name', type=str, default='AD_CN')
parser.add_argument('--task', type=str, default='AD_CN')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--train_type', type=str, default='image_level')
parser.add_argument('--output_size', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--w_decay', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--epoch_size', type=int, default=400)
parser.add_argument('--drop_out', type=float, default=0.5)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--image_folder', type=str, default='/media/tedi/Elements/ADNI_Database/Images/PROCESS/subjects/')
parser.add_argument('--train_path', type=str, default=None)
parser.add_argument('--val_path', type=str, default=None)
parser.add_argument('--model_name', type=str, default='SEModule')
parser.add_argument('--model_kwargs', nargs='*', default={},
                    action=ParseKwargs)  # example --model_kwargs embed_size=64 number_head=8
args = parser.parse_args()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training


    if args.train_path is None:
      train_path = f'/media/tedi/Elements/ADNI_Database/XLS_Files/{args.train_type}/{args.task}/{args.task}_train_fold_{args.fold}.xlsx'
    else:
      train_path = args.train_path
    if args.val_path is None:
      val_path = f'/media/tedi/Elements/ADNI_Database/XLS_Files/{args.train_type}/{args.task}/{args.task}_val_fold_{args.fold}.xlsx'
    else:
      val_path = args.val_path
      
    train_image_path, train_label_dict = df_reader(train_path, process_path=args.image_folder)
    val_image_path, val_label_dict = df_reader(val_path, process_path=args.image_folder)

    train_dataset = MRIDataset(
      train_image_path, train_label_dict, task=args.task
    )
    valid_dataset = MRIDataset(
      val_image_path, val_label_dict, task=args.task
    )

    print('Number of train files', len(train_dataset))
    print('Number of val files', len(valid_dataset))

    train_loader = DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=True
    )

    valid_loader = DataLoader(
      valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Model configuration
    model = create_model(
      model_name=args.model_name,
      num_classes=args.output_size,
      **args.model_kwargs,
    )
    train.train(
      model=model,
      train_loader=train_loader,
      val_loader=valid_loader,
      epoch_size=args.epoch_size,
      lr_scheduler=True,
      learning_rate=args.learning_rate, optimizer_setup='Adam', w_decay=args.w_decay,
      patience=args.patience, save_last=True,
      name=args.experiment_name, fold=args.fold
    )
