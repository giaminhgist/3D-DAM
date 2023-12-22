from lib.dataloader.df_reader import df_reader
from lib.dataloader.MRIDataloader import MRIDataset
from lib.training import normal_train, patch_train
from lib.model.create_model import create_model
from lib.utils.utils import ParseKwargs
from torch.utils.data import DataLoader
import argparse
import torch
import numpy as np

# python train.py --experiment_name SEModuel_AD_CN_MCI --task AD_CN_MCI --fold 0 --train_type image_level --output_size 3 --learning_rate 1e-4 --w_decay 1e-4 --batch_size 16 --epoch_size 400 --patience 15

parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--experiment_name', type=str, default='AD_CN_MCI')
parser.add_argument('--task', type=str, default='AD_CN_MCI')  # #1: AD_CN_MCI #2 pMCI_sMCI
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--train_type', type=str, default='image_level')  # #1: image_level #2 patient_level
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
parser.add_argument('--unlabeled_train_path', type=str, default=None)
parser.add_argument('--val_path', type=str, default=None)
parser.add_argument('--is_patch', type=str, default='False')
parser.add_argument('--is_VIT', type=str, default='False')
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

    # Data setting
    if args.is_VIT == 'True':
        is_VIT = True
    else:
        is_VIT = False

    if args.is_patch == 'True':
        is_patch = True
    else:
        is_patch = False

    # Training

    if is_patch:

        # Train path
        if args.train_path is None:
            train_path = f'/media/tedi/Elements/ADNI_Database/XLS_Files/Patch_{args.train_type}/{args.task}/{args.task}_train_fold_{args.fold}.xlsx'
        else:
            train_path = args.train_path
        if args.val_path is None:
            val_path = f'/media/tedi/Elements/ADNI_Database/XLS_Files/Patch_{args.train_type}/{args.task}/{args.task}_val_fold_{args.fold}.xlsx'
        else:
            val_path = args.val_path

        print(train_path)

        train_image_path, train_label_dict, train_feature_dict = df_reader(train_path, process_path=args.image_folder)
        val_image_path, val_label_dict, val_feature_dict = df_reader(val_path, process_path=args.image_folder)

        train_dataset = MRIDataset(train_image_path, train_label_dict, train_feature_dict, task=args.task,
                                   is_patch=is_patch, patch_size=args.patch_size)
        valid_dataset = MRIDataset(val_image_path, val_label_dict, val_feature_dict, task=args.task,
                                   is_patch=is_patch, patch_size=args.patch_size)

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

        patch_train.train(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            epoch_size=args.epoch_size,
            lr_scheduler=True,
            learning_rate=args.learning_rate, optimizer_setup='Adam', w_decay=args.w_decay,
            patience=args.patience, save_last=True,
            name=args.experiment_name, fold=args.fold
        )
    # Normal Training
    else:

        if args.train_path is None:
            train_path = f'/media/tedi/Elements/ADNI_Database/XLS_Files/{args.train_type}/{args.task}/{args.task}_train_fold_{args.fold}.xlsx'
        else:
            train_path = args.train_path
        if args.val_path is None:
            val_path = f'/media/tedi/Elements/ADNI_Database/XLS_Files/{args.train_type}/{args.task}/{args.task}_val_fold_{args.fold}.xlsx'
        else:
            val_path = args.val_path

        train_image_path, train_label_dict, train_feature_dict = df_reader(train_path, process_path=args.image_folder)
        val_image_path, val_label_dict, val_feature_dict = df_reader(val_path, process_path=args.image_folder)

        train_dataset = MRIDataset(train_image_path, train_label_dict, train_feature_dict, task=args.task,
                                   is_patch=is_patch, patch_size=args.patch_size)
        valid_dataset = MRIDataset(val_image_path, val_label_dict, val_feature_dict, task=args.task,
                                   is_patch=is_patch, patch_size=args.patch_size)

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

        normal_train.train(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            epoch_size=args.epoch_size,
            lr_scheduler=True,
            learning_rate=args.learning_rate, optimizer_setup='Adam', w_decay=args.w_decay,
            patience=args.patience, save_last=True,
            name=args.experiment_name, fold=args.fold
        )
