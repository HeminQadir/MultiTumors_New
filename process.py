import os

from tqdm import tqdm
from dataset.data_split_fold import datafold_read
from monai.losses import DiceCELoss, DiceLoss
from monai.transforms import AsDiscrete

from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, UNet
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
from tensorboardX import SummaryWriter
from dataset.dataloader import get_loader 
from utils import *
from helper import * 


def process(args):

    # check if this directory is available, if not make it 
    make_dir(args.save_directory)

    # Setup the model 
    model = model_setup(args) 

    # Split the dataset into train and validation subsets 
    train_files, val_files = datafold_read(args) 

    # get the data loader an apply augmenations  
    train_loader, val_loader, subset_loader = get_loader(train_files, val_files, args)

    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    #loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    post_label = AsDiscrete(to_onehot=args.no_class)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.no_class)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    dice_val_best = 0.0
    
    writer = SummaryWriter(log_dir='out/' + args.model_name)
    print('Writing Tensorboard logs to ', 'out/' + args.model_name)

    while args.epoch < args.max_epoch:

        try: 
            # Traininig 
            ave_loss = train(model, train_loader, loss_function, optimizer, args.epoch, scaler, args.device)
            writer.add_scalar('train_dice_loss', ave_loss, args.epoch)

            # Validation 
            if (args.epoch) % args.val_interval == 0:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Epoch) (dice=X.X)", dynamic_ncols=True)
                print("starting the validation phase")
                dice_val = validation(model, epoch_iterator_val, post_label, post_pred, dice_metric, args.epoch, args.device) 

                epoch_iterator_train = tqdm(subset_loader, desc="Train validate (X / X Epoch) (dice=X.X)", dynamic_ncols=True)
                metric_train = validation(model, epoch_iterator_train, post_label, post_pred, dice_metric, args.epoch, args.device) 

                if dice_val > dice_val_best:
                    dice_val_best = dice_val

                    torch.save(model.state_dict(), os.path.join(args.save_directory, "best_metric_model.pth"))
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                
                writer.add_scalar("Validation/Average Accuracy:", scalar_value=torch.tensor(dice_val), global_step=args.epoch+1)
                writer.add_scalar("Train/Average Accuracy:", scalar_value=torch.tensor(metric_train), global_step=args.epoch+1)

            args.epoch += 1

        except: 
            torch.save(model.state_dict(), os.path.join(args.save_directory, "last_epoch_model.pth"))


    torch.save(model.state_dict(), os.path.join(args.save_directory, "last_epoch_model.pth"))
    print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {args.epoch}")



