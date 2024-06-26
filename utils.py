import os
from tqdm import tqdm

from monai.inferers import sliding_window_inference

from monai.networks.nets import SwinUNETR, UNet
from monai.networks.layers import Norm

from monai.data import decollate_batch
import warnings
warnings.filterwarnings("ignore")
from helper import * 
import torch


def validation(model, validation_loader, post_label, post_pred, dice_metric, epoch, device):
    model.eval()
    with torch.no_grad():
        for batch in validation_loader:
            val_inputs, val_labels, name = (batch["image"].to(device), batch["label"].to(device), batch['name'])

            found = any("Task06_Lung" in text or "Task10_Colon" in text for text in name)
            if found:
                val_labels = (val_labels == 1)*1
            else:
                val_labels = (val_labels == 2)*1

            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            
            validation_loader.set_description("Validate (%d / %d Epoch)" % (epoch, 10.0))  # noqa: B038
        
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val



def train(model, train_loader, loss_function, optimizer, epoch, scaler, device):
    model.train()
    epoch_loss = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        x, y, name = (batch["image"].to(device), batch["label"].to(device), batch['name'])

        # Creating the binary masks 
        found = any("Task06_Lung" in text or "Task10_Colon" in text for text in name)
        if found:
            y = (y == 1)*1
        else:
            y = (y == 2)*1

        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f)" % (
                epoch, step, len(train_loader), loss.item())
        )

        epoch_loss += loss.item()
        #torch.cuda.empty_cache()

    print('Epoch=%d: Average_loss=%2.5f' % (epoch, epoch_loss/len(epoch_iterator)))
    return epoch_loss/len(epoch_iterator)



def model_setup(args):
    # loading the model
    if args.model_name == 'swin':
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=args.no_class,
            feature_size=48,
            #use_checkpoint=True,
        ).to(args.device)
    # elif args.model_name == 'unet':
    #     pass 
    else: 
        raise ValueError("The model specified can not be found. Please select from the list [unet, resunet, swin].")

    #Load pre-trained weights
    if args.pretrain is not None:
        model.load_params(torch.load(args.pretrain)["state_dict"])

    # print the model architecture and number of parameters 
    print(model)
    count_parameters(model)

    # Define the path to the saved model file
    saved_model_path = os.path.join(args.save_directory, "best_metric_model.pth")

    if args.resume:
        # Check if the path exists
        if os.path.exists(saved_model_path):
            # Load the saved model weights into the model
            model.load_state_dict(torch.load(saved_model_path))
            print("The model is restored from a pretrained .pth file")
        else:
            print("Training the model from scratch")
    else:
        print("Training the model from scratch")

    return model
