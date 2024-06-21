import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset.data_split_fold import datafold_read
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    SpatialPadd,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, UNet
from monai.networks.layers import Norm

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    DataLoader,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
import warnings
warnings.filterwarnings("ignore")
import random 
from torch.utils.data import Subset
import torch
from tensorboardX import SummaryWriter

print_config()

root_dir = "./trained_models"
# Check if the directory exists
if not os.path.exists(root_dir):
    # Create the directory
    os.makedirs(root_dir)


json_list = '/home/hemin/MultiTumors_New/dataset/dataset_json_files/training_data_debug.json'

train_files, val_files = datafold_read(datalist=json_list, fold=0)


num_samples = 4
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        #EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)


val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        #EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ]
)

train_ds = CacheDataset(
     data=train_files,
     transform=train_transforms,
     #cache_num=24,    # this guy make it so slow 
     cache_rate=1.0,
     num_workers=8,
 )
#train_ds = Dataset(data=train_files, transform=train_transforms)


# Get the length of the dataset
dataset_length = len(train_ds)
# Generate a random range of integers with the maximum being the length of train_dataset
subset_indices = random.sample(range(dataset_length), int(dataset_length/4)) 
print(f"number of subset from training set {len(subset_indices)}")

subset = Subset(train_ds, subset_indices)

subset_loader = DataLoader(subset, batch_size=1, 
                            shuffle=False, 
                            num_workers=0)


#train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
train_loader = DataLoader(train_ds, 
                            batch_size=1, 
                            shuffle=True, 
                            num_workers=0)
                            #collate_fn=list_data_collate, 
                            #sampler=train_sampler, 
                            #pin_memory=True)

# val_ds = Dataset(data=val_files, transform=val_transforms)

val_ds = CacheDataset(data=val_files, transform=val_transforms, 
                    #cache_num=6,
                    cache_rate=1.0, num_workers=4)


#val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

val_loader = DataLoader(val_ds, 
                        batch_size=1, 
                        num_workers=0)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=2,
    feature_size=48,
    #use_checkpoint=True,
).to(device)


# Define the path to the saved model file
saved_model_path = os.path.join(root_dir, "best_metric_model.pth")

# Check if the path exists
if os.path.exists(saved_model_path):
    # Load the saved model weights into the model
    model.load_state_dict(torch.load(saved_model_path))
    print("The model is restored from a pretrained .pth file")
else:
    print("Training the model from scratch")


# If you're using GPU, you may need to transfer the model to the GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Now you can use the loaded model for inference or further training



# model = UNet(
#         spatial_dims=3,
#         in_channels=1,
#         out_channels=2,
#         channels=(16, 32, 64, 128, 256),
#         strides=(2, 2, 2, 2),
#         num_res_units=2,
#         norm=Norm.BATCH,
#     ).to(device)


# weight = torch.load("/home/hemin/Downloads/model_swinvit.pt")
# model.load_from(weights=weight)
# print("Using pretrained self-supervied Swin UNETR backbone weights !")


torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
#loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scaler = torch.cuda.amp.GradScaler()


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
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
            
            epoch_iterator_val.set_description("Validate (%d / %d Epoch)" % (epoch, 10.0))  # noqa: B038
        
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()

    return mean_dice_val


def train(train_loader):

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


epoch = 0
max_epoch = 600 
val_interval = 1

post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

dice_val_best = 0.0

log_name = "siwn"
writer = SummaryWriter(log_dir='out/' + log_name)
print('Writing Tensorboard logs to ', 'out/' + log_name)


while epoch < max_epoch:

    try: 
        # Traininig 
        ave_loss = train(train_loader)
        writer.add_scalar('train_dice_loss', ave_loss, epoch)

        # Validation 
        if (epoch) % val_interval == 0:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Epoch) (dice=X.X)", dynamic_ncols=True)
            print("starting the validation phase")
            dice_val = validation(epoch_iterator_val)

            epoch_iterator_train = tqdm(subset_loader, desc="Train validate (X / X Epoch) (dice=X.X)", dynamic_ncols=True)
            metric_train = validation(epoch_iterator_train)

            if dice_val > dice_val_best:
                dice_val_best = dice_val

                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            
            writer.add_scalar("Validation/Average Accuracy:", scalar_value=torch.tensor(dice_val), global_step=epoch+1)
            writer.add_scalar("Train/Average Accuracy:", scalar_value=torch.tensor(metric_train), global_step=epoch+1)

        epoch += 1

    except: 
     torch.save(model.state_dict(), os.path.join(root_dir, "last_epoch_model.pth"))


torch.save(model.state_dict(), os.path.join(root_dir, "last_epoch_model.pth"))
print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {epoch}")