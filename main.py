import logging
import wandb
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
from dice_score import dice_loss

from unet import UNet
from dataset import SaltDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
#                                  TRAINING                                   #
###############################################################################

# To do next:
#  [ ] Implement dataset and dataloader parts --- FIGURE OUT HOW TO LOAD 16-BIT GRAYSCALE
#  [ ] Tweak UNet so that you just specify depth. I'll try this as a way to compensate for
#      images that are too small to fit dimensions from original paper by Ronnenberger et al
#  [ ] Continue implementing training loop
#
#
# To study further:
#  [ ] what is RMSprop optimizer? how does it different from Adam, etc?
#  [ ] how does a "learning rate scheduler" (optim.lr_scheduler) work?
#  [ ] what is --amp? (mixed-precision training?) (not currently using)
#  [ ] what is gradient clipping?

logging.basicConfig(level=logging.INFO, format='[TRAIN] %(levelname)s: %(message)s')

def train_model(
    model,
    dataset=None,
    device='cuda',
    num_epochs=1000,
    val_percent=0.01,
    batch_size=128,
    lr=1e-6,

    # momentum and weight_decay are values used with RMS_prop optimizer
    momentum=0.99,
    weight_decay: float = 1e-8,
):
    # 1. split the input dataset into training / validation, convert to data loader
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args) # drop_last=True drops last "batch", not example

    logging.info(f'''Starting training:
        Epochs:          {num_epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Momentum:        {momentum}
        Device:          {device.type}
        Gradient clipping False
    ''')

    # 2. set up optimizer and loss function
    #       todo: set up optim.lr_scheduler to reduce lr on plateau (see https://github.com/milesial/Pytorch-UNet/blob/master/train.py)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, foreach=True)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    lossfn = nn.BCELoss()

    # 3. Perform training
    _counter=0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{num_epochs}', unit='img') as pbar:
            for (images, masks) in train_loader:
                assert images.shape[1] == model.in_channels, \
                    f"Unet was defined to take {images.shape[1]} input channels, but training images have {images.shape[1]}"

                # channels_last to make better use of locality? Just a guess..
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                masks = masks.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                model = model.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                masks_pred = model(images)
                loss = lossfn(masks_pred.squeeze(), masks.squeeze())

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss': f"{loss.item():.3f}"})
                wandb.log({"loss_train": loss})
                if epoch % 10 == 0:
                    wandb.watch(model)
                    m_pred = wandb.Image(masks_pred, caption="Prediction")
                    m = wandb.Image(masks, caption="Target")
                    wandb.log({
                        "masks_pred": m_pred,
                        "masks_target": m
                    })


                # do validation after this....
                # but I'll do this when I see that training on training set is working

                # visualize performance on the training set
                if _counter % 1000 == 0:
                    print("saving example...")
                    save_image(images[0], f"{_counter}-in.png")
                    save_image(masks[0] * 255, f"{_counter}-label.png")
                    save_image(masks_pred[0] * 255, f"{_counter}-pred.png")
                _counter = _counter + 1



if __name__ == "__main__":
    # depth tells us how many times we want to max pool - if dimensionality doesn't work out,
    # then UNet will complain on forward pass
    model = UNet(in_channels=3, out_channels=1, side=96)
    model = model.to(device=device, memory_format=torch.channels_last)
    logging.info(f'Using device {device}')
    sds=SaltDataset(
        "/home/ubuntu/salt-dataset/train/images",
        "/home/ubuntu/salt-dataset/train/masks",
        transform=transforms.Resize(96)
    )
    wandb.init(project="unet-salt")

    subsetids = list(range(100))
    sds = torch.utils.data.Subset(sds, subsetids)

    train_model(model, sds, device)

