import logging
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import optim

from unet import UNet
from dataset import SaltDataset

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

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(
    model,
    dataset=None,
    device='cuda',
    num_epochs=10,
    val_percent=0.1,
    batch_size=1,
    lr=1e-5,

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
    criterion = nn.CrossEntropyLoss()

    # 3. Perform training
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{num_epochs}', unit='img') as pbar:
            for (images, masks) in train_loader:
                assert images.shape[1] == model.in_channels, \
                    f"Unet was defined to take {images.shape[1]} input channels, but training images have {images.shape[1]}"

                # channels_last to make better use of locality? Just a guess..
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                masks = masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=False):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, masks)
                    # loss += dice_loss(
                    #     F.softmax(masks_pred, dim=1).float(),
                    #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    #     multiclass=True
                    # )
                
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), False)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # do validation after this....
                # but I'll do this when I see that training on training set is working


if __name__ == "__main__":
    # depth tells us how many times we want to max pool - if dimensionality doesn't work out,
    # then UNet will complain on forward pass
    model = UNet(in_channels=3, out_channels=1, side=100, depth=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    sds=SaltDataset(
        "/Users/robertkotcher/Synthesis/unet/tgs-salt-identification-challenge/train/images",
        "/Users/robertkotcher/Synthesis/unet/tgs-salt-identification-challenge/train/masks"
    )

    model.to(device)

    train_model(model, sds, device)

