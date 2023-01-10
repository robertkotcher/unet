import logging
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import optim

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
#  [x] what is cross entropy loss?
#           ~ a different way to measure loss that gives preference to really bad predictions
#             (SSR is linear and only gives slight preference to really bad predictions)
#           ~ Use -log(x), where "x" is the value after applying softmax
#             compare with "cross entropy", which sums -p(x)log(x), but since p(x) is 0 for
#             all other terms, we only care about the entropy of current label
#           ~ see Josh Starmer's cross entropy video
#  [ ] how does a "learning rate scheduler" (optim.lr_scheduler) work?
#  [ ] what is --amp? (mixed-precision training?) (not currently using)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device {device}")

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
    ''')

    # 2. set up optimizer and loss function
    #       todo: set up optim.lr_scheduler to reduce lr on plateau (see https://github.com/milesial/Pytorch-UNet/blob/master/train.py)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, foreach=True)
    loss = nn.CrossEntropyLoss()

    # 3. Perform training
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{num_epochs}', unit='img') as pbar:
            for batch in train_loader:
                continue


if __name__ == "__main__":
    # depth tells us how many times we want to max pool - if dimensionality doesn't work out,
    # then UNet will complain on forward pass
    model = UNet(in_channels=3, out_channels=1, side=100, depth=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    sds=SaltDataset(
        "/content/drive/MyDrive/deep_learning/salt_identification_challenge/train/images",
        "/content/drive/MyDrive/deep_learning/salt_identification_challenge/train/masks"
    )

    model.to(device)

    train_model(model, sds, device)

