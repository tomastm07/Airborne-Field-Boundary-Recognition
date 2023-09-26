#%%
import torch
from dataset import DroneDataset
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

#%%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# %%
def plot_result(image, label, pred, trial):
    trial_dir = f"../result/trial{trial.number}"
    os.mkdir(trial_dir)
    for sm in range(image.shape[0]):
        sample_no = sm
        gt = (image[sample_no]/255.0).permute(1,2,0).detach().cpu()
        lab1 = label.squeeze(1)[sample_no].detach().cpu()
        img1 = torch.sigmoid(pred[sample_no].squeeze(0).detach().cpu())
        m = 0.02
        gridspec_kw = dict(left=m, right=1 - m, top=1 - m, bottom=m, hspace=m, wspace=m)
        fig, ax = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw=gridspec_kw)
        for axis in ax.flat:
            axis.axis('off')
        ax[0].imshow(gt)
        ax[0].set_title("RGB")
        ax[1].imshow(lab1, cmap='Greys')
        ax[1].set_title("Ground Truth")
        ax[2].imshow(img1, cmap='Greys')
        ax[2].set_title("Edge Prediction")        
        plt.savefig(f"{trial_dir}/trial{trial.number}_pred{sample_no}.jpg", bbox_inches='tight')

def sobel_kernel(k=3):
    sobel_1D = torch.linspace(-(k // 2), k // 2, k)
    x, y = torch.meshgrid(sobel_1D, sobel_1D)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    sobel_xy = torch.stack((sobel_2D.T, sobel_2D)).reshape(2, 1, k, k)
    return sobel_xy

def get_pyramid(mask):
    stack_height = 6
    SOBEL = nn.Conv2d(1, 2, 1, padding=1, padding_mode='replicate', bias=False)
    SOBEL.weight.requires_grad = False
    SOBEL.weight.set_(sobel_kernel(k=3))
    SOBEL = SOBEL.to(DEVICE)
    with torch.no_grad():
        masks = [mask]
        ## Build mip-maps
        for _ in range(stack_height):
            # Pretend we have a batch
            big_mask = masks[-1]
            small_mask = F.avg_pool2d(big_mask, 2)
            masks.append(small_mask)

        targets = []
        for mask in masks:
            sobel = torch.any(SOBEL(mask) != 0, dim=1, keepdims=True).float()
            #targets.append(sobel)
            targets.append(torch.cat([mask, sobel], dim=1))

    return targets

def auto_weight_BCE(y_hat_log, y):
    with torch.no_grad():
        beta = y.mean(dim=[2, 3], keepdims=True)
    logit_1 = F.logsigmoid(y_hat_log)
    logit_0 = F.logsigmoid(-y_hat_log)
    loss = -(1 - beta) * logit_1 * y \
           - beta * logit_0 * (1 - y)
    return loss.mean()
# %%

def get_data(image_dir, mask_dir, trainBS, testBS):
    TRANSFORM = A.Compose(
            [   
                A.Resize(height=320, width=320),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                ToTensorV2(),
            ],
        )

    DATASET = DroneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=TRANSFORM)
    test_size = int(0.1*len(DATASET))
    train_size = len(DATASET) - test_size
    train_set, val_set = random_split(DATASET, [train_size, test_size], generator=torch.Generator().manual_seed(10))
    train_loader = DataLoader(train_set, batch_size=trainBS, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=testBS, pin_memory=True, shuffle=False)
    #len(val_set), len(train_set)
    return train_loader, val_loader

# %%
def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler):
    total_loss = 0
    for _, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data/1.0
        data = data.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        y_hat, y_hat_levels = model(data)
        target = get_pyramid(labels)
        loss_levels = []
        for y_hat_el, y in zip(y_hat_levels, target):
            loss_levels.append(criterion(y_hat_el, y))

        # Overall Loss
        loss_final = criterion(y_hat, target[0])
        # Pyramid Losses (Deep Supervision)
        loss_deep_super = torch.sum(torch.stack(loss_levels))
        loss = loss_final + loss_deep_super
        
        target = target[0]
        edge_pred = (y_hat[:, 0] > 0).float()        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    final_loss = total_loss/len(train_loader)
    return final_loss
# %%

def validate(model, val_loader, criterion, device):
    with torch.no_grad():
        total_loss = 0
        for _, (data, labels) in enumerate(val_loader):
            data = data/1.0
            data = data.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(device)
            y_hat, _ = model(data)
            edge_pred = (y_hat[:, 0] > 0).float()
            loss = criterion(y_hat, labels)
            total_loss += loss.item()
        final_loss = total_loss/len(val_loader)
    
    return final_loss, edge_pred, data, labels


# %%
def dice_coefficient(test_pred, test_labels):
    dice_score = 0
    preds = torch.sigmoid(test_pred)
    preds = (preds > 0.5).float()
    dice_score = (2 * (preds * test_labels).sum()) / (
        (preds + test_labels).sum() + 1e-8
    )
    return dice_score