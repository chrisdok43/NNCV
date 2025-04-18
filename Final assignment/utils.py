import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    ColorJitter,
)
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
import torch.quantization as tq

def compute_pixel_accuracy(preds, labels, ignore_index=255):
    mask = labels != ignore_index
    correct = (preds[mask] == labels[mask]).sum()
    total = mask.sum()
    return (correct.float() / total.float()).item()

def compute_mIoU(preds, labels, num_classes=19, ignore_index=255):
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    return sum(i for i in ious if not torch.isnan(torch.tensor(i))) / len([i for i in ious if not torch.isnan(torch.tensor(i))])


def convert_to_train_id(label_img: torch.Tensor, id_to_trainid: dict) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

def convert_train_id_to_color(prediction: torch.Tensor, train_id_to_color: dict) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image

image_transform = Compose([
    ToImage(),
    Resize((256, 256)),
    ToDtype(torch.float32, scale=True),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    Normalize(mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)),
])

label_transform = Compose([
    ToImage(),
    Resize((256, 256), interpolation=0),
    ToDtype(torch.int64),
    lambda x: x.squeeze(0),  # Squeeze the channel dimension if present
])

def custom_transform(img, targets, id_to_trainid):
    semantic_target = targets
    img = image_transform(img)
    semantic_target = label_transform(semantic_target)
    semantic_target = convert_to_train_id(semantic_target, id_to_trainid)
    return img, semantic_target


def visualize_prediction(image, mask, prediction):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(mask.cpu().squeeze(), cmap='tab20')
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(prediction.cpu().squeeze(), cmap='tab20')
    plt.axis("off")
    
    plt.show()

def compute_mean_DICE_old(preds, labels, num_classes=19, ignore_index=255):
    dices = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().float()
        pred_sum = pred_inds.sum().float()
        target_sum = target_inds.sum().float()
        denom = pred_sum + target_sum
        if denom == 0:
            dices.append(float('nan'))  # No ground truth or prediction
        else:
            dice = (2.0 * intersection) / denom
            dices.append(dice.item())
    return sum(i for i in dices if not torch.isnan(torch.tensor(i))) / len([i for i in dices if not torch.isnan(torch.tensor(i))])


import torch

def compute_mean_DICE(predictions, labels, num_classes=19, ignore_index=255, epsilon=1e-6):

    predictions = predictions.contiguous().view(-1)
    labels = labels.contiguous().view(-1)

    dice_per_class = []

    for cls in range(num_classes):
        if cls != ignore_index:
            pred_inds = predictions == cls
            label_inds = labels == cls

            # Exclude pixels with ignore_index in labels
            valid_mask = labels != ignore_index

            pred_inds = pred_inds[valid_mask]
            label_inds = label_inds[valid_mask]

            intersection = (pred_inds & label_inds).float().sum()
            union = pred_inds.float().sum() + label_inds.float().sum()

            dice = (2.0 * intersection + epsilon) / (union + epsilon)

            dice_per_class.append(dice.item())

    mean_dice = sum(dice_per_class) / len(dice_per_class)

    return mean_dice


def visualize_image_tensor(image_tensor):
    # Check if the tensor has 3 dimensions (CxHxW or HxW)
    if image_tensor.ndimension() == 3:
        # If the tensor has 3 channels (RGB image)
        if image_tensor.size(0) == 3:
            # Convert the tensor from CxHxW to HxWxC (for matplotlib)
            image_tensor = image_tensor.permute(1, 2, 0)
        # If the tensor has 1 channel (grayscale image)
        elif image_tensor.size(0) == 1:
            # Squeeze the channel dimension
            image_tensor = image_tensor.squeeze(0)
        # Convert the tensor to a numpy array
        image = image_tensor.cpu().numpy()
        # Plot the image using matplotlib
        plt.imshow(image)
        plt.axis('off')  # Turn off axis
        plt.show()
    elif image_tensor.ndimension() == 1:
        # If the tensor is 1D, plot it as a line graph
        plt.plot(image_tensor.cpu().numpy())
        plt.show()


semantic_to_polygon = {
    'road': ['road', 'rectification border'],
    'sidewalk': ['sidewalk'],
    'building': ['building'],
    'wall': ['wall'],
    'fence': ['fence', 'guard rail'],
    'pole': ['pole', 'polegroup'],
    'traffic light': ['traffic light'],
    'traffic sign': ['traffic sign'],
    'vegetation': ['vegetation'],
    'terrain': ['terrain'],
    'sky': ['sky'],
    'person': ['person', 'persongroup'],
    'rider': ['rider', 'ridergroup'],
    'car': ['car', 'cargroup'],
    'truck': ['truck'],
    'bus': ['bus'],
    'train': ['train'],
    'motorcycle': ['motorcycle', 'motorcyclegroup'],
    'bicycle': ['bicycle', 'bicyclegroup'],
    'license plate': ['license plate'],
    'dynamic': ['dynamic'],
    'static': ['static'],
    'parking': ['parking'],
    'caravan': ['caravan'],
    'rail track': ['rail track'],
    'tunnel': ['tunnel'],
    'ego vehicle': ['ego vehicle'],
    'out of roi': ['out of roi'],
    'trailer': ['trailer'],
    'ground': ['ground'],
    'bridge': ['bridge']
}


polygon_to_semantic = {
    'rectification border': 'road',
    'dynamic': 'dynamic',
    'static': 'static',
    'out of roi': 'out of roi',
    'road': 'road',
    'sidewalk': 'sidewalk',
    'building': 'building',
    'wall': 'wall',
    'fence': 'fence',
    'guard rail': 'fence',
    'pole': 'pole',
    'polegroup': 'pole',
    'traffic light': 'traffic light',
    'traffic sign': 'traffic sign',
    'vegetation': 'vegetation',
    'terrain': 'terrain',
    'sky': 'sky',
    'person': 'person',
    'persongroup': 'person',
    'rider': 'rider',
    'ridergroup': 'rider',
    'car': 'car',
    'cargroup': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'train': 'train',
    'motorcycle': 'motorcycle',
    'motorcyclegroup': 'motorcycle',
    'bicycle': 'bicycle',
    'bicyclegroup': 'bicycle',
    'ground': 'ground',
    'license plate': 'license plate',
    'ego vehicle': 'ego vehicle',
    'bridge': 'bridge',
    'rail track': 'rail track',
    'parking': 'parking',
    'caravan': 'caravan',
    'trailer': 'trailer',
    'tunnel': 'tunnel'
}

class RareClassBoostedDataset(Dataset):

    def __init__(self, base_dataset, id_to_trainid, rare_train_ids: set, ultra_rare_train_ids: set, rare_sample_multiplier: int, ultra_rare_sample_multiplier: int):
        
        
        self.base_dataset = base_dataset # The base dataset to wrap
        self.id_to_trainid = id_to_trainid # Mapping from IDs to train IDs
        self.rare_train_ids = rare_train_ids # The set of rare train IDs to boost
        self.ultra_rare_train_ids = ultra_rare_train_ids # The set of ultra-rare train IDs to boost
        self.rare_sample_multiplier = rare_sample_multiplier # The multiplier for boosting rare classes
        self.ultra_rare_sample_multiplier = ultra_rare_sample_multiplier
        self.rare_indices = self._find_rare_indices() # Find the indices of rare samples in the base dataset
        self.ultra_rare_indices = self._find_ultra_rare_indices()
        self.indices = list(range(len(base_dataset))) + self.rare_indices * (self.rare_sample_multiplier - 1) + self.ultra_rare_indices * (self.ultra_rare_sample_multiplier - 1)

    def _find_rare_indices(self):
        rare_indices = []
        for idx in range(len(self.base_dataset)):
            _, label_image = self.base_dataset[idx]
            labels_set = set(label_image.flatten().numpy())

            if any(train_id in self.rare_train_ids for train_id in labels_set):
                rare_indices.append(idx)

        print(f"Found {len(rare_indices)} rare samples out of {len(self.base_dataset)} total.")
        return rare_indices
    
    def _find_ultra_rare_indices(self):
        ultra_rare_indices = []
        for idx in range(len(self.base_dataset)):
            _, label_image = self.base_dataset[idx]
            labels_set = set(label_image.flatten().numpy())

            if any(train_id in self.ultra_rare_train_ids for train_id in labels_set):
                ultra_rare_indices.append(idx)

        print(f"Found {len(ultra_rare_indices)} ultra-rare samples out of {len(self.base_dataset)} total.")
        return ultra_rare_indices


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = self.indices[index]
        return self.base_dataset[real_index]


def dice_loss_old(outputs, targets, smooth=1e-5):
    # outputs: [B, C, H, W] (logits), targets: [B, H, W] (int class IDs)
    num_classes = outputs.shape[1]
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    outputs_soft = F.softmax(outputs, dim=1)
    intersection = (outputs_soft * targets_one_hot).sum(dim=(0, 2, 3))
    outputs_sum = outputs_soft.sum(dim=(0, 2, 3))
    targets_sum = targets_one_hot.sum(dim=(0, 2, 3))
    dice_coeff = (2. * intersection + smooth) / (outputs_sum + targets_sum + smooth)
    return 1 - dice_coeff.mean()


def dice_loss(pred, target, smooth=1e-5, ignore_index=255):

    # Get class probabilities with softmax
    pred = torch.softmax(pred, dim=1)
    num_classes = pred.shape[1]

    # Create a mask of valid pixels (those not equal to ignore_index)
    valid_mask = (target != ignore_index)
    
    # Avoid out-of-bound indices by replacing ignore_index with a valid index (e.g., 0)
    target_fixed = target.clone()
    target_fixed[~valid_mask] = 0  # Replace ignore values with 0 for one-hot encoding

    # One-hot encode the target_fixed: resulting shape will be (N, H, W, C)
    target_onehot = torch.eye(num_classes, device=pred.device)[target_fixed]
    # Rearrange to (N, C, H, W)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    # Expand valid_mask to have the same shape as the predictions and one-hot encoded target
    valid_mask = valid_mask.unsqueeze(1).float()

    # Zero out the contributions from ignore_index pixels
    pred = pred * valid_mask
    target_onehot = target_onehot * valid_mask

    # Compute intersection and union per class over H and W dimensions
    intersection = (pred * target_onehot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))

    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1 - dice_score.mean()

    return loss


import pandas as pd

def generate_per_class_report(predictions, labels, num_classes=19, class_names=None, previous_report=None):
    report = []
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        label_mask = (labels == cls)

        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()
        total_pred = pred_mask.sum().item()
        total_gt = label_mask.sum().item()

        dice_denominator = total_pred + total_gt
        iou_denominator = union
        acc_denominator = total_gt

        dice = (2 * intersection) / (dice_denominator + 1e-6) if dice_denominator != 0 else None
        iou = intersection / (iou_denominator + 1e-6) if iou_denominator != 0 else None
        accuracy = intersection / (acc_denominator + 1e-6) if acc_denominator != 0 else None

        prev_dice = None
        delta_dice = None
        if previous_report is not None:
            prev_row = previous_report[previous_report["Class ID"] == cls]
            if not prev_row.empty:
                prev_dice = prev_row.iloc[0]["DICE"]
                if isinstance(prev_dice, (int, float)) and prev_dice != 0:
                    delta_dice = ((dice - prev_dice) / (prev_dice + 1e-6)) * 100.0
                else:
                    delta_dice = None

        report.append({
            "Class ID": cls,
            "Class Name": class_names[cls] if class_names else f"Class {cls}",
            "Accuracy": round(accuracy, 4) if accuracy is not None else "N/A",
            "IoU": round(iou, 4) if iou is not None else "N/A",
            "DICE": round(dice, 4) if dice is not None else "N/A",
            "GT Pixels": total_gt,
            "Pred Pixels": total_pred,
            "Intersection": intersection,
            "Prev DICE": round(prev_dice, 4) if prev_dice is not None else "N/A",
            "Delta DICE (%)": round(delta_dice, 2) if delta_dice is not None else "N/A",
        })

    pd.set_option('display.max_rows', None) 
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', None) 
    pd.set_option('display.max_colwidth', None)  

    df = pd.DataFrame(report)
    print("\n📊 Per-Class Metrics Report")
    print(df)
    
    return df




def focal_loss(inputs, targets, alpha=None, gamma=2.0, ignore_index=255):

    logpt = -F.cross_entropy(
        inputs, targets, weight=alpha, ignore_index=ignore_index, reduction="none"
    )
    pt = torch.exp(logpt)
    focal_loss = ((1 - pt) ** gamma) * (-logpt)

    # Mask ignored indices
    mask = targets != ignore_index
    focal_loss = focal_loss * mask  # Set loss to 0 for ignored pixels
    loss = focal_loss.sum() / mask.sum()  # Normalize by valid pixels

    return loss

def combined_loss(inputs, targets, alpha=0.5, class_weights=None, focal_gamma=2.0,
                  ignore_index=255, dice_smooth=1e-5):
    
    fl = focal_loss(inputs, targets, alpha=class_weights, gamma=focal_gamma, ignore_index=ignore_index)
    #dl = dice_loss(inputs, targets, smooth=dice_smooth, ignore_index=ignore_index)
    ce = F.cross_entropy(inputs, targets, weight=class_weights, ignore_index=ignore_index)

    #return alpha * fl + (1 - alpha) * dl
    return alpha * fl + (1 - alpha) * ce


# def apply_post_training_quantization(model, calib_loader, device):
#     model.eval()
#     model = fuse_resnet_backbone(model)
#     model.qconfig = tq.get_default_qconfig('fbgemm')
#     tq.prepare(model, inplace=True)
#     with torch.no_grad():
#         for i, (images, _) in enumerate(calib_loader):
#             if i >= 10: break
#             model(images.to(device))
#     quantized_model = tq.convert(model, inplace=False)
#     return quantized_model

# def apply_qat(model, train_loader, device, epochs, criterion, optimizer):
#     model.train()
#     model = fuse_resnet_backbone(model)
#     model.qconfig = tq.get_default_qat_qconfig('fbgemm')
#     tq.prepare_qat(model, inplace=True)
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for images, masks in train_loader:
#             images, masks = images.to(device), masks.to(device)
#             outputs = model(images)['out']
#             loss = criterion(outputs, masks)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f"[QAT] Epoch {epoch+1}/{epochs}: loss={running_loss/len(train_loader):.4f}")
#     model.eval()
#     quantized_model = tq.convert(model.cpu(), inplace=False)
#     return quantized_model