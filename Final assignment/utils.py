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
import torch.nn.utils.prune as prune
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop, RandomRotation, ColorJitter, GaussianBlur, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode

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
    RandomHorizontalFlip(p=0.5),
    RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    Normalize(mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)),
    GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
])

label_transform = Compose([
    ToImage(),
    Resize((256, 256), interpolation=0),
    ToDtype(torch.int64),
    lambda x: x.squeeze(0),  
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
            dices.append(float('nan')) 
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
    if image_tensor.ndimension() == 3:
        if image_tensor.size(0) == 3:
            image_tensor = image_tensor.permute(1, 2, 0)
        elif image_tensor.size(0) == 1:
            image_tensor = image_tensor.squeeze(0)
        image = image_tensor.cpu().numpy()
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    elif image_tensor.ndimension() == 1:
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
    num_classes = outputs.shape[1]
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    outputs_soft = F.softmax(outputs, dim=1)
    intersection = (outputs_soft * targets_one_hot).sum(dim=(0, 2, 3))
    outputs_sum = outputs_soft.sum(dim=(0, 2, 3))
    targets_sum = targets_one_hot.sum(dim=(0, 2, 3))
    dice_coeff = (2. * intersection + smooth) / (outputs_sum + targets_sum + smooth)
    return 1 - dice_coeff.mean()


def dice_loss(pred, target, smooth=1e-5, ignore_index=255):

    pred = torch.softmax(pred, dim=1)
    num_classes = pred.shape[1]

    valid_mask = (target != ignore_index)
    
    target_fixed = target.clone()
    target_fixed[~valid_mask] = 0  

    target_onehot = torch.eye(num_classes, device=pred.device)[target_fixed]
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1).float()

    pred = pred * valid_mask
    target_onehot = target_onehot * valid_mask

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
    focal_loss = focal_loss * mask  
    loss = focal_loss.sum() / mask.sum()  

    return loss

def combined_loss(inputs, targets, alpha=0.5, class_weights=None, focal_gamma=2.0,
                  ignore_index=255, dice_smooth=1e-5):
    
    # fl = focal_loss(inputs, targets, alpha=class_weights, gamma=focal_gamma, ignore_index=ignore_index)
    # dl = dice_loss(inputs, targets, smooth=dice_smooth, ignore_index=ignore_index)
    ce = F.cross_entropy(inputs, targets, weight=None, ignore_index=ignore_index)
    
    return ce
    #return alpha * fl + (1 - alpha) * dl
    # return alpha * fl + (1 - alpha) * ce


import torch.nn.utils.prune as prune

def apply_unstructured_pruning(model, amount):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

def apply_structured_pruning(model: torch.nn.Module,
                             amount: float = 0.3,
                             n: int = 1,
                             dim: int = 0):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=n, dim=dim)
    return model

def remove_pruning_reparametrization(model: torch.nn.Module):

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # only remove if pruning was applied
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
    return model


def count_params(model: nn.Module):

    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(int(torch.count_nonzero(p)) for p in model.parameters())
    return total, nonzero


from torch.ao.quantization import fuse_modules

def fuse_deeplabv3_resnet(model):
    fuse_modules(model.backbone,
                          ['conv1', 'bn1', 'relu'],
                          inplace=True)

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        for block in getattr(model.backbone, layer_name):
            fuse_modules(block, ['conv1', 'bn1', 'relu'], inplace=True)
            fuse_modules(block, ['conv2', 'bn2'], inplace=True)
            fuse_modules(block, ['conv3', 'bn3'], inplace=True)

            if block.downsample is not None:
                fuse_modules(block.downsample, ['0', '1'], inplace=True)

    return model

def fuse_deeplabv3_head(model):

    head = model.classifier
    aspp = head[0]                    
    for conv_mod in aspp.convs[:-1]:
        fuse_modules(conv_mod, ['0', '1', '2'], inplace=True)
    fuse_modules(aspp.convs[-1], ['1', '2', '3'], inplace=True)
    fuse_modules(aspp.project, ['0', '1', '2'], inplace=True)
    fuse_modules(head, ['1', '2', '3'], inplace=True)

    return model

from torch.ao.quantization import QConfig
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.quantize_fx import fuse_fx, prepare_fx, convert_fx
from torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor
import torch
from torch.cuda.amp import autocast

def quantize_model_gpu(model, device):

    assert device.type in ("cuda", "mps")
    
    model.eval()
    
    fuse_deeplabv3_resnet(model)
    fuse_deeplabv3_head(model)
    torch.ao.quantization.fuse_modules(model.aux_classifier, ["0", "1", "2"], inplace=True)
    
    model = model.to(device).half()
    
    return model

from torch.ao.quantization import get_default_qconfig, prepare, convert, fuse_modules
import copy

def quantize_model_ptq(model, calib_loader, device, num_calib_batches=10):
    """
    Perform post‐training static quantization (PTQ) on `model` using 
    `calib_loader` for calibration, then move to `device`.
    """
    m = copy.deepcopy(model).cpu().eval()
    fuse_deeplabv3_resnet(m)
    fuse_deeplabv3_head(m)
    fuse_modules(m.aux_classifier, ["0","1","2"], inplace=True)
    m.qconfig = get_default_qconfig("fbgemm")
    prepare(m, inplace=True)
    with torch.no_grad():
        for i, (imgs, _) in enumerate(calib_loader):
            if i >= num_calib_batches:
                break
            m(imgs.cpu())
    convert(m, inplace=True)
    return m.to(device)


def quantize_model_cpu(model, data_path, device):

    torch.backends.quantized.engine = "qnnpack"
    model = model.to(device).eval()

    fuse_deeplabv3_resnet101(model)
    fuse_deeplabv3_head(model)
    torch.ao.quantization.fuse_modules(model.aux_classifier, ['0','1','2'], inplace=True)

    act_fake_quant = FakeQuantize.with_args(
        observer=MinMaxObserver,
        quant_min=0,    
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine
    )
    weight_fake_quant = FakeQuantize.with_args(
        observer=MinMaxObserver,
        quant_min=-127, 
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric
    )

    qcfg = QConfig(activation=act_fake_quant, weight=weight_fake_quant)
    qconfig_dict = {"": qcfg}

    model_fused = fuse_fx(model, qconfig_dict)

    example_inputs = (torch.randn(1,3,256,256,device=device),)
    model_prepared = prepare_fx(model_fused, qconfig_dict, example_inputs)

    def custom_trans(img, target):
        return ToTensor()(img), ToTensor()(target)

    calib_ds = Cityscapes(
        root=data_path,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=custom_trans
    )
    with torch.no_grad():
        for i, (img, _) in enumerate(calib_ds):
            if i >= 10: break
            img = img.unsqueeze(0).to(device)
            model_prepared(img)

    model_quantized = convert_fx(model_prepared)

    model_quantized.to(device)

    return model_quantized