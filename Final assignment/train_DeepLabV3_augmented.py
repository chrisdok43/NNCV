
import os
from argparse import ArgumentParser
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
import torchvision.models.segmentation as models
from utils import *
from collections import Counter
import copy
import torch.nn.utils.prune as pr
import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet101, deeplabv3_resnet50, deeplabv3_mobilenet_v3_large,
    DeepLabV3_ResNet101_Weights, DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
)

semantic_label_to_id = { i.name: i.id for i in Cityscapes.classes }
print(semantic_label_to_id)
semantic_label_to_train_id = { i.name: i.train_id for i in Cityscapes.classes }
print(semantic_label_to_train_id)

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

# Mapping polygon labels to semantic labels
def get_label_counter():

    dataset = Cityscapes(
    root=args.data_dir,
    split="train",
    mode="fine",
    target_type="polygon",
    transforms=None
    )

    # Initialize a Counter to tally occurrences of each label
    label_counter = Counter()

    # Iterate over the dataset with a progress bar
    for i in range(len(dataset)):
        _, target = dataset[i]

        # Expecting target to be a dictionary with key "objects"
        objects = target.get("objects", [])
        
        for obj in objects:
            # Get the label
            label = obj.get("label", None)
            semantic_label = polygon_to_semantic.get(label, None)
            if semantic_label is None:
                continue

            if label:
                label_counter[semantic_label] += 1

    return label_counter

def get_class_weights(device):

    label_counter = get_label_counter()
    train_id_counter = Counter()
    for label, count in label_counter.items():
        train_id = semantic_label_to_train_id.get(label, None)
        if train_id is not None:
            train_id_counter[train_id] += count

    # remove the 255 id from the train_id_counter
    train_id_counter.pop(255, None)
    # remove the -1 id from the train_id_counter
    train_id_counter.pop(-1, None)

    num_classes = 19  # Cityscapes classes
    class_counts = np.array([train_id_counter.get(i, 0) for i in range(num_classes)])
    print("class_counts", class_counts)
    # Compute class weights (inverse frequency)
    class_weights = 1 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    return class_weights_tensor

def get_rare_classes(num = 2000):

    # Get the label counter
    label_counter = get_label_counter()

    low_count_labels = {label: count for label, count in label_counter.items() if count < num}
    #low_count_ids = {semantic_label_to_id[label]: count for label, count in low_count_labels.items() if label in semantic_label_to_id}
    low_count_train_ids = {semantic_label_to_train_id[label]: count for label, count in low_count_labels.items() if label in semantic_label_to_id}
    rare_train_ids = set(low_count_train_ids.keys())  
    rare_train_ids.discard(255)  # Remove the ignored label ID
    print("Rare train IDs:", rare_train_ids)
    return rare_train_ids

def custom_transform(img, semantic_target):
    img = image_transform(img)
    semantic_target = label_transform(semantic_target)
    semantic_target = convert_to_train_id(semantic_target, id_to_trainid)
    return img, semantic_target

def train(loader, model, optimizer, criterion, epoch, device=None):
    
    model.train()
    running_loss = 0.0

    for i, (images, masks) in enumerate(loader):
        print(f"Batch {i+1}/{len(loader)}")
        images = images.to(device)
        masks = masks.to(device)

        
        model_name = model.__class__.__name__
        print(f"Model name: {model_name}")
        if model_name == "DeepLabV3":
            outputs = model(images)["out"]
        else:   
            outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(loader) + i)

        running_loss += loss.item()
        print(f"Batch {i+1}/{len(loader)}, Loss: {loss.item()}")
    
    avg_loss = running_loss / len(loader)
    wandb.log({"avg_train_loss": avg_loss}, step=(epoch + 1) * len(loader) - 1)
    return avg_loss

@torch.no_grad()
def validate(model, loader, criterion, epoch, train_loader_len, report_df, prune_amount=0, quant_mode=None ,device=None):
    print("Validating...")

    val_model = copy.deepcopy(model)

    if quant_mode == "ptq":
        print("Applying post-training quantization...")
        total_pre, nonzero_pre = count_params(eval_model)
        size_pre_mb = total_pre * 4 / (1024**2)  # float32: 4 bytes each
        wandb.log({
            "total_params_pre_quant": total_pre,
            "nonzero_params_pre_quant": nonzero_pre,
            "model_size_pre_quant_MB": size_pre_mb,
        }, step=(epoch + 1) * train_loader_len - 1)
        print(f"Before quantization: {nonzero_pre}/{total_pre} non-zero "
            f"({100 * nonzero_pre/total_pre:.2f}%), size ≈ {size_pre_mb:.2f} MB")
        #eval_model = quantize_model_gpu(eval_model, device=device)

        eval_model = quantize_model_ptq(eval_model, loader, device='cpu')
        
        total_post, nonzero_post = count_params(eval_model)
        size_post_mb = total_post * 4 / (1024**2)
        wandb.log({
            "total_params_post_quant": total_post,
            "nonzero_params_post_quant": nonzero_post,
            "model_size_post_quant_MB": size_post_mb
        }, step=(epoch + 1) * train_loader_len - 1)
        print("total_post", total_post)
        print("nonzero_post", nonzero_post)
        
        # print(f"After quantization:  {nonzero_post}/{total_post} non-zero "
        #         f"({100 * nonzero_post/total_post:.2f}%), size ≈ {size_post_mb:.2f} MB")
        print("Quantization applied.")


    if prune_amount > 0:
        total_pre, nonzero_pre = count_params(eval_model)
        size_pre_mb = total_pre * 4 / (1024**2)  # float32: 4 bytes each
        wandb.log({
            "total_params_pre_prune": total_pre,
            "nonzero_params_pre_prune": nonzero_pre,
            "model_size_pre_prune_MB": size_pre_mb,
        }, step=(epoch + 1) * train_loader_len - 1)
        print(f"Before pruning: {nonzero_pre}/{total_pre} non-zero "
            f"({100 * nonzero_pre/total_pre:.2f}%), size ≈ {size_pre_mb:.2f} MB")
        print(f"Applying unstructured pruning @ {prune_amount*100:.1f}% sparsity.")
        for module in eval_model.modules():
            if isinstance(module, nn.Conv2d):
                # clean up any existing masks
                if hasattr(module, 'weight_orig'):
                    prune.remove(module, 'weight')
                # apply new prune mask
                prune.l1_unstructured(module, name='weight', amount=prune_amount)

        # re‑eval mode to freeze masks / batchnorm
        eval_model.eval()

        total_post, nonzero_post = count_params(eval_model)
        size_post_mb = total_post * 4 / (1024**2)
        wandb.log({
            "total_params_post_prune": total_post,
            "nonzero_params_post_prune": nonzero_post,
            "model_size_post_prune_MB": size_post_mb,
            "pruned_sparsity": 1 - nonzero_post / total_post
        }, step=(epoch + 1) * train_loader_len - 1)
        print(f"After pruning:  {nonzero_post}/{total_post} non-zero "
              f"({100 * nonzero_post/total_post:.2f}%), size ≈ {size_post_mb:.2f} MB")
    else:
        print("No pruning applied.")


    eval_model = eval_model.to(device).eval()
    total_loss = 0.0
    total_acc, total_iou, total_dice = 0, 0, 0

    for i, (images, labels) in enumerate(loader):
        print(f"Batch in Validating {i+1}/{len(loader)}")

        images = images.to(device)
        params = list(eval_model.parameters())
        if params and params[0].dtype == torch.float16:
            images = images.half()

        labels = labels.to(device)

        model_name = eval_model.__class__.__name__
        if model_name == "DeepLabV3":
            outputs = eval_model(images)["out"]
        else:   
            outputs = eval_model(images)  
                  
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        predictions = outputs.argmax(1)
        acc = compute_pixel_accuracy(predictions, labels)
        miou = compute_mIoU(predictions, labels)
        dice = compute_mean_DICE(predictions, labels)

        total_acc += acc
        total_iou += miou
        total_dice += dice

        if i == len(loader) - 1:
            print("Generating per-class report...")
            res_df = generate_per_class_report(predictions.cpu(), labels.cpu(), num_classes=19, class_names=[c.name for c in Cityscapes.classes if c.train_id != 255], previous_report=report_df)

        if i == 0:
            predictions = outputs.softmax(1).argmax(1)

            predictions = predictions.unsqueeze(1)
            labels = labels.unsqueeze(1)

            predictions = convert_train_id_to_color(predictions, train_id_to_color)
            labels = convert_train_id_to_color(labels, train_id_to_color)

            predictions_img = make_grid(predictions.cpu(), nrow=8)
            labels_img = make_grid(labels.cpu(), nrow=8)

            predictions_img = predictions_img.permute(1, 2, 0).numpy()
            labels_img = labels_img.permute(1, 2, 0).numpy()

            wandb.log({
                "predictions": [wandb.Image(predictions_img)],
                "labels": [wandb.Image(labels_img)],
            }, step=(epoch + 1) * train_loader_len - 1)

    print(f"Validation Loss: {total_loss / len(loader)}")
    print(f"Pixel Accuracy: {total_acc / len(loader)}")
    print(f"Mean IoU: {total_iou / len(loader)}")
    print(f"Mean DICE: {total_dice / len(loader)}")

    valid_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    avg_miou = total_iou / len(loader)
    avg_dice = total_dice / len(loader)

    wandb.log({
        "valid_loss": valid_loss,
        "pixel_accuracy": avg_acc,
        "mIoU": avg_miou,
        "mean_DICE": avg_dice
    }, step=(epoch + 1) * train_loader_len - 1)
        
    return valid_loss, avg_dice, res_df


def get_segmentation_model(
        model_name: str,
        num_classes: int = 19,
        pretrained: bool = True,
        device: torch.device = None
    ) -> nn.Module:
    
    name = model_name.lower()
    if name == "deeplabv3_resnet101":
        weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
        model = deeplabv3_resnet101(weights=weights)
    elif name == "deeplabv3_resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        model = deeplabv3_resnet50(weights=weights)
    elif name == "deeplabv3_mobilenet_v3_large":
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = deeplabv3_mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'")

    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    if device is not None:
        model = model.to(device)
    return model

def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--backbone", type=str, default="deeplabv3_resnet101", help="Model backbone")
    parser.add_argument("--quant_mode", choices=["none", "ptq", "qat"], default="none", help="Quantization mode")
    parser.add_argument("--prune", type=float, default=0, help="Unstructured pruning percentage")

    return parser


def main(args):

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    wandb.init(
        project="5lsm0-cityscapes-segmentation", 
        name=args.experiment_id,  
        config=vars(args),  
    )

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting creation of the datasets...")
    train_dataset_raw = Cityscapes(
        args.data_dir, 
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=custom_transform
    )
    print(f"Loaded {len(train_dataset_raw)} samples from the raw dataset.")

    rare_train_ids = get_rare_classes()

    underperforming_classes = ["train", "wall", "fence", "bus", "traffic light", "pole", "rider"]
    underperforming_classes_train_ids = [semantic_label_to_train_id[label] for label in underperforming_classes if label in semantic_label_to_train_id]

    train_dataset = RareClassBoostedDataset(
        base_dataset=train_dataset_raw,
        id_to_trainid=id_to_trainid,
        rare_train_ids=rare_train_ids,
        ultra_rare_train_ids=underperforming_classes_train_ids,
        rare_sample_multiplier=3,
        ultra_rare_sample_multiplier=4
    )

    print(f"Number of samples in the original dataset: {len(train_dataset_raw)}")

    print(f"Number of samples in the boosted dataset: {len(train_dataset)}")

    val_dataset = Cityscapes(
        args.data_dir, 
        split="val",
        mode="fine",
        target_type="semantic",
        transforms=custom_transform  
    )

    # train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    # val_dataset = wrap_dataset_for_transforms_v2(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = get_segmentation_model(
        model_name=args.backbone,      
        num_classes=19,
        pretrained=True,
        device=device
    )   

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)

    wandb.log({
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        "model/size_MB": model_size_mb,
    })

    # Define the loss function
    # criterion = nn.CrossEntropyLoss(ignore_index=255)
    # criterion = dice_loss
    # criterion = combined_loss
    # criterion = focal_loss
    
    # Get class weights
    class_weights_tensor = get_class_weights(device)
    print("Class weights:", class_weights_tensor)

    criterion = lambda inputs, targets: combined_loss(
        inputs, targets, 
        alpha=0.5, 
        class_weights=class_weights_tensor, 
        focal_gamma=2.0, 
        ignore_index=255
    )

    # Define the optimizer
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ], weight_decay=1e-3)

    best_dice = 0.0
    current_best_model_path = None

    report_df = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")
        _ = train(train_loader, model, optimizer, criterion, epoch, device)
        train_loader_len = len(train_loader)
        valid_loss, mean_dice, report_df  = validate(model, val_loader, criterion, epoch, train_loader_len, report_df, args.prune, args.quant_mode, device)
        
        if mean_dice > best_dice:
            best_dice = mean_dice
            if current_best_model_path is not None:
                os.remove(current_best_model_path)
            current_best_model_path = os.path.join(
                output_dir, 
                f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}-dice={mean_dice:04}.pth"
            )
            print(f"Best model saved at {current_best_model_path}")
            torch.save(model.state_dict(), current_best_model_path)

    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
