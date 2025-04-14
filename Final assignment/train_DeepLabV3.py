
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
)
import torchvision.models.segmentation as models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

from torch import optim 
from utils import *

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def custom_transform(img, targets):
    semantic_target = targets
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
def validate(model, loader, criterion, epoch, output_dir, current_best_model_path=None, device=None):
    model.eval()
    total_loss = 0.0
    total_acc, total_iou = 0.0, 0.0

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        model_name = model.__class__.__name__
        if model_name == "DeepLabV3":
            outputs = model(images)["out"]
        else:   
            outputs = model(images)  
                  
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        predictions = outputs.argmax(1)
        acc = compute_pixel_accuracy(predictions, labels)
        miou = compute_mIoU(predictions, labels)
        total_acc += acc
        total_iou += miou

        if i == 0:
            predictions = outputs.softmax(1).argmax(1)

            predictions = predictions.unsqueeze(1)
            labels = labels.unsqueeze(1)

            predictions = convert_train_id_to_color(predictions)
            labels = convert_train_id_to_color(labels)

            predictions_img = make_grid(predictions.cpu(), nrow=8)
            labels_img = make_grid(labels.cpu(), nrow=8)

            predictions_img = predictions_img.permute(1, 2, 0).numpy()
            labels_img = labels_img.permute(1, 2, 0).numpy()

            wandb.log({
                "predictions": [wandb.Image(predictions_img)],
                "labels": [wandb.Image(labels_img)],
            }, step=(epoch + 1) * len(loader) - 1)
    valid_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    avg_miou = total_iou / len(loader)

    wandb.log({
        "valid_loss": valid_loss,
        "pixel_accuracy": avg_acc,
        "mIoU": avg_miou,
    }, step=(epoch + 1) * len(loader) - 1)
        
    return valid_loss


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Cityscapes(
        args.data_dir, 
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=custom_transform
    )
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
        batch_size=8,
        shuffle=True,
        num_workers=0, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )

    # Use the new weights argument
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.deeplabv3_resnet50(weights=weights)

    # Modify the classifier to output 19 classes (for Cityscapes)
    model.classifier[4] = torch.nn.Conv2d(256, 19, kernel_size=(1, 1))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)

    wandb.log({
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        "model/size_MB": model_size_mb,
    })

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Define the optimizer
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ], weight_decay=1e-4)

    best_valid_loss = float('inf')
    current_best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")
        _ = train(train_loader, model, optimizer, criterion, epoch, device)
        valid_loss = validate(model, val_loader, criterion, epoch, output_dir, current_best_model_path, device)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Optionally remove the previous best model if exists
            if current_best_model_path is not None:
                os.remove(current_best_model_path)
            current_best_model_path = os.path.join(
                output_dir, 
                f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
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
