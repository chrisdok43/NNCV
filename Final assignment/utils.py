import torch

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

