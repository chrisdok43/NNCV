wandb login

python3 train_DeepLabV3_augmented.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --backbone deeplabv3_resnet50 \
    --experiment-id "DL-RSN50-HIGH-AUGM-CE" \
