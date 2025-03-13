import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.metrics import compute_meandice, compute_iou
from monai.transforms import Activations, AsDiscrete
from Dataloader import prepare_semi_supervised

# Load Data
data_dir = "/path/to/dataset"  # Change to your dataset path
train_loader, unlabeled_loader, test_loader = prepare_semi_supervised(data_dir, cache=True)

# MONAI U-Net Model
def get_unet():
    return UNet(
        spatial_dims=3,  
        in_channels=3,  
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).cuda()

# Semi-supervised Training with Mean Teacher
def train_semi_supervised(student, teacher, train_loader, unlabeled_loader, optimizer, epochs=20, alpha=0.99):
    dice_loss = DiceLoss(sigmoid=True)
    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        student.train()
        teacher.eval()

        for (train_batch, unlabeled_batch) in zip(train_loader, unlabeled_loader):
            l_img, l_mask = train_batch["t2w"].cuda(), train_batch["seg"].cuda()
            u_img = unlabeled_batch["t2w"].cuda()

            # Forward pass (supervised)
            pred_labeled = student(l_img)
            supervised_loss = dice_loss(pred_labeled, l_mask)

            # Forward pass (unsupervised)
            with torch.no_grad():
                pseudo_label = teacher(u_img)
            pred_unlabeled = student(u_img)
            consistency_loss = mse_loss(pred_unlabeled, pseudo_label)

            total_loss = supervised_loss + 0.1 * consistency_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update teacher model (EMA)
            for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data

        print(f"Epoch {epoch + 1}, Supervised Loss: {supervised_loss.item()}, Consistency Loss: {consistency_loss.item()}")

# Model Evaluation (Dice & IoU)
def evaluate_model(model, test_loader):
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    model.eval()

    dice_scores, iou_scores = [], []
    post_pred = AsDiscrete(threshold=0.5)
    post_label = AsDiscrete(threshold=0.5)

    with torch.no_grad():
        for batch in test_loader:
            image, label = batch["t2w"].cuda(), batch["seg"].cuda()
            pred = model(image)

            pred = post_pred(pred)
            label = post_label(label)

            dice = compute_meandice(pred, label, include_background=True)
            iou = compute_iou(pred, label)

            dice_scores.append(dice.item())
            iou_scores.append(iou.item())

    return dice_scores, iou_scores

# Save Results to Excel
def save_results_to_excel(dice_scores, iou_scores, filename="results.xlsx"):
    df = pd.DataFrame({"Dice Score": dice_scores, "IoU Score": iou_scores})
    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")

# Initialize Models
student_unet = get_unet()
teacher_unet = get_unet()
teacher_unet.load_state_dict(student_unet.state_dict())

optimizer = optim.Adam(student_unet.parameters(), lr=1e-4)

# Train Model
train_semi_supervised(student_unet, teacher_unet, train_loader, unlabeled_loader, optimizer)

# Evaluate Model
dice_scores, iou_scores = evaluate_model(student_unet, test_loader)

# Save Evaluation Metrics
save_results_to_excel(dice_scores, iou_scores)
