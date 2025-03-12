import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.utils.data import Dataset, DataLoader

# Custom Dataset for NIfTI images
class NIfTIDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx]).get_fdata()
        img = np.expand_dims(img, axis=0)  # Add channel dim
        
        if self.mask_paths:
            mask = nib.load(self.mask_paths[idx]).get_fdata()
            mask = np.expand_dims(mask, axis=0)  # Keep same shape
        else:
            mask = np.zeros_like(img)  # Placeholder for unlabeled data

        img, mask = torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
        
        if self.transform:
            img, mask = self.transform(img), self.transform(mask)

        return img, mask

# MONAI U-Net Model
def get_unet():
    return UNet(
        spatial_dims=2,  # Change to 3 if working with full 3D volumes
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).cuda()

# Semi-supervised Training with Mean Teacher
def train_semi_supervised(student, teacher, labeled_loader, unlabeled_loader, optimizer, epochs=20, alpha=0.99):
    dice_loss = DiceLoss(sigmoid=True)
    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        student.train()
        teacher.eval()

        for (l_img, l_mask), (u_img, _) in zip(labeled_loader, unlabeled_loader):
            l_img, l_mask, u_img = l_img.cuda(), l_mask.cuda(), u_img.cuda()

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

            # Update teacher using Exponential Moving Average (EMA)
            for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data

        print(f"Epoch {epoch + 1}, Supervised Loss: {supervised_loss.item()}, Consistency Loss: {consistency_loss.item()}")

# Load Data
labeled_images = ["path_to_labeled_image.nii.gz"]
labeled_masks = ["path_to_labeled_mask.nii.gz"]
unlabeled_images = ["path_to_unlabeled_image.nii.gz"]

labeled_dataset = NIfTIDataset(labeled_images, labeled_masks)
unlabeled_dataset = NIfTIDataset(unlabeled_images)

labeled_loader = DataLoader(labeled_dataset, batch_size=2, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=2, shuffle=True)

# Initialize Student and Teacher Models
student_unet = get_unet()
teacher_unet = get_unet()
teacher_unet.load_state_dict(student_unet.state_dict())

optimizer = optim.Adam(student_unet.parameters(), lr=1e-4)

# Train Model
train_semi_supervised(student_unet, teacher_unet, labeled_loader, unlabeled_loader, optimizer)
