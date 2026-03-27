import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CIFAR_CNN

CLASSES = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = CIFAR_CNN()
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.to(DEVICE).eval()

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])
val_set    = datasets.CIFAR10(root="./data", train=False, download=False, transform=val_transforms)
val_loader = DataLoader(val_set, batch_size=64, shuffle=True)

imgs, labels = next(iter(val_loader))
with torch.no_grad():
    preds = model(imgs.to(DEVICE)).argmax(1).cpu()

# Show a 4x4 grid of predictions
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
mean = np.array([0.4914, 0.4822, 0.4465])
std  = np.array([0.2470, 0.2435, 0.2616])

for i, ax in enumerate(axes.flat):
    img = imgs[i].permute(1, 2, 0).numpy()
    img = np.clip(img * std + mean, 0, 1)
    ax.imshow(img)
    color = "green" if preds[i] == labels[i] else "red"
    ax.set_title(f"pred: {CLASSES[preds[i]]}\ntrue: {CLASSES[labels[i]]}", fontsize=7, color=color)
    ax.axis("off")

plt.tight_layout()
plt.savefig("predictions.png")
print("Saved predictions.png")