import torch
from torch.utils.data import DataLoader
from model import get_model
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(num_classes=4)
model.load_state_dict(torch.load("best_model.pth"))
model = model.to(device)
model.eval()

test_ds = torch.load("test_dataset.pt") 
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

test_correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        test_correct += (pred == labels).sum().item()

print(f"Son test basarisi: {100 * test_correct / len(test_ds):.2f}%")