# =============================================================================
# BASELINE SECUENCIAL: CIFAR-10 + ResNet18
# =============================================================================
# Cómo usarlo en Kaggle:
#   1. Crea un nuevo Notebook en Kaggle (kaggle.com/code -> New Notebook).
#   2. En el panel derecho, Settings -> Accelerator -> elige "GPU T4 x2"
#      (aunque este script solo use 1 GPU, actívalo así porque el siguiente
#      script, el de entrenamiento distribuido, sí necesita las 2).
#   3. Settings -> Internet -> ON (para descargar CIFAR-10 automáticamente).
#   4. Copia cada bloque separado por "# --- CELDA N ---" en una celda
#      distinta del notebook, en orden, y ejecuta.
#
# Qué hace:
#   - Descarga CIFAR-10 (torchvision se encarga).
#   - Adapta ResNet18 (diseñado para ImageNet 224x224) a imágenes de 32x32:
#     cambia el primer conv a 3x3/stride 1 y quita el maxpool inicial.
#     Sin este cambio, ResNet18 reduce demasiado la imagen y el accuracy
#     se queda muy bajo en CIFAR-10.
#   - Entrena de forma 100% secuencial (1 GPU, sin paralelismo).
#   - Guarda tiempos por época y métricas en baseline_metrics.json:
#     esto es lo que vas a comparar después contra la versión distribuida.
# =============================================================================

# --- CELDA 1: imports y configuración ---
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

SEED = 42
NUM_EPOCHS = 15          # súbelo a 30-40 si te alcanza el tiempo de Kaggle
BATCH_SIZE = 128
LR = 0.1
NUM_WORKERS = 2
OUTPUT_METRICS_FILE = "baseline_metrics.json"
OUTPUT_MODEL_FILE = "baseline_resnet18_cifar10.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

# --- CELDA 2: datos (CIFAR-10) ---
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
)
test_loader = DataLoader(
    test_set, batch_size=256, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

print(f"Train: {len(train_set)} imágenes | Test: {len(test_set)} imágenes")

# --- CELDA 3: modelo (ResNet18 adaptado a CIFAR-10) ---
def build_resnet18_cifar(num_classes=10):
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    # ResNet18 estándar está pensado para imágenes de 224x224 (ImageNet).
    # Para 32x32 hay que evitar reducir la resolución demasiado rápido:
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

model = build_resnet18_cifar(num_classes=10).to(device)
print(model)

# --- CELDA 4: loss, optimizer, scheduler ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# --- CELDA 5: funciones de entrenamiento y evaluación ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total

# --- CELDA 6: loop principal de entrenamiento ---
history = {
    "epoch": [], "train_loss": [], "train_acc": [],
    "test_loss": [], "test_acc": [], "epoch_time_sec": [],
}

total_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()

    epoch_time = time.time() - epoch_start

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    history["epoch_time_sec"].append(epoch_time)

    print(f"[Época {epoch:02d}/{NUM_EPOCHS}] "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
          f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% | "
          f"tiempo={epoch_time:.1f}s")

total_time = time.time() - total_start
print(f"\nEntrenamiento secuencial terminado en {total_time:.1f}s "
      f"({total_time/60:.2f} min) para {NUM_EPOCHS} épocas.")

# --- CELDA 7: guardar métricas y modelo (para comparar luego con la versión distribuida) ---
metrics = {
    "mode": "sequential_baseline",
    "num_gpus": 1,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "total_time_sec": total_time,
    "avg_epoch_time_sec": sum(history["epoch_time_sec"]) / len(history["epoch_time_sec"]),
    "final_train_acc": history["train_acc"][-1],
    "final_test_acc": history["test_acc"][-1],
    "history": history,
}

with open(OUTPUT_METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=2)

torch.save(model.state_dict(), OUTPUT_MODEL_FILE)
print(f"Métricas guardadas en {OUTPUT_METRICS_FILE}")
print(f"Modelo guardado en {OUTPUT_MODEL_FILE}")

# --- CELDA 8: graficar curvas ---
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
axes[0].plot(history["epoch"], history["test_loss"], label="Test loss")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss por época (baseline secuencial)")
axes[0].legend()

axes[1].plot(history["epoch"], history["train_acc"], label="Train acc")
axes[1].plot(history["epoch"], history["test_acc"], label="Test acc")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Accuracy por época (baseline secuencial)")
axes[1].legend()

plt.tight_layout()
plt.savefig("baseline_curves.png", dpi=150)
plt.show()

print(f"\nResumen: {NUM_EPOCHS} épocas, {total_time:.1f}s totales, "
      f"{metrics['avg_epoch_time_sec']:.1f}s/época en promedio, "
      f"test_acc final = {metrics['final_test_acc']:.2f}%")
