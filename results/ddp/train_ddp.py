
# =============================================================================
# ENTRENAMIENTO DISTRIBUIDO DDP: CIFAR-10 + ResNet18
# =============================================================================
#
# Ejecutar en Kaggle:
#
#   !torchrun --standalone --nproc_per_node=2 train_ddp.py
#
# Configuración Kaggle:
#   Settings -> Accelerator -> GPU T4 x2
#   Settings -> Internet -> ON
#
# Comparación contra baseline:
#
#   Baseline:
#       1 GPU
#       batch global = 128
#
#   DDP:
#       2 GPUs
#       batch global = 128
#       GPU 0 -> batch 64
#       GPU 1 -> batch 64
#
# =============================================================================


# =============================================================================
# IMPORTANTE: CONFIGURACIÓN NCCL
# =============================================================================
#
# Estas variables DEBEN definirse antes de inicializar
# torch.distributed.
#
# Kaggle puede presentar problemas de comunicación P2P entre las T4.
#
# =============================================================================

import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


# =============================================================================
# IMPORTS
# =============================================================================

import json
import random
import time

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

SEED = 42

NUM_EPOCHS = 15

# ============================================================================
# IMPORTANTE
#
# Este es el BATCH GLOBAL.
#
# Con 2 GPUs:
#
#   128 / 2 = 64
#
# GPU 0 -> 64
# GPU 1 -> 64
#
# Esto permite comparar contra el baseline:
#
#   Baseline -> batch 128
#   DDP      -> batch global 128
# ============================================================================

GLOBAL_BATCH_SIZE = 128

LR = 0.1

NUM_WORKERS = 2

RESULTS_DIR = "results_ddp"

MEAN = (0.4914, 0.4822, 0.4465)

STD = (0.2470, 0.2435, 0.2616)


# =============================================================================
# SEMILLA
# =============================================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# =============================================================================
# MODELO
# =============================================================================

def build_resnet18_cifar(num_classes=10):

    model = torchvision.models.resnet18(
        weights=None,
        num_classes=num_classes
    )

    # =========================================================================
    # ResNet18 original está diseñada para ImageNet (224x224).
    #
    # CIFAR-10 utiliza imágenes 32x32.
    #
    # Por eso:
    #
    #   Conv original:
    #       7x7, stride 2
    #
    #   Conv adaptada:
    #       3x3, stride 1
    #
    # Además eliminamos el MaxPool inicial.
    # =========================================================================

    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()

    return model


# =============================================================================
# ACCURACY
# =============================================================================

def accuracy_from_logits(outputs, targets):

    _, predicted = outputs.max(1)

    total = targets.size(0)

    correct = predicted.eq(targets).sum().item()

    return correct, total


# =============================================================================
# ENTRENAMIENTO DE UNA ÉPOCA
# =============================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    # =========================================================================
    # CADA GPU PROCESA SU PROPIO SHARD DEL DATASET
    # =========================================================================

    for inputs, targets in loader:

        inputs = inputs.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )


        # ---------------------------------------------------------------------
        # Forward
        # ---------------------------------------------------------------------

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets
        )


        # ---------------------------------------------------------------------
        # Backward
        # ---------------------------------------------------------------------

        loss.backward()

        # =========================================================================
        # IMPORTANTE:
        #
        # DDP sincroniza automáticamente los gradientes entre las GPUs
        # durante el backward().
        #
        # Conceptualmente:
        #
        #       GPU 0              GPU 1
        #         │                  │
        #       grad 0             grad 1
        #         │                  │
        #         └──── ALL-REDUCE ─┘
        #                  │
        #             grad promedio
        #                  │
        #              optimizer
        #
        # =========================================================================


        optimizer.step()


        # ---------------------------------------------------------------------
        # Métricas locales
        # ---------------------------------------------------------------------

        running_loss += (
            loss.item() * inputs.size(0)
        )

        c, t = accuracy_from_logits(
            outputs,
            targets
        )

        correct += c

        total += t


    # =========================================================================
    # COMBINAR MÉTRICAS DE LAS DOS GPUs
    # =========================================================================

    stats = torch.tensor(
        [
            running_loss,
            correct,
            total
        ],
        dtype=torch.float64,
        device=device
    )

    dist.all_reduce(
        stats,
        op=dist.ReduceOp.SUM
    )

    running_loss = stats[0].item()

    correct = stats[1].item()

    total = stats[2].item()


    return (
        running_loss / total,
        100.0 * correct / total
    )


# =============================================================================
# EVALUACIÓN
# =============================================================================
#
# IMPORTANTE:
#
# TODOS LOS RANKS ejecutan esta función.
#
# Esto corrige el error del código anterior.
#
# Si evaluate() utiliza all_reduce(), todos los procesos deben participar.
#
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    for inputs, targets in loader:

        inputs = inputs.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )


        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets
        )


        running_loss += (
            loss.item() * inputs.size(0)
        )

        c, t = accuracy_from_logits(
            outputs,
            targets
        )

        correct += c

        total += t


    # =========================================================================
    # TODOS LOS PROCESOS LLEGAN AQUÍ
    #
    # Por lo tanto el all_reduce NO se queda esperando.
    # =========================================================================

    stats = torch.tensor(
        [
            running_loss,
            correct,
            total
        ],
        dtype=torch.float64,
        device=device
    )

    dist.all_reduce(
        stats,
        op=dist.ReduceOp.SUM
    )

    running_loss = stats[0].item()

    correct = stats[1].item()

    total = stats[2].item()


    return (
        running_loss / total,
        100.0 * correct / total
    )


# =============================================================================
# MAIN
# =============================================================================

def main():


    # =========================================================================
    # 1. INICIALIZAR PROCESO DISTRIBUIDO
    # =========================================================================

    dist.init_process_group(
        backend="nccl"
    )


    # =========================================================================
    # 2. IDENTIFICAR RANK Y GPU
    # =========================================================================

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    rank = dist.get_rank()

    world_size = dist.get_world_size()


    # =========================================================================
    # 3. ASIGNAR GPU
    # =========================================================================

    torch.cuda.set_device(
        local_rank
    )

    device = torch.device(
        f"cuda:{local_rank}"
    )


    # =========================================================================
    # 4. SEMILLA
    # =========================================================================

    set_seed()


    # =========================================================================
    # 5. INFORMACIÓN DEL EXPERIMENTO
    # =========================================================================

    if rank == 0:

        os.makedirs(
            RESULTS_DIR,
            exist_ok=True
        )

        print()
        print("========================================")
        print("ENTRENAMIENTO DISTRIBUIDO DDP")
        print("========================================")

        print(
            f"Mundo distribuido: world_size={world_size}"
        )

        print(
            f"GPUs utilizadas: {world_size}"
        )

        print(
            f"NCCL_P2P_DISABLE = "
            f"{os.environ['NCCL_P2P_DISABLE']}"
        )

        print(
            f"NCCL_IB_DISABLE = "
            f"{os.environ['NCCL_IB_DISABLE']}"
        )


    # =========================================================================
    # 6. BATCH POR GPU
    # =========================================================================

    assert GLOBAL_BATCH_SIZE % world_size == 0

    per_gpu_batch_size = (
        GLOBAL_BATCH_SIZE // world_size
    )


    if rank == 0:

        print(
            f"Batch global: "
            f"{GLOBAL_BATCH_SIZE}"
        )

        print(
            f"Batch por GPU: "
            f"{per_gpu_batch_size}"
        )

        print()


    # =========================================================================
    # 7. TRANSFORMACIONES
    # =========================================================================

    transform_train = transforms.Compose([

        transforms.RandomCrop(
            32,
            padding=4
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )

    ])


    transform_test = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )

    ])


    # =========================================================================
    # 8. DATASET
    # =========================================================================

    train_set = torchvision.datasets.CIFAR10(

        root="./data",

        train=True,

        download=True,

        transform=transform_train

    )


    test_set = torchvision.datasets.CIFAR10(

        root="./data",

        train=False,

        download=True,

        transform=transform_test

    )


    if rank == 0:

        print(
            f"Train: {len(train_set)} imágenes"
        )

        print(
            f"Test: {len(test_set)} imágenes"
        )

        print()


    # =========================================================================
    # 9. DISTRIBUTED SAMPLER
    # =========================================================================
    #
    # El sampler divide el dataset entre los procesos.
    #
    # Con 2 GPUs:
    #
    #       Dataset
    #          │
    #     ┌────┴────┐
    #     ↓         ↓
    #   GPU 0     GPU 1
    #   shard 0   shard 1
    #
    # =========================================================================

    train_sampler = DistributedSampler(

        train_set,

        num_replicas=world_size,

        rank=rank,

        shuffle=True,

        seed=SEED

    )


    # =========================================================================
    # 10. TRAIN DATALOADER
    # =========================================================================

    train_loader = DataLoader(

        train_set,

        batch_size=per_gpu_batch_size,

        sampler=train_sampler,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        drop_last=True

    )


    # =========================================================================
    # 11. TEST DATALOADER
    # =========================================================================
    #
    # Ambos procesos evalúan.
    #
    # Esto es necesario porque evaluate() utiliza all_reduce().
    #
    # =========================================================================

    test_loader = DataLoader(

        test_set,

        batch_size=256,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True

    )


    # =========================================================================
    # 12. MODELO
    # =========================================================================

    model = build_resnet18_cifar(
        num_classes=10
    ).to(device)


    # =========================================================================
    # 13. CONVERTIR MODELO A DDP
    # =========================================================================

    model = DDP(

        model,

        device_ids=[local_rank]

    )


    # =========================================================================
    # 14. LOSS
    # =========================================================================

    criterion = nn.CrossEntropyLoss()


    # =========================================================================
    # 15. OPTIMIZER
    # =========================================================================

    optimizer = optim.SGD(

        model.parameters(),

        lr=LR,

        momentum=0.9,

        weight_decay=5e-4

    )


    # =========================================================================
    # 16. LEARNING RATE SCHEDULER
    # =========================================================================

    scheduler = optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=NUM_EPOCHS

    )


    # =========================================================================
    # 17. HISTORIAL
    # =========================================================================

    history = {

        "epoch": [],

        "train_loss": [],

        "train_acc": [],

        "test_loss": [],

        "test_acc": [],

        "epoch_time_sec": []

    }


    # =========================================================================
    # 18. SINCRONIZAR ANTES DE COMENZAR
    # =========================================================================

    dist.barrier()


    total_start = time.time()


    # =========================================================================
    # 19. LOOP DE ENTRENAMIENTO
    # =========================================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):


        # =====================================================================
        # Cambiar el orden del dataset.
        #
        # Todos los procesos utilizan el mismo epoch.
        # =====================================================================

        train_sampler.set_epoch(
            epoch
        )


        # =====================================================================
        # Sincronización antes de medir
        # =====================================================================

        dist.barrier()


        epoch_start = time.time()


        # =====================================================================
        # ENTRENAMIENTO
        # =====================================================================

        train_loss, train_acc = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device

        )


        # =====================================================================
        # Esperar a que AMBAS GPUs terminen
        # =====================================================================

        dist.barrier()


        epoch_time = (
            time.time() - epoch_start
        )


        # =====================================================================
        # Actualizar learning rate
        # =====================================================================

        scheduler.step()


        # =====================================================================
        # EVALUACIÓN
        #
        # IMPORTANTE:
        #
        # TODOS los ranks ejecutan evaluate().
        #
        # Pero SOLO rank 0 guarda/imprime.
        # =====================================================================

        test_loss, test_acc = evaluate(

            model,

            test_loader,

            criterion,

            device

        )


        # =====================================================================
        # SOLO GPU 0 GUARDA RESULTADOS
        # =====================================================================

        if rank == 0:

            history["epoch"].append(
                epoch
            )

            history["train_loss"].append(
                train_loss
            )

            history["train_acc"].append(
                train_acc
            )

            history["test_loss"].append(
                test_loss
            )

            history["test_acc"].append(
                test_acc
            )

            history["epoch_time_sec"].append(
                epoch_time
            )


            print(

                f"[Época {epoch:02d}/{NUM_EPOCHS}] "

                f"train_loss={train_loss:.4f} "

                f"train_acc={train_acc:.2f}% | "

                f"test_loss={test_loss:.4f} "

                f"test_acc={test_acc:.2f}% | "

                f"tiempo={epoch_time:.1f}s"

            )


    # =========================================================================
    # 20. TIEMPO TOTAL
    # =========================================================================

    dist.barrier()


    total_time = (
        time.time() - total_start
    )


    # =========================================================================
    # 21. GUARDAR RESULTADOS
    # =========================================================================

    if rank == 0:


        print()

        print(
            "========================================"
        )

        print(
            "ENTRENAMIENTO DDP TERMINADO"
        )

        print(
            "========================================"
        )

        print(
            f"Tiempo total: "
            f"{total_time:.1f}s"
        )

        print(
            f"Tiempo total: "
            f"{total_time / 60:.2f} minutos"
        )

        print(
            f"GPUs utilizadas: "
            f"{world_size}"
        )


        # =====================================================================
        # MÉTRICAS
        # =====================================================================

        metrics = {

            "mode": "ddp",

            "num_gpus": world_size,

            "num_epochs": NUM_EPOCHS,

            "global_batch_size":
                GLOBAL_BATCH_SIZE,

            "per_gpu_batch_size":
                per_gpu_batch_size,

            "total_time_sec":
                total_time,

            "avg_epoch_time_sec":
                sum(
                    history["epoch_time_sec"]
                )
                /
                len(
                    history["epoch_time_sec"]
                ),

            "final_train_acc":
                history["train_acc"][-1],

            "final_test_acc":
                history["test_acc"][-1],

            "history":
                history

        }


        # =====================================================================
        # GUARDAR JSON
        # =====================================================================

        with open(

            os.path.join(
                RESULTS_DIR,
                "ddp_metrics.json"
            ),

            "w"

        ) as f:

            json.dump(

                metrics,

                f,

                indent=2

            )


        # =====================================================================
        # GUARDAR MODELO
        # =====================================================================

        torch.save(

            model.module.state_dict(),

            os.path.join(

                RESULTS_DIR,

                "ddp_resnet18_cifar10.pth"

            )

        )


        # =====================================================================
        # GRÁFICAS
        # =====================================================================

        import matplotlib.pyplot as plt


        fig, axes = plt.subplots(

            1,

            2,

            figsize=(12, 4)

        )


        # ---------------------------------------------------------------------
        # LOSS
        # ---------------------------------------------------------------------

        axes[0].plot(

            history["epoch"],

            history["train_loss"],

            label="Train loss"

        )


        axes[0].plot(

            history["epoch"],

            history["test_loss"],

            label="Test loss"

        )


        axes[0].set_xlabel(
            "Época"
        )


        axes[0].set_ylabel(
            "Loss"
        )


        axes[0].set_title(
            "Loss por época (DDP)"
        )


        axes[0].legend()


        # ---------------------------------------------------------------------
        # ACCURACY
        # ---------------------------------------------------------------------

        axes[1].plot(

            history["epoch"],

            history["train_acc"],

            label="Train accuracy"

        )


        axes[1].plot(

            history["epoch"],

            history["test_acc"],

            label="Test accuracy"

        )


        axes[1].set_xlabel(
            "Época"
        )


        axes[1].set_ylabel(
            "Accuracy (%)"
        )


        axes[1].set_title(
            "Accuracy por época (DDP)"
        )


        axes[1].legend()


        plt.tight_layout()


        plt.savefig(

            os.path.join(

                RESULTS_DIR,

                "ddp_curves.png"

            ),

            dpi=150

        )


        plt.close()


        # =====================================================================
        # RESUMEN
        # =====================================================================

        print()

        print(
            "========================================"
        )

        print(
            "RESULTADOS DDP"
        )

        print(
            "========================================"
        )

        print(
            f"Tiempo total: "
            f"{metrics['total_time_sec']:.1f}s"
        )

        print(
            f"Tiempo promedio/época: "
            f"{metrics['avg_epoch_time_sec']:.1f}s"
        )

        print(
            f"Train accuracy final: "
            f"{metrics['final_train_acc']:.2f}%"
        )

        print(
            f"Test accuracy final: "
            f"{metrics['final_test_acc']:.2f}%"
        )

        print(
            f"GPUs: "
            f"{metrics['num_gpus']}"
        )

        print(
            f"Batch global: "
            f"{metrics['global_batch_size']}"
        )

        print(
            f"Batch por GPU: "
            f"{metrics['per_gpu_batch_size']}"
        )

        print(
            "========================================"
        )


    # =========================================================================
    # 22. FINALIZAR PROCESS GROUP
    # =========================================================================

    dist.destroy_process_group()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
