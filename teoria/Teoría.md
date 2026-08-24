
````
# Paralela Project — Entrenamiento Distribuido con PyTorch DDP

## 1. Descripción

Este proyecto implementa y evalúa el **entrenamiento paralelo de una red neuronal convolucional** utilizando **Data Parallelism** mediante **PyTorch DistributedDataParallel (DDP)**.

Se utiliza el modelo **ResNet18** sobre el dataset **CIFAR-10**, comparando:

- Entrenamiento secuencial con 1 GPU.
- Entrenamiento distribuido con 2 GPUs.

El objetivo principal es determinar cuánto se acelera el entrenamiento al utilizar dos GPUs y analizar el costo asociado a la comunicación y sincronización entre procesos.

---

# 2. Problema

El entrenamiento de redes neuronales profundas requiere una gran cantidad de operaciones computacionales.

Cuando el dataset o el modelo aumenta de tamaño, el tiempo de entrenamiento en una sola GPU puede convertirse en un cuello de botella.

La idea del paralelismo es dividir el trabajo entre varios procesadores o GPUs:

```text
                    Trabajo total
                         │
              ┌──────────┴──────────┐
              │                     │
            GPU 0                 GPU 1
              │                     │
           Trabajo 1              Trabajo 2
              │                     │
              └──────────┬──────────┘
                         │
                   Sincronización
                         │
                   Resultado final
````

Sin embargo:

> **Utilizar el doble de GPUs no significa obtener automáticamente el doble de velocidad.**

Existe un costo adicional producido por la comunicación, sincronización y coordinación entre los procesos.

---

# 3. Objetivos

### Objetivo general

Evaluar experimentalmente el beneficio del entrenamiento distribuido mediante Data Parallelism utilizando PyTorch DDP.

### Objetivos específicos

1. Implementar un baseline secuencial con una GPU.
2. Implementar entrenamiento distribuido con dos GPUs.
3. Mantener una configuración experimental comparable.
4. Medir los tiempos de ejecución.
5. Comparar la calidad de los modelos mediante accuracy.
6. Calcular:

   * Speedup.
   * Eficiencia paralela.
   * Reducción del tiempo.
   * Overhead.
7. Analizar las diferencias respecto al speedup ideal.
8. Relacionar los resultados con la Ley de Amdahl y el costo de comunicación.

---

# 4. Conceptos fundamentales

## 4.1 Concurrencia

La concurrencia consiste en gestionar múltiples tareas que pueden avanzar de manera intercalada.

No necesariamente significa que las tareas se ejecuten físicamente al mismo tiempo.

```text
Tarea A ────┐
            ├──> ejecución coordinada
Tarea B ────┘
```

---

## 4.2 Paralelismo

El paralelismo ocurre cuando varias unidades de procesamiento ejecutan partes del trabajo simultáneamente.

En este proyecto:

```text
GPU 0 ──> procesa una parte del batch
GPU 1 ──> procesa otra parte del batch
```

Por lo tanto:

> **Paralelismo = dividir trabajo para ejecutarlo simultáneamente.**

---

# 5. Data Parallelism

El proyecto utiliza **Data Parallelism**.

La idea consiste en mantener una copia completa del modelo en cada GPU y dividir los datos entre ellas.

Para un batch global de 128:

```text
Batch global = 128

             ┌───────────────┐
             │  Batch = 128  │
             └───────┬───────┘
                     │
             DistributedSampler
                     │
          ┌──────────┴──────────┐
          │                     │
       GPU 0                 GPU 1
       Batch 64              Batch 64
          │                     │
       ResNet18              ResNet18
          │                     │
       Gradientes            Gradientes
          │                     │
          └─────────┬───────────┘
                    │
                 ALLREDUCE
                    │
            Gradientes sincronizados
                    │
                    ▼
              optimizer.step()
```

En consecuencia:

[
128 = 64 + 64
]

Cada GPU procesa una parte diferente del conjunto de datos.

---

# 6. PyTorch DistributedDataParallel

**DistributedDataParallel (DDP)** es la herramienta de PyTorch utilizada para implementar el entrenamiento distribuido.

En el proyecto:

```text
Proceso 0 ──> GPU 0 ──> ResNet18
Proceso 1 ──> GPU 1 ──> ResNet18
```

Cada proceso controla una GPU.

DDP se encarga de sincronizar los gradientes entre las réplicas del modelo.

---

# 7. DistributedSampler

El `DistributedSampler` divide el dataset entre los procesos.

En lugar de que ambas GPUs procesen exactamente los mismos datos:

```text
Dataset
   │
   ▼
DistributedSampler
   │
   ├──> GPU 0: subconjunto A
   │
   └──> GPU 1: subconjunto B
```

Esto evita trabajo duplicado y permite que las GPUs trabajen en paralelo.

Además:

```python
train_sampler.set_epoch(epoch)
```

permite modificar el orden de los datos de forma controlada en cada época.

---

# 8. Forward y Backward

Cada GPU ejecuta de manera independiente:

### Forward

```text
Datos
  ↓
ResNet18
  ↓
Predicción
  ↓
Loss
```

### Backward

```text
Loss
 ↓
Backpropagation
 ↓
Gradientes
```

Después del `backward()` ocurre la sincronización de gradientes mediante DDP.

---

# 9. ALLREDUCE

La operación colectiva fundamental del proyecto es **ALLREDUCE**.

Su función es combinar los gradientes calculados por las diferentes GPUs.

Conceptualmente:

```text
GPU 0 → gradiente G0 ──┐
                       │
                       ▼
                    ALLREDUCE
                       │
                       ▼
GPU 1 → gradiente G1 ──┘

Resultado:
gradiente sincronizado
```

De manera simplificada:

[
G = \frac{G_0 + G_1}{2}
]

Así, ambas réplicas utilizan gradientes consistentes para actualizar sus parámetros.

---

# 10. Sincronización

El entrenamiento distribuido requiere coordinación entre procesos.

El proyecto utiliza:

```python
dist.barrier()
```

`barrier()` obliga a que los procesos esperen hasta que todos hayan alcanzado ese punto.

Conceptualmente:

```text
GPU 0 ────────────────┐
                      │
                      ▼
                    BARRIER
                      ▲
                      │
GPU 1 ────────────────┘
```

Esto permite sincronizar determinadas etapas de la medición y ejecución.

Sin embargo, esta sincronización también introduce overhead.

---

# 11. NCCL

El backend utilizado es:

```python
dist.init_process_group(
    backend="nccl"
)
```

**NCCL (NVIDIA Collective Communications Library)** está diseñado para realizar comunicaciones colectivas eficientes entre GPUs NVIDIA.

Entre las operaciones soportadas se encuentran:

* ALLREDUCE
* BROADCAST
* ALLGATHER
* REDUCE
* BARRIER

En el entorno Kaggle utilizado fue necesario deshabilitar:

```python
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
```

Esto permitió evitar problemas de comunicación GPU-GPU propios del entorno virtualizado y completar correctamente el entrenamiento distribuido.

---

# 12. Modelo utilizado

Se utiliza **ResNet18**.

ResNet18 fue originalmente diseñada para datasets con imágenes mayores, como ImageNet.

CIFAR-10 utiliza imágenes de:

[
32 \times 32
]

Por ello se modifica la arquitectura:

```python
model.conv1 = nn.Conv2d(
    3, 64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

model.maxpool = nn.Identity()
```

Esto evita reducir demasiado pronto la resolución espacial de las imágenes.

---

# 13. Dataset

Se utiliza **CIFAR-10**.

Características:

* 50 000 imágenes de entrenamiento.
* 10 000 imágenes de prueba.
* 10 clases.
* Imágenes RGB.
* Resolución de:

[
32 \times 32
]

Las clases incluyen:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

---

# 14. Configuración experimental

Los dos experimentos utilizan una configuración equivalente:

| Parámetro     |       Secuencial |              DDP |
| ------------- | ---------------: | ---------------: |
| Dataset       |         CIFAR-10 |         CIFAR-10 |
| Modelo        |         ResNet18 |         ResNet18 |
| GPUs          |                1 |                2 |
| Épocas        |               15 |               15 |
| Batch global  |              128 |              128 |
| Batch/GPU     |              128 |               64 |
| Optimizer     |              SGD |              SGD |
| Learning rate |              0.1 |              0.1 |
| Momentum      |              0.9 |              0.9 |
| Weight decay  |             5e-4 |             5e-4 |
| Scheduler     | Cosine Annealing | Cosine Annealing |

La condición fundamental para comparar rendimiento es mantener constante el **batch global**:

[
B_{global}=128
]

En DDP:

[
128/2=64
]

por GPU.

---

# 15. Métricas de rendimiento

Para evaluar el paralelismo se utilizan cuatro métricas principales.

## 15.1 Tiempo de ejecución

Es el tiempo necesario para completar el entrenamiento.

Se define:

[
T_1 = \text{tiempo secuencial}
]

[
T_2 = \text{tiempo utilizando 2 GPUs}
]

---

# 16. Speedup

El speedup mide cuánto más rápido es la versión paralela respecto a la secuencial.

[
S = \frac{T_1}{T_2}
]

En el experimento:

[
S =
\frac{667.22}{389.63}
]

[
\boxed{S \approx 1.71\times}
]

Esto significa que el entrenamiento distribuido fue aproximadamente **1.71 veces más rápido**.

---

# 17. Speedup ideal

Si todo el programa fuera perfectamente paralelizable y no existiera ningún costo de comunicación:

[
S_{ideal}=N
]

Para dos GPUs:

[
S_{ideal}=2
]

Pero experimentalmente:

[
S_{real}=1.71
]

Por lo tanto:

[
1.71 < 2
]

Esto demuestra que existe overhead.

---

# 18. Eficiencia paralela

La eficiencia mide qué tan cerca estamos del speedup ideal.

[
E=\frac{S}{N}
]

Para dos GPUs:

[
E=\frac{1.71}{2}
]

[
\boxed{E\approx85.62%}
]

Esto significa que se aprovechó aproximadamente el **85.62 %** de la capacidad paralela ideal.

---

# 19. Reducción del tiempo

La reducción porcentual del tiempo se calcula como:

[
R=
\frac{T_1-T_2}{T_1}\times100
]

En este proyecto:

[
R=
\frac{667.22-389.63}{667.22}
\times100
]

[
\boxed{R\approx41.60%}
]

El entrenamiento distribuido redujo el tiempo total aproximadamente un **41.60 %**.

---

# 20. Overhead

Con dos GPUs, el tiempo ideal sería:

[
T_{ideal}=\frac{T_1}{2}
]

[
T_{ideal}=
\frac{667.22}{2}
================

333.61s
]

Sin embargo, el tiempo real fue:

[
T_2=389.63s
]

Por tanto:

[
Overhead=T_2-T_{ideal}
]

[
Overhead=389.63-333.61
]

[
\boxed{Overhead\approx56.02s}
]

Estos aproximadamente 56 segundos representan el tiempo adicional respecto al escenario ideal.

---

# 21. ¿De dónde viene el overhead?

El overhead puede producirse por diferentes factores:

### Comunicación

Las GPUs deben intercambiar información durante las operaciones colectivas.

### ALLREDUCE

Los gradientes deben sincronizarse después del cálculo del backward.

### Sincronización

Las operaciones `barrier()` obligan a coordinar los procesos.

### DistributedSampler

Existe trabajo adicional para distribuir correctamente los datos.

### DataLoader

Cada proceso tiene su propio proceso de carga de datos.

### Runtime distribuido

DDP requiere procesos adicionales, coordinación y administración de recursos.

Por tanto:

[
T_{paralelo}
============

T_{trabajo}
+
T_{comunicación}
+
T_{sincronización}
+
T_{overhead}
]

---

# 22. Ley de Amdahl

La **Ley de Amdahl** explica por qué aumentar el número de procesadores no produce un speedup ilimitado.

Si una fracción (P) del programa puede paralelizarse y una fracción (1-P) permanece secuencial:

[
S(N)=
\frac{1}
{(1-P)+\frac{P}{N}}
]

Incluso con un número muy grande de procesadores, la parte secuencial limita el speedup.

En entrenamiento distribuido, además de las partes seriales, aparecen costos adicionales de comunicación y sincronización.

Por eso:

[
S_{real}<S_{ideal}
]

---

# 23. Resultados experimentales

## Baseline

```text
GPUs:              1
Tiempo:            667.22 s
Tiempo/época:      44.5 s
Train Accuracy:    94.40 %
Test Accuracy:     90.54 %
```

## DDP

```text
GPUs:              2
Tiempo:            389.63 s
Tiempo/época:      23.1 s
Train Accuracy:    95.32 %
Test Accuracy:     90.99 %
```

---

# 24. Comparación final

| Métrica        | Secuencial | DDP 2 GPUs |
| -------------- | ---------: | ---------: |
| GPUs           |          1 |          2 |
| Tiempo         |   667.22 s |   389.63 s |
| Tiempo/época   |     44.5 s |     23.1 s |
| Train Accuracy |    94.40 % |    95.32 % |
| Test Accuracy  |    90.54 % |    90.99 % |

Métricas paralelas:

| Métrica              |   Resultado |
| -------------------- | ----------: |
| Speedup              |   **1.71×** |
| Speedup ideal        |      **2×** |
| Eficiencia           | **85.62 %** |
| Reducción del tiempo | **41.60 %** |
| Overhead             | **56.02 s** |

---

# 25. Calidad del modelo

El objetivo no es únicamente reducir el tiempo.

También debe verificarse que la paralelización no degrade el modelo.

Comparación:

[
Accuracy_{baseline}=90.54%
]

[
Accuracy_{DDP}=90.99%
]

Diferencia:

[
90.99-90.54=0.45
]

Por tanto:

[
\boxed{\Delta Accuracy=+0.45\ puntos\ porcentuales}
]

El modelo distribuido obtuvo una accuracy ligeramente superior.

Esto demuestra que el uso de DDP **no produjo una degradación observable de la calidad predictiva** bajo esta configuración experimental.

---

# 26. Convergencia

Durante las 15 épocas, el modelo distribuido mostró una reducción progresiva de la loss y un incremento de la accuracy.

```text
Época       Test Accuracy

1              40.57 %
5              65.65 %
10             84.79 %
15             90.99 %
```

Además:

```text
Train Loss:
1.8886 → 0.1425

Train Accuracy:
31.63 % → 95.32 %
```

Esto indica que el proceso de entrenamiento convergió correctamente.

---

# 27. Interpretación de los resultados

El experimento demuestra que el paralelismo fue efectivo:

```text
1 GPU
667.22 s
   │
   │ DDP
   ▼
2 GPUs
389.63 s
```

El tiempo disminuyó aproximadamente un:

[
41.60%
]

y se obtuvo:

[
1.71\times
]

de speedup.

Sin embargo, no se alcanzó el speedup ideal de:

[
2\times
]

debido principalmente a los costos de comunicación y sincronización inherentes al entrenamiento distribuido.

---

# 28. Conclusiones

1. Se implementó exitosamente **Data Parallelism** mediante PyTorch DDP.

2. Se utilizaron dos GPUs NVIDIA T4 para procesar CIFAR-10 de manera distribuida.

3. El tiempo de entrenamiento disminuyó de:

[
667.22s \rightarrow 389.63s
]

4. Se obtuvo un speedup de:

[
\boxed{1.71\times}
]

5. La eficiencia paralela fue:

[
\boxed{85.62%}
]

6. El tiempo de entrenamiento se redujo:

[
\boxed{41.60%}
]

7. El overhead respecto al tiempo ideal fue aproximadamente:

[
\boxed{56.02s}
]

8. La accuracy pasó de:

[
90.54%\rightarrow90.99%
]

por lo que la paralelización no produjo una pérdida de calidad.

9. El experimento confirma que:

> **El paralelismo real no equivale a duplicar automáticamente la velocidad.**

La diferencia entre el speedup ideal y el real se debe a los costos de comunicación, sincronización y coordinación.

---

# 29. Conceptos demostrados

Este proyecto permite demostrar experimentalmente los siguientes conceptos de Computación Paralela:

```text
Data Parallelism
       │
       ▼
DistributedDataParallel
       │
       ├── DistributedSampler
       │
       ├── Múltiples procesos
       │
       ├── Múltiples GPUs
       │
       ├── Forward / Backward
       │
       ├── ALLREDUCE
       │
       └── Sincronización
              │
              ▼
       Métricas de rendimiento
              │
       ├── Speedup
       ├── Eficiencia
       ├── Overhead
       └── Reducción del tiempo
              │
              ▼
       Ley de Amdahl
```

---

# 30. Resumen ejecutivo

```text
PROYECTO
ResNet18 + CIFAR-10 + PyTorch DDP

BASELINE
1 GPU
T1 = 667.22 s
Accuracy = 90.54 %

DDP
2 GPUs
T2 = 389.63 s
Accuracy = 90.99 %

RESULTADOS
Speedup       = 1.71×
Eficiencia    = 85.62 %
Reducción     = 41.60 %
Overhead      = 56.02 s

SPEEDUP IDEAL = 2×

CONCLUSIÓN

DDP redujo significativamente el tiempo de entrenamiento
manteniendo la calidad del modelo.

El speedup fue sublineal debido al costo de comunicación,
sincronización y coordinación entre las GPUs.
```

---

# 31. Tecnologías utilizadas

* Python
* PyTorch
* TorchVision
* CUDA
* NCCL
* CIFAR-10
* ResNet18
* PyTorch DistributedDataParallel
* NVIDIA T4
* Kaggle

---

# 32. Estructura conceptual del proyecto

```text
Paralela_project/
│
├── train_baseline.py
│       └── Entrenamiento secuencial
│
├── train_ddp.py
│       └── Entrenamiento distribuido
│
├── baseline_metrics.json
│       └── Resultados baseline
│
├── results_ddp/
│       ├── ddp_metrics.json
│       ├── ddp_resnet18_cifar10.pth
│       └── ddp_curves.png
│
├── comparacion_tiempo.png
├── comparacion_accuracy.png
├── speedup_comparacion.png
├── eficiencia_reduccion.png
└── tiempo_ideal_vs_real.png
```

---

## Resultado final

> **El entrenamiento distribuido mediante PyTorch DDP sobre 2 GPUs redujo el tiempo de entrenamiento de 667.22 s a 389.63 s, obteniendo un speedup de 1.71× y una eficiencia paralela de 85.62 %, mientras mantuvo prácticamente intacta la calidad del modelo (90.54 % → 90.99 % de accuracy). El experimento evidencia que el speedup real es sublineal debido al overhead de comunicación y sincronización entre las GPUs.**

```
```
