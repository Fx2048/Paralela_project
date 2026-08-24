
# 33. Limitaciones y trabajo futuro

## Limitaciones

El experimento fue realizado utilizando dos GPUs NVIDIA T4
dentro del entorno virtualizado de Kaggle.

Debido a las características del entorno, fue necesario
deshabilitar la comunicación P2P e InfiniBand de NCCL:

NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1

Por lo tanto, los resultados obtenidos representan el
rendimiento de DDP bajo esta infraestructura específica
y no necesariamente el rendimiento máximo que podría
obtenerse en un sistema con comunicación GPU-GPU dedicada.

Además, el experimento utiliza solamente 15 épocas y dos GPUs.
Por ello, no se pretende generalizar el resultado a cualquier
cantidad de GPUs, modelos o datasets.

## Trabajo futuro

Como extensión del proyecto se podría:

1. Comparar 1, 2, 4 y más GPUs.
2. Evaluar diferentes tamaños de batch.
3. Comparar diferentes arquitecturas de redes neuronales.
4. Analizar el impacto del tamaño del modelo sobre el overhead.
5. Medir el comportamiento con diferentes cantidades de épocas.
6. Comparar DDP con otras estrategias de paralelización.
7. Evaluar el rendimiento utilizando comunicación GPU-GPU
   mediante NVLink u otra infraestructura dedicada.
