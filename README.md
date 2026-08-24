# Paralela_project

**Entrenamiento distribuido de redes neuronales mediante Data Parallelism y PyTorch DDP**

---

## 1. Problema

El entrenamiento de redes neuronales profundas es un proceso computacionalmente intensivo. A medida que los modelos y los datasets crecen, el tiempo requerido para completar el entrenamiento en una sola GPU se vuelve prohibitivo.

El desafío principal consiste en **acelerar el proceso de entrenamiento** sin degradar la calidad del modelo (accuracy), aprovechando múltiples GPUs de forma eficiente. Sin embargo, la paralelización introduce costos de comunicación y sincronización que impiden alcanzar un speedup ideal.

Este proyecto aborda el problema mediante la implementación de **Data Parallelism** utilizando **PyTorch DistributedDataParallel (DDP)** sobre dos GPUs, midiendo de forma rigurosa el speedup, la eficiencia paralela y el overhead asociado.

---

## 2. Objetivos

1. Implementar un entrenamiento **baseline** (secuencial) de ResNet18 sobre CIFAR-10 utilizando una sola GPU.
2. Implementar el mismo entrenamiento de forma **distribuida** mediante PyTorch DDP con 2 GPUs.
3. Comparar tiempos de ejecución, accuracy de entrenamiento y accuracy de prueba entre ambas versiones.
4. Calcular **speedup**, **eficiencia paralela** y **overhead** de comunicación/sincronización.
5. Demostrar que la paralelización reduce significativamente el tiempo de entrenamiento **sin sacrificar** la calidad del modelo.
6. Relacionar los resultados experimentales con conceptos teóricos de computación paralela (Ley de Amdahl, overhead de comunicación, operaciones colectivas).

---

## 3. Marco teórico

### Data Parallelism

En Data Parallelism, el dataset se divide entre múltiples dispositivos. Cada GPU mantiene una **réplica completa del modelo** y procesa un subconjunto diferente del batch. Después del cálculo de gradientes se realiza una sincronización (normalmente mediante **ALLREDUCE**) para que todas las réplicas actualicen sus pesos de forma consistente.

### PyTorch DistributedDataParallel (DDP)

DDP es la API oficial de PyTorch para entrenamiento distribuido. Características principales:

- Cada proceso controla una GPU.
- Utiliza `DistributedSampler` para particionar el dataset sin solapamiento.
- Sincroniza gradientes automáticamente después de `loss.backward()` mediante operaciones colectivas (ALLREDUCE).
- Garantiza que todos los procesos ejecuten `optimizer.step()` con gradientes idénticos.

### Métricas de rendimiento paralelo

- **Speedup**: \( S = \dfrac{T_1}{T_N} \)
- **Eficiencia**: \( E = \dfrac{S}{N} \)
- **Overhead**: diferencia entre el tiempo ideal (\(T_1 / N\)) y el tiempo real observado.

Estas métricas permiten cuantificar hasta qué punto la paralelización es efectiva y cuánto cuesta la coordinación entre procesos.

---

## 4. Metodología

Se realizaron dos experimentos bajo **configuración idéntica**:

| Parámetro              | Valor                  |
|------------------------|------------------------|
| Dataset                | CIFAR-10               |
| Modelo                 | ResNet18               |
| Épocas                 | 15                     |
| Batch size global      | 128                    |
| Optimizador            | SGD (configuración estándar) |
| Framework              | PyTorch + DDP          |
| Hardware               | GPUs NVIDIA T4         |

- **Baseline**: 1 GPU, batch size = 128.
- **DDP**: 2 GPUs, batch size por GPU = 64 (batch global = 128).

Se midieron:

- Tiempo total de entrenamiento.
- Tiempo promedio por época.
- Accuracy de entrenamiento y de prueba.
- Evolución de la loss y de la accuracy a lo largo de las épocas.

---

## 5. Arquitectura

El esquema de paralelismo implementado es el siguiente:

```text
                 Dataset CIFAR-10
                       │
                       ▼
                DistributedSampler
                 /               \
                /                 \
           GPU 0                 GPU 1
          Batch 64              Batch 64
             │                     │
             ▼                     ▼
        ResNet18               ResNet18
             │                     │
             ▼                     ▼
         Gradientes             Gradientes
             \                   /
              \                 /
               ──── ALLREDUCE ────
                       │
                       ▼
              Gradientes sincronizados
                       │
                       ▼
                Actualización de pesos
                (optimizer.step())
```

Cada GPU procesa una parte diferente del batch global:

\[
128 = 64 + 64
\]

Ambas GPUs mantienen una copia idéntica del modelo. Tras el `backward()`, DDP sincroniza los gradientes y posteriormente se actualizan los pesos de forma consistente en todos los procesos.

---

## 6. Experimento baseline

- **GPUs**: 1
- **Batch size**: 128
- **Épocas**: 15
- **Tiempo total**: **667.2 s**
- **Tiempo promedio por época**: **44.5 s**
- **Train Accuracy**: **94.40 %**
- **Test Accuracy**: **90.54 %**

Este experimento sirve como referencia (\(T_1\)) para el cálculo de speedup y eficiencia.

---

## 7. Experimento distribuido

- **GPUs**: 2 (DDP)
- **Batch size global**: 128 (64 por GPU)
- **Épocas**: 15
- **Tiempo total**: **389.6 s**
- **Tiempo promedio por época**: **23.1 s**
- **Train Accuracy**: **95.32 %**
- **Test Accuracy**: **90.99 %**

### Evolución del modelo distribuido

| Época | Test Accuracy |
|-------|---------------|
| 1     | 40.57 %       |
| 5     | 65.65 %       |
| 10    | 84.79 %       |
| 15    | 90.99 %       |

- `train_loss`: \(1.8886 \rightarrow 0.1425\)
- `train_accuracy`: \(31.63\% \rightarrow 95.32\%\)

El modelo convergió de forma correcta y estable.

---

## 8. Resultados

| Métrica          | Secuencial | DDP 2 GPUs |
|------------------|------------|------------|
| GPUs             | 1          | 2          |
| Épocas           | 15         | 15         |
| Batch global     | 128        | 128        |
| Batch/GPU        | 128        | 64         |
| Tiempo total     | **667.2 s**| **389.6 s**|
| Tiempo/época     | **44.5 s** | **23.1 s** |
| Train Accuracy   | **94.40 %**| **95.32 %**|
| Test Accuracy    | **90.54 %**| **90.99 %**|

**Observaciones clave:**

- El tiempo de entrenamiento se redujo de 667.2 s a 389.6 s.
- La precisión de prueba se mantuvo comparable (incluso ligeramente superior en DDP).
- La paralelización **no deterioró** la calidad del modelo.

---

## 9. Speedup

\[
\text{Speedup} = \frac{T_1}{T_2} = \frac{667.2}{389.6} \approx \mathbf{1.71\times}
\]

Esto equivale a una **reducción del tiempo de entrenamiento del 41.6 %**.

El tiempo promedio por época pasó de **44.5 s** a **23.1 s**.

### Comparación con el speedup ideal

- Speedup ideal con 2 GPUs: \(2\times\)
- Speedup real obtenido: \(1.71\times\)
- Diferencia: \(2 - 1.71 = 0.29\)

No se alcanzó el paralelismo perfecto debido al overhead de comunicación y sincronización.

---

## 10. Eficiencia

\[
E = \frac{\text{Speedup}}{N} = \frac{1.71}{2} \approx \mathbf{85.5\%}
\]

Una eficiencia del **85.5 %** con 2 GPUs es un resultado sólido, indicando que la mayor parte del trabajo útil se aprovecha y que el overhead, aunque presente, se mantiene dentro de límites razonables.

---

## 11. Overhead

El tiempo ideal teórico con 2 GPUs sería:

\[
\frac{667.2}{2} = 333.6\ \text{s}
\]

El tiempo real observado fue **389.6 s**. Por tanto, el overhead es:

\[
389.6 - 333.6 = \mathbf{56.0\ s}
\]

Este overhead se explica por:

- Operaciones colectivas **ALLREDUCE** para sincronizar gradientes.
- Comunicación entre procesos.
- Sincronización (`barrier`) entre GPUs.
- Overhead de los `DistributedSampler` y DataLoaders.
- Coordinación general del runtime de DDP.

**Conclusión teórica**:  
\[
\boxed{\text{Paralelismo} \neq \text{velocidad ideal}}
\]

Duplicar el número de GPUs **no** duplica exactamente la velocidad. Existe un costo inherente a la coordinación y comunicación, coherente con la **Ley de Amdahl** y los costos de las operaciones colectivas.

---

## 12. Conclusiones

1. **Se implementó exitosamente Data Parallelism** mediante PyTorch DistributedDataParallel utilizando dos GPUs T4.

2. El tiempo de entrenamiento se redujo de **667.2 s** a **389.6 s**, obteniendo un **speedup de 1.71×** y una **eficiencia paralela del 85.5 %**.

3. La precisión del modelo se mantuvo comparable:
   - Baseline: **90.54 %**
   - DDP: **90.99 %**  
   (diferencia de +0.45 puntos porcentuales, sin deterioro del rendimiento predictivo).

4. La paralelización **no sacrificó** la calidad del aprendizaje; el modelo convergió de forma correcta en ambas configuraciones.

5. El overhead de aproximadamente **56 s** ilustra de manera concreta que el paralelismo real siempre incluye costos de comunicación y sincronización.

6. Los resultados demuestran de forma experimental los conceptos estudiados en Computación Paralela:  
   speedup sublineal, eficiencia < 100 %, overhead de comunicación y la imposibilidad de alcanzar el speedup ideal en presencia de secciones seriales y costos de coordinación.

### Números finales de referencia

```text
BASELINE
T1          = 667.2 s
Accuracy    = 90.54 %

DDP (2 GPUs)
T2          = 389.6 s
Accuracy    = 90.99 %

SPEEDUP     = 1.71×
EFICIENCIA  = 85.5 %
REDUCCIÓN   = 41.6 %
OVERHEAD    ≈ 56.0 s
```

**El experimento demuestra de forma cuantitativa que el entrenamiento distribuido con DDP es efectivo, eficiente y no degrada la calidad del modelo.**

---

*Proyecto de Computación Paralela – Entrenamiento distribuido con PyTorch DDP*

