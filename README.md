# MORSE-PIC: Context-Aware Visual Code Generation via Evolutionary Prompt Optimization

This repository implements **MORSE-PIC**, a framework designed to enable **self-improvement for smaller Vision-Language Models (VLMs)**. By optimizing the code generation process programmatically, the model learns to generate Python scripts that accurately reconstruct visual inputs, effectively "training" itself on its own generated data verification loop.

Unlike standard inference-time correction methods, this approach utilizes **Training-Time Optimization (TTO)** via **Generative Evolutionary Prompt Optimization (GEPA)** to refine system prompts and few-shot examples based on a visual similarity metric.

## Algorithm

The framework formulates the code generation task as a context-conditional optimization problem.

### 1. Context Partitioning
Let $\mathcal{D} = \{(x_i, I_i, c_i)\}_{i=1}^N$ be the dataset, where $x_i$ is the reference image, $I_i$ is the instruction, and $c_i$ is the semantic context (e.g., "geometry"). We partition $\mathcal{D}$ into context-specific subsets:
$$
\mathcal{D}_k = \{ (x, I) \mid c = k \}
$$

### 2. Code Generation & Execution
For a given context $k$, we optimize a system prompt $\phi_k$. The policy $\pi_\theta$ (the VLM) generates code $z$ conditioned on the image and instruction:
$$
z \sim \pi_\theta(z \mid x, I; \phi_k)
$$
The code is executed in a sandbox to produce a candidate image $\hat{x}$:
$$
\hat{x} = \text{Exec}(z)
$$

### 3. Visual Objective
We define a visual fidelity score $S$ using a pre-trained visual encoder $E$ (DINOv2):
$$
S(x, \hat{x}) = \frac{E(x) \cdot E(\hat{x})}{\|E(x)\| \|E(\hat{x})\|}
$$

### 4. Optimization (GEPA)
The objective is to find the optimal prompt $\phi_k^*$ for each context that maximizes the expected visual similarity score:
$$
\phi_k^* = \operatorname*{argmax}_{\phi} \mathbb{E}_{(x, I) \sim \mathcal{D}_k} \left[ S(x, \text{Exec}(\pi_\theta(\cdot \mid x, I; \phi))) \right]
$$
We solve this using **GEPA** (Generative Evolutionary Prompt Optimization), which iteratively evolves the population of prompts $\{\phi^{(t)}\}$ via mutation and crossover operations guided by the fitness function $S$.

## Methodology

The core of our approach shifts the computational burden from inference-time iteration to offline prompt optimization.

### 1. Visual Similarity Metric (DINOv2)
We employ `facebook/dinov3-vits16-pretrain-lvd1689m` as a perceptual metric. Generated scripts are executed in a sandboxed environment to produce an image, which is then embedded and compared to the reference image using cosine similarity. This continuous score serves as the fitness function for the evolutionary optimizer.

### 2. Context-Specific Optimization
To handle diverse visual domains (e.g., geometry, function plots, fractals), we partition the training dataset into distinct contexts ($C_1, C_2, ..., C_n$). We independently optimize a VLM system prompt $\phi_i$ for each context $C_i$ using GEPA. 

### 3. Evolutionary Optimization (GEPA)
We use the **DSPy** library's implementation of GEPA to iteratively evolve instructions and few-shot examples. The optimization objective is to maximize the expected DINOv2 similarity score over the training set for a given context.

## Repository Structure

```
.
├── src/
│   ├── code.py           # VLMTask signature and ChainOfThought module
│   └── metrics.py        # DINOv2 visual similarity metric & execution sandbox
├── train_gepa.py         # Main optimization script (GEPA loop)
├── run_inference.py      # Inference script with context-aware module loading
└── data/                 # Dataset storage (Parquet format)
```

## Setup and Usage

### Prerequisites
The code is built on **DSPy** and requires a GPU-enabled environment for the DINOv2 metric (though it supports CPU execution).

```bash
pip install dspy-ai gepa transformers torch pillow pandas pyarrow
```

### Data Format
The framework expects a Parquet file (`data/testmini.parquet`) containing:
- `image_path`: Absolute path to the reference image.
- `metadata`: JSON string containing a `context` key (e.g., `{"context": "geometry"}`) used for clustering.

### Optimization
To run the evolutionary optimization process for all discovered contexts:

```bash
python train_gepa.py
```
This process yields a set of optimized programs `optimized_vlm_{hash}.json`, effectively "training" the agent for distinct visual tasks.

### Inference & Evaluation
To evaluate the optimized models:

```bash
python run_inference.py
```
The script dynamically loads the appropriate optimized prompt for each test instance based on its context.
