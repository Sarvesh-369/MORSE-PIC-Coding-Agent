# MORSE-PIC: Context-Aware Visual Code Generation via Evolutionary Prompt Optimization

This repository contains the implementation for **MORSE-PIC**, a framework for optimizing Vision-Language Models (VLMs) to generate executable Python code that visually reconstructs input images. Unlike standard inference-time correction methods (e.g., self-correction loops), this approach utilizes **Training-Time Optimization (TTO)** via **Generative Evolutionary Prompt Optimization (GEPA)** to refine system prompts and few-shot examples based on a visual similarity metric.

##  Methodology

The core of our approach is to shift the computational burden from inference-time iteration to offline prompt optimization. The framework consists of three main components:

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

## License
MIT
