# MORSE-PIC Coding Agent

A Vision-Language Model (VLM) agent that generates Python code to visually recreate input images. This repository implements **Training-Time Optimization** using **DSPy** and **GEPA** (Generative Evolutionary Prompt Optimization), guided by a **DINOv2** visual similarity metric.

## 🚀 Features

- **Visual Fidelity Optimization**: Uses `facebook/dinov3-vits16-pretrain-lvd1689m` (DINOv2 compatible) to score the similarity between the reference image and the generated script's output.
- **Context-Aware Prompts**: Automatically groups training data by context (e.g., "geometry", "plots") and optimizes specific prompts for each category.
- **DSPy Integration**: Leverages DSPy's `ChainOfThought` and `teleprompt` modules for robust prompt engineering and selection.
- **Sandboxed Execution**: Safely executes generated Python code in temporary environments to produce visual outputs for verification.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sarvesh-369/MORSE-PIC-Coding-Agent.git
   cd MORSE-PIC-Coding-Agent
   ```

2. **Install dependencies**:
   ```bash
   pip install dspy-ai gepa transformers torch pillow pandas pyarrow
   ```

3. **Environment Setup**:
   Set your LLM provider credentials (compatible with OpenAI-style APIs like Qwen-VL):
   ```bash
   export OPENAI_API_KEY="your_key"
   export OPENAI_API_BASE="http://localhost:8000/v1"
   export OPENAI_MODEL="Qwen/Qwen3-VL-8B-Instruct"
   ```

## 📊 Data Format

The pipeline expects data in **Parquet** format (e.g., `data/testmini.parquet`). 
Required columns:
- **`image_path`**: Absolute path to the reference image.
- **`question`**: (Optional) Textual instruction (e.g., "Draw this circle").
- **`metadata`**: A JSON string or dictionary containing a **`context`** key. This is used to group examples for specific optimization.
  - Example: `{"context": "geometry"}`

## 🖥️ Usage

### 1. Training (Prompt Optimization)
Run the optimization loop to generate refined system prompts for each unique context in your dataset:

```bash
python train_gepa.py
```
**Output**: 
- Saves optimized DSPy programs as `optimized_vlm_<context_hash>.json`.
- Creates `context_map.txt` mapping hashes to context names.

### 2. Inference & Evaluation
Run the optimized agent on your dataset to generate code and metrics:

```bash
python run_inference.py
```
**Output**:
- Creates `runs/<pid>/` directories containing:
  - `generated_code.py`: The generated Python script.
  - `metadata.json`: Visual similarity score, rationle, and execution details.
- Saves an aggregate summary in `runs/inference_summary_<timestamp>.json`.

## 📂 Structure

- **`src/code.py`**: Core VLMTask signature and Module definition.
- **`src/metrics.py`**: Visual similarity logic using DINOv2 and execution sandbox.
- **`train_gepa.py`**: Script for GEPA-based prompt optimization.
- **`run_inference.py`**: Script for running inference and evaluation.
