import os
import glob
import json
import dspy
import pandas as pd
import sys
import datetime
import subprocess 
import shutil
import uuid
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from code import VLMModule, VLMTask
from metrics import visual_similarity_metric

def main():
    # 1. Setup Environment
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"inference_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Configure LM
    student_model = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    
    if not student_model.startswith("openai/") and not student_model.startswith("gpt-"):
        student_model = "openai/" + student_model

    lm = dspy.LM(
        model=student_model,
        api_base=api_base,
        api_key=api_key,
        model_type='chat'
    )
    dspy.configure(lm=lm)
    
    # 2. Load Module
    module = VLMModule()
    optimized_path = "optimized_vlm.json"
    if os.path.exists(optimized_path):
        print(f"Loading optimized module from {optimized_path}")
        module.load(optimized_path)
    else:
        print("Warning: optimized_vlm.json not found. Using un-optimized module.")

    # 3. Load Data
    examples = []
    # Try finding any parquet in data/
    parquet_files = glob.glob("data/*.parquet")
    if parquet_files:
        for p_file in parquet_files:
            print(f"Loading data from {p_file}...")
            try:
                df = pd.read_parquet(p_file)
                # Logic to determine columns
                # MathVista often has 'image', 'question' or similar
                # We'll look for common usage
                for idx, row in df.iterrows():
                    img_path = row.get('image_path') or row.get('image')
                    # If image path is relative or needs handling
                    # Assuming local paths for now as per instructions
                    if img_path: 
                         # Verify path
                         if not os.path.isabs(img_path):
                             img_path = os.path.abspath(img_path)
                         if os.path.exists(img_path):
                             ex = dspy.Example(
                                 image_path=img_path,
                                 question=row.get('question', "Write a Python script to recreate this image visually."),
                                 pid=row.get('pid', str(idx)) # unique id if available
                             ).with_inputs("image_path", "question")
                             examples.append(ex)
            except Exception as e:
                print(f"Error reading {p_file}: {e}")
    
    # Fallback to images dir
    if not examples:
        image_files = glob.glob("data/images/*.png")
        for i, img_path in enumerate(image_files):
            ex = dspy.Example(
                image_path=os.path.abspath(img_path),
                question="Write a Python script to recreate this image visually.",
                pid=str(i)
            ).with_inputs("image_path", "question")
            examples.append(ex)
            
    if not examples:
        print("No data found for inference.")
        return

    print(f"Running inference on {len(examples)} examples...")

    # 4. Inference Loop
    results = []

    # Ensure runs directory exists
    runs_base_dir = "runs"
    os.makedirs(runs_base_dir, exist_ok=True)

    import ast
    import hashlib

    # Keep a cache of loaded modules to avoid reloading from disk every time
    module_cache = {}
    base_module = VLMModule()
    
    for ex in tqdm(examples):
        pid = getattr(ex, 'pid', 'unknown')
        context_str = getattr(ex, 'context', '') or 'default'
        
        # Structure: runs/<pid>/
        ex_dir = os.path.join(runs_base_dir, str(pid))
        os.makedirs(ex_dir, exist_ok=True)
        
        try:
            # Determine which module to use
            ctx_hash = hashlib.md5(context_str.encode('utf-8')).hexdigest()
            opt_path = f"optimized_vlm_{ctx_hash}.json"
            
            current_module = base_module
            
            if os.path.exists(opt_path):
                if ctx_hash not in module_cache:
                    print(f"Loading optimized module for context '{context_str}'...")
                    loaded_mod = VLMModule()
                    loaded_mod.load(opt_path)
                    module_cache[ctx_hash] = loaded_mod
                current_module = module_cache[ctx_hash]
            
            # Generate code
            # Note: No 'context' passed as input field, per instruction
            # Pass dspy.Image object as per "pass image itself" request
            img_obj = dspy.Image(ex.image_path)
            pred = current_module(image=img_obj, question=ex.question)
            
            # Save raw code
            code_path = os.path.join(ex_dir, "generated_code.py")
            with open(code_path, "w") as f:
                f.write(pred.code)
            
            # Calculate Metric
            # Note: visual_similarity_metric creates its own temp sandbox,
            # so we let it do its verify.
            # But we also want to save the result image here for inspection if possible.
            # actually visual_similarity_metric puts it in temp and deletes it...

            # So we might want to manually execute here to keep the image,
            # OR just trust the metric score.
            # The USER Instruction said: "Save the code to the run directory and execute it to produce the final image."
            # So we must execute it here too.

            score = visual_similarity_metric(ex, pred)
            
            # Execute locally to save image
            clean_code = pred.code
            if clean_code.startswith("```python"): clean_code = clean_code[9:]
            elif clean_code.startswith("```"): clean_code = clean_code[3:]
            if clean_code.endswith("```"): clean_code = clean_code[:-3]
            
            with open(code_path, "w") as f:
                f.write(clean_code) # overwrite with clean code for execution
                
            try:
                subprocess.run(
                    ["python3", "generated_code.py"], 
                    cwd=ex_dir,
                    timeout=20,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"Execution failed for {pid}: {e}")

            # Collect metadata
            result_entry = {
                "pid": pid,
                "image_path": ex.image_path,
                "question": ex.question,
                "context": context_str,
                "optimized_prompt_used": os.path.exists(opt_path),
                "generated_code_path": code_path,
                "dino_score": score,
                "rationale": getattr(pred, 'rationale', "")
            }
            results.append(result_entry)
            
            # Save per-problem metadata in its folder too
            with open(os.path.join(ex_dir, "metadata.json"), "w") as f:
                json.dump(result_entry, f, indent=2)
            
        except Exception as e:
            print(f"Error on example {pid}: {e}")

    # 5. Save Global Metadata
    # Since we are not doing a single "run" folder anymore effectively, we can save summary in runs/
    # or create a summary file with timestamp
    summary_path = os.path.join(runs_base_dir, f"inference_summary_{run_id}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Calculate average score
    scores = [r['dino_score'] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"Inference complete. Average Visual Similarity Score: {avg_score:.4f}")
    print(f"Results saved to {run_dir}")

if __name__ == "__main__":
    main()
