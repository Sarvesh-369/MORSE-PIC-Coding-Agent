import os
import glob
import dspy
import pandas as pd
import sys

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from code import VLMModule
from metrics import visual_similarity_metric

# Try importing GEPA
try:
    from dspy.teleprompt import GEPA
except ImportError:
    try:
        from gepa import GEPA
    except ImportError:
        print("Error: GEPA not found in dspy.teleprompt or as a standalone package 'gepa'.")
        print("Please ensure 'gepa' is installed.")
        # Fallback to BootstrapFewShot just so the script doesn't crash if GEPA is missing during dry run logic
        # But explicitly warn.
        from dspy.teleprompt import BootstrapFewShot as GEPA 
        print("WARNING: Falling back to BootstrapFewShot for demonstration purposes.")

def main():
    # 1. Configure LMs
    # Student
    student_model = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    
    if not student_model.startswith("openai/") and not student_model.startswith("gpt-"):
        student_model = "openai/" + student_model

    lm_student = dspy.LM(
        model=student_model,
        api_base=api_base,
        api_key=api_key,
        model_type='chat',
        # Timeout can be important for optimization loops
        kwargs={"timeout": 60} 
    )
    
    # Teacher (Reflection)
    # Using same model for self-reflection as requested
    lm_teacher = dspy.LM(
        model=student_model,
        api_base=api_base,
        api_key=api_key,
        model_type='chat'
    )
    
    dspy.configure(lm=lm_student)
    
    # 2. Load Data
    examples = []
    
    # Try Parquet first
    parquet_path = "data/testmini.parquet"
    if os.path.exists(parquet_path):
        print(f"Loading data from {parquet_path}...")
        try:
            df = pd.read_parquet(parquet_path)
            import json
            import ast
            
            for _, row in df.iterrows():
                # specific column names depend on dataset, assuming commonly used ones or 'image_path'
                img_path = row.get('image_path') or row.get('image')
                if img_path and os.path.exists(img_path):
                     # Parse metadata for context
                     metadata_raw = row.get('metadata', '{}')
                     context_text = ""
                     if metadata_raw:
                         if isinstance(metadata_raw, str):
                             try:
                                 meta_dict = json.loads(metadata_raw)
                             except:
                                 try:
                                     meta_dict = ast.literal_eval(metadata_raw)
                                 except:
                                     meta_dict = {}
                         elif isinstance(metadata_raw, dict):
                             meta_dict = metadata_raw
                         else:
                             meta_dict = {}
                         
                         context_text = meta_dict.get('context', '') or meta_dict.get('image_context', '')

                     # Construct example
                     ex = dspy.Example(
                         image_path=os.path.abspath(img_path),
                         question=row.get('question', "Recreate this image visually."),
                         context_key=context_text # Store for grouping, not input
                     ).with_inputs("image_path", "question")
                     examples.append(ex)
        except Exception as e:
            print(f"Error loading parquet: {e}")
            
    # If no examples from parquet, try globbing pngs
    if not examples:
        print("Loading images from data/images/...")
        image_files = glob.glob("data/images/*.png")
        for img_path in image_files:
            ex = dspy.Example(
                image_path=os.path.abspath(img_path),
                question="Write a Python script to recreate this image visually.",
                context_key="default"
            ).with_inputs("image_path", "question")
            examples.append(ex)
            
    if not examples:
        print("No data found. Exiting.")
        return

    print(f"Loaded {len(examples)} examples.")
    
    # Group by context
    from collections import defaultdict
    grouped_examples = defaultdict(list)
    for ex in examples:
        ctx = getattr(ex, 'context_key', 'default')
        if not ctx: ctx = 'default'
        grouped_examples[ctx].append(ex)
        
    print(f"Found {len(grouped_examples)} unique contexts.")
    
    # Optimize for each context
    import hashlib
    
    for context_str, group in grouped_examples.items():
        print(f"Optimizing for context: '{context_str}' with {len(group)} examples...")
        
        # Skip if too few examples? Let's try anyway or maybe strict minimum
        if len(group) < 1:
            continue
            
        teleprompter = GEPA(
            metric=visual_similarity_metric,
            prompt_model=lm_teacher,
            breadth=5,
            depth=3,
            verbose=True
        )
        
        program = VLMModule()
        
        try:
            optimized_program = teleprompter.compile(
                program,
                trainset=group,
            )
            
            # Create safe filename
            # Use hash of context to avoid filesystem issues
            ctx_hash = hashlib.md5(context_str.encode('utf-8')).hexdigest()
            output_path = f"optimized_vlm_{ctx_hash}.json"
            
            # Also save a mapping of hash -> context text for reference
            with open("context_map.txt", "a") as f:
                f.write(f"{ctx_hash}\t{context_str}\n")
                
            optimized_program.save(output_path)
            print(f"Optimized program for '{context_str}' saved to {output_path}")
            
        except Exception as e:
            print(f"Error optimizing for context '{context_str}': {e}")

if __name__ == "__main__":
    main()
