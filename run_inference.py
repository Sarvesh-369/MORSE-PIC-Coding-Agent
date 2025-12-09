
import os
import dspy
import json
import traceback
from PIL import Image
from src.build_gepa_dataset import BuildGEPADataset
from src.vlm import get_vlm_program
from src.gepa_metrics import GEPAMetrics

def main():
    # 1. Load Data
    print("Loading test dataset...")
    # Instantiate builder mostly to access load_test method
    builder = BuildGEPADataset()
    test_file = os.path.join("data", "testmini.parquet")
    test_set = builder.load_test(test_file)
    print(f"Test size: {len(test_set)}")

    # 2. Load Compiled Program
    print("Loading compiled program...")
    lm, _ = get_vlm_program() # Need LM configured to run program
    
    from src.vlm import GenerateResponse
    program = dspy.ChainOfThought(GenerateResponse)
    
    compiled_path = "compiled_program.json"
    if os.path.exists(compiled_path):
        program.load(compiled_path)
        print("Loaded compiled program.")
    else:
        print(f"Warning: {compiled_path} not found. Running uncompiled program.")
    
    # 3. Setup Metrics for similarity calculation
    print("Initializing metrics...")
    metrics = GEPAMetrics()
    
    # 4. Inference Loop
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)
    
    print("Starting inference...")
    for idx, example in enumerate(test_set):
        pid = example.get('pid', str(idx)) # Assuming pid exists or using index
        
        # Create folder
        run_folder = os.path.join(runs_dir, pid)
        os.makedirs(run_folder, exist_ok=True)
        
        try:
            # Predict
            pred = program(
                image=example.image, 
                question=example.question, 
                choices=example.choices
            )
            
            # Extract Code
            code = ""
            program_code = getattr(pred, 'program', '')
            if "```python" in program_code:
                code = program_code.split("```python")[1].split("```")[0].strip()
            elif "```" in program_code:
                code = program_code.split("```")[0].strip()
            else:
                code = program_code
            
            # Save generated code
            with open(os.path.join(run_folder, "generate.py"), "w") as f:
                f.write(code)
                
            # Execute to get image
            local_scope = {}
            generated_image = None
            exec_error = None
            
            try:
                exec(code, {}, local_scope)
                
                # Find image
                if 'image' in local_scope and isinstance(local_scope['image'], Image.Image):
                    generated_image = local_scope['image']
                elif 'img' in local_scope and isinstance(local_scope['img'], Image.Image):
                    generated_image = local_scope['img']
                else:
                    for val in reversed(list(local_scope.values())):
                        if isinstance(val, Image.Image):
                            generated_image = val
                            break
            except Exception as e:
                exec_error = str(e)
                
            # Save image
            if generated_image:
                generated_image.save(os.path.join(run_folder, "image.png"))
            
            # Compute Similarity
            similarity = 0.0
            sim_msg = "No image generated"
            if generated_image:
                similarity, sim_msg = metrics.compute_similarity(example.image, generated_image)
            
            # Save Metadata
            metadata = {
                "question": example.question,
                "choices": example.choices,
                # "programatically generated ground truth" - dataset might not have GT code?
                # We'll save ground truth answer or whatever is available e.g. 'answer'
                "ground_truth_answer": getattr(example, 'answer', None),
                "image_similarity": similarity,
                "similarity_msg": sim_msg,
                "execution_error": exec_error,
                "reasoning": getattr(pred, 'reasoning', '')
            }
            
            with open(os.path.join(run_folder, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)
                
            print(f"Processed {pid}: Sim={similarity:.4f}")
            
        except Exception as e:
            print(f"Error processing {pid}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
