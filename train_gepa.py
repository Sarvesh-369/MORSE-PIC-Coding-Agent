
import os
import dspy
from dspy import GEPA # Assuming user's import is correct
from src.build_gepa_dataset import BuildGEPADataset
from src.vlm import get_vlm_program
from src.gepa_metrics import GEPAMetrics

def main():
    # 1. Dataset Setup
    print("Building dataset...")
    builder = BuildGEPADataset(save_dataset=False)
    train_set, val_set = builder.build()
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")

    # 2. VLM Setup
    print("Initializing VLM...")
    lm, program = get_vlm_program()
    
    # 3. Metrics Setup
    print("Initializing Metrics...")
    # Requires torch/transformers which might fail if dependencies missing, 
    # but code will look correct.
    metrics_obj = GEPAMetrics()
    
    # 4. Optimizer Setup
    print("Setting up GEPA Optimizer...")
    # Use the same LM for reflection
    reflection_lm = lm 
    
    optimizer = GEPA(
        metric=metrics_obj.metric,
        auto="heavy",
        num_threads=32,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=reflection_lm,
        log_dir="./gepa_logs"
    )

    # 5. Optimization
    print("Starting compilation...")
    optimized_program = optimizer.compile(
        program,
        trainset=train_set,
        valset=val_set,
    )
    
    # 6. Save
    output_path = "compiled_program.json"
    print(f"Saving compiled program to {output_path}...")
    optimized_program.save(output_path)
    print("Done.")

if __name__ == "__main__":
    main()
