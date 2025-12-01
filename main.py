import argparse
import pandas as pd
from pathlib import Path
import sys
import os
import json
import ast

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="Run MORSE-PIC Coding Agent")
    parser.add_argument("--pid", type=str, required=True, help="Problem ID")
    parser.add_argument("--provider", type=str, required=True, help="LLM Provider")
    parser.add_argument("--max-iters", type=int, default=5, help="Max iterations")
    parser.add_argument("--verification-passes", type=int, default=1, help="Verification passes")
    parser.add_argument("--stop-on-pass", action="store_true", help="Stop on pass")
    parser.add_argument("--data", type=str, required=True, help="Path to parquet data file")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    
    args = parser.parse_args()
    
    # Read data
    try:
        df = pd.read_parquet(args.data)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)
        
    # Filter by pid
    # pid in parquet might be int or str, args.pid is str.
    # Let's try to match both.
    row = df[df['pid'].astype(str) == args.pid]
    
    if row.empty:
        print(f"PID {args.pid} not found in data.")
        sys.exit(1)
        
    # Extract data
    # Assuming columns: question, answer, metadata (dict or stringified dict)
    # The user said: "metadata which has a field context"
    
    item = row.iloc[0]
    question = item['question']
    answer = item['answer'] # generated_ground_truth? User said "generated ground truth" in orchestrator args.
    # "generated ground truth" usually means the expected answer.
    
    # Metadata
    metadata_raw = item['metadata']
    if isinstance(metadata_raw, str):
        try:
            metadata = json.loads(metadata_raw)
        except:
            try:
                metadata = ast.literal_eval(metadata_raw)
            except:
                metadata = {}
    else:
        metadata = metadata_raw if metadata_raw else {}
        
    image_context = metadata.get('context', '')
    
    # Image path
    # User said "images data/images".
    # We need to find the image file for this PID.
    # Assuming it's {pid}.png or similar.
    # Or maybe the filename is in the dataframe?
    # Let's look for a column 'image' or 'filename'.
    # If not found, default to {pid}.png in images dir.
    
    image_filename = f"{args.pid}.png"
    if 'image' in item and isinstance(item['image'], str):
         image_filename = item['image']
    elif 'filename' in item and isinstance(item['filename'], str):
         image_filename = item['filename']
         
    image_path = Path(args.images) / image_filename
    
    if not image_path.exists():
        # Try finding it recursively or with different extensions?
        # For now, just warn and proceed (or fail).
        # Let's try to check if it exists.
        print(f"Warning: Image file {image_path} does not exist.")
        # We might fail later in code.py if image doesn't exist.
        
    # Instantiate Orchestrator
    orchestrator = Orchestrator(
        project_root=project_root,
        runs_dir=project_root / "runs",
        data_dir=Path(args.data).parent,
        seed=42, # Default seed
        image_path=str(image_path),
        question=question,
        image_context=image_context,
        provider=args.provider,
        cliagent_template_path=project_root / "src/cliagent.md",
        verification_passes=args.verification_passes,
        stop_on_pass=args.stop_on_pass,
        max_iters=args.max_iters,
        phase_timeout_seconds=300, # Default timeout
        pid=args.pid,
        generated_ground_truth=str(answer)
    )
    
    orchestrator.run()

if __name__ == "__main__":
    main()
