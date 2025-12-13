
import os
import dspy
from src.vlm import get_vlm_program, GenerateResponse
from src.gepa_metrics import GEPAMetrics

def main():
    # 1. Initialize VLM
    print("Initializing VLM program...")
    lm, program = get_vlm_program()
    
    # 2. Initialize Metrics
    print("Initializing GEPA Metrics...")
    # This might take time to load the vision encoder
    metrics = GEPAMetrics(similarity_threshold=0.6) 

    # 3. Define Test Input
    # Using a known image from data/images if available
    image_path = "data/images/1.jpg"
    if not os.path.exists(image_path):
        # Fallback handling or try to find an existing jpg
        possible_images = [f for f in os.listdir("data/images") if f.endswith(".jpg")]
        if possible_images:
            image_path = os.path.join("data/images", possible_images[0])
        else:
            print("No test images found in data/images.")
            return

    question = "What is depicted in this image?"
    choices = ["A: A cat", "B: A dog", "C: A car", "D: A house", "E: Geometric shape"]
    
    print(f"\n--- Running Test ---")
    print(f"Image: {image_path}")
    print(f"Question: {question}")
    
    # 4. Run VLM Prediction
    # Ensure dspy.Image wrapping as per previous fixes
    dspy_image = dspy.Image(image_path)
    
    print("Querying VLM...")
    try:
        prediction = program(
            image=dspy_image, 
            question=question, 
            choices=choices
        )
    except Exception as e:
        print(f"VLM Query failed: {e}")
        return

    print("Prediction received.")
    print(f"Reasoning: {getattr(prediction, 'reasoning', 'N/A')[:100]}...")
    print(f"Program Code Length: {len(getattr(prediction, 'program', ''))} chars")

    # 5. Run Metric Evaluation
    print("\nExecuting Metric Evaluation...")
    
    # The metric expects an 'example' with the ground truth image
    # For this test, the ground truth is the input image itself (since we want to recreate it)
    example = dspy.Example(image=image_path) 
    
    try:
        result = metrics.metric(example, prediction)
        print(f"\n--- Metric Result ---")
        print(f"Score: {result.score}")
        print(f"Feedback: {result.feedback}")
    except Exception as e:
        print(f"Metric evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
