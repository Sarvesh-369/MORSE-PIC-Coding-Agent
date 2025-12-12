import dspy
import os

class GenerateResponse(dspy.Signature):
    """Generate a python program to recreate the given image based on the question and choices. Use any libraries you need that can recreate the image to the best of your ability."""
    image: dspy.Image = dspy.InputField(desc="The Input Image to Analyze and recreate")
    question = dspy.InputField(desc="The question to answer")
    choices = dspy.InputField(desc="The multiple choice options")
    reasoning = dspy.OutputField(desc="Reasoning for the program")   
    program = dspy.OutputField(desc="The python program to recreate the image and enclose it in ```python and ```")

def get_vlm_program(model_name: str = None, api_base: str = None, api_key: str = "EMPTY"):
    """
    Configures the VLLM model and returns the dspy program.
    
    Args:
        model_name: The name of the model being served. Defaults to env 'OPENAI_MODEL' or 'Qwen/Qwen3-VL-8B-Instruct'
        api_base: The base URL of the VLLM server. Defaults to env 'OPENAI_API_BASE' or 'http://localhost:8000/v1'
        api_key: The API key (default "EMPTY" for local VLLM)
        
    Returns:
        dspy.ChainOfThought(GenerateResponse)
    """
    
    # Defaults
    if not model_name:
        model_name = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    
    if not api_base:
        api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")

    # Configure dspy to use the self-hosted VLLM model
    # VLLM provides an OpenAI-compatible API
    # prepend openai/ to model name
    model_name = "openai/" + model_name
    lm = dspy.LM(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        max_tokens=32000
    )
    
    dspy.settings.configure(lm=lm)
    
    program = dspy.ChainOfThought(GenerateResponse)
    return lm, program


if __name__ == "__main__":
    # Example usage
    print("Initializing VLM program...")
    lm, program = get_vlm_program()
    print(f"Program created for model: {dspy.settings.lm.model}")

    image_path = "../data/images/1.jpg"
    question = "What is depicted in this image?"
    choices = ["A: A cat", "B: A dog", "C: A car", "D: A house", "E: 2D image"]

    print(f"\nRunning program with image: {image_path}, question: '{question}', choices: {choices}")
    prediction = program(image=dspy.Image(image_path), question=question, choices=choices)

    print("\nGenerated Program:")
    print(prediction.program)
