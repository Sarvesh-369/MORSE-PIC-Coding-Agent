import dspy
import argparse
import os
import sys

# Define Signature
class VLMTask(dspy.Signature):
    """
    You are an expert Python developer and Artist.
    Your task is to write a Python script that visually recreates the input image.
    
    1. Analyze the input image to understand its shapes, colors, layout, and style.
    2. Write a Python script using libraries like Matplotlib, PIL, Turtle, or Pygame to generate an image that looks as close as possible to the input.
    3. The code should save the output to 'image.png' or similar.
    4. Ensure the code is self-contained and runnable.
    """
    
    image: dspy.Image = dspy.InputField(desc="Reference image object")
    question: str = dspy.InputField(desc="Specific instructions or 'Recreate this image visually'")
    rationale: str = dspy.OutputField(desc="Step-by-step reasoning on how to recreate the image")
    code: str = dspy.OutputField(desc="Executable Python code enclosed in markdown code blocks")

# Define Module
class VLMModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(VLMTask)

    def forward(self, image, question="Recreate this image visually."):
        # User requested to pass the image object itself.
        # The input 'image' should already be a dspy.Image object from the caller.
        return self.prog(image=image, question=question)

def main():
    parser = argparse.ArgumentParser(description="VLM Code Generator using dspy and GEPA")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("question", nargs="?", default="Recreate this image visually.", help="Question to ask about the image")
    parser.add_argument("output_path", help="Path to save the generated code")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-VL-8B-Instruct"), help="Model name")
    parser.add_argument("--api_base", default=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"), help="API base URL")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"), help="API key")

    args = parser.parse_args()

    # Configure dspy LM
    model_name = args.model
    if not model_name.startswith("openai/") and not model_name.startswith("gpt-"):
        model_name = "openai/" + model_name
        
    lm = dspy.LM(
        model=model_name,
        api_base=args.api_base,
        api_key=args.api_key,
        model_type='chat'
    )
    dspy.configure(lm=lm)

    # Execute
    try:
        if not os.path.exists(args.image_path):
             print(f"Error: Image file '{args.image_path}' not found.", file=sys.stderr)
             sys.exit(1)

        module = VLMModule()
        # Ensure we pass absolute path if possible or just the path provided
        abs_image_path = os.path.abspath(args.image_path)
        
        # Instantiate dspy.Image object
        img = dspy.Image(abs_image_path)
        
        response = module(image=img, question=args.question)
        
        print("Rationale:", response.rationale)
        print("Code:", response.code)

        # Save code to file
        code_content = response.code
        # Simple cleanup of markdown code blocks
        if code_content.startswith("```python"):
            code_content = code_content[9:]
        elif code_content.startswith("```"):
            code_content = code_content[3:]
        if code_content.endswith("```"):
            code_content = code_content[:-3]
        
        with open(args.output_path, "w") as f:
            f.write(code_content.strip())
        print(f"Generated code saved to {args.output_path}")

    except Exception as e:
        print(f"Error during VLM execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

