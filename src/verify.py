import dspy
import argparse
import os
import sys

# Define Signature
class VerificationTask(dspy.Signature):
    """Verify if the generated image matches the reference image."""
    question: str = dspy.InputField(desc="Question about the image")
    reference_image: dspy.Image = dspy.InputField(desc="Path to the reference image")
    generated_image: dspy.Image = dspy.InputField(desc="Path to the generated image")
    generated_ground_truth: str = dspy.InputField(desc="The ground truth for the generated image for the given question")
    verification_result: str = dspy.OutputField(desc="The verification result")

# Define Module
class VerificationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(VerificationTask)

    def forward(self, reference_image, generated_image):
        return self.predictor(reference_image=reference_image, generated_image=generated_image)

def main():
    parser = argparse.ArgumentParser(description="VLM Verification Agent using dspy")
    parser.add_argument("reference_image", help="Path to the reference image")
    parser.add_argument("generated_image", help="Path to the generated image")
    parser.add_argument("question", help="Question about the image")
    parser.add_argument("generated_ground_truth", help="The ground truth for the generated image for the given question")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-VL-8B-Thinking"), help="Model name (default: env OPENAI_MODEL or gpt-4-vision-preview)")
    parser.add_argument("--api_base", default=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"), help="API base URL (default: env OPENAI_API_BASE or http://localhost:8000/v1)")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"), help="API key (default: env OPENAI_API_KEY or EMPTY)")

    args = parser.parse_args()

    # Configure dspy LM
    lm = dspy.LM(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        model_type='chat'
    )
    dspy.configure(lm=lm)

    # Read system prompt
    try:
        with open("src/verify.md", "r") as f:
            system_prompt = f.read().strip()
            # Update the docstring of the signature with the system prompt
            VerificationTask.__doc__ = system_prompt
    except FileNotFoundError:
        print("Error: src/verify.md not found.", file=sys.stderr)
        sys.exit(1)

    # Execute
    try:
        if not os.path.exists(args.reference_image):
             print(f"Error: Reference image '{args.reference_image}' not found.", file=sys.stderr)
             sys.exit(1)
        if not os.path.exists(args.generated_image):
             print(f"Error: Generated image '{args.generated_image}' not found.", file=sys.stderr)
             sys.exit(1)

        # Create dspy Image objects
        ref_img = dspy.Image.from_file(args.reference_image)
        gen_img = dspy.Image.from_file(args.generated_image)

        module = VerificationModule()
        response = module(reference_image=ref_img, generated_image=gen_img, question=args.question, generated_ground_truth=args.generated_ground_truth)
        print(response.verification_result)

    except Exception as e:
        print(f"Error during verification execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
