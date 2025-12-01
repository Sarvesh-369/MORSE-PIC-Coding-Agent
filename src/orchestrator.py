import os
import json
import subprocess
from pathlib import Path
from typing import Optional

class Orchestrator:
    def __init__(
        self,
        *,
        project_root: Path,
        runs_dir: Path,
        data_dir: Path,
        seed: int,
        image_path: str,
        question: str,
        image_context: str,
        provider: str,
        cliagent_template_path: Path,
        verification_passes: int,
        stop_on_pass: bool,
        max_iters: int,
        phase_timeout_seconds: int,
        pid: str,
        generated_ground_truth: str
    ):
        self.project_root = project_root
        self.runs_dir = runs_dir
        self.data_dir = data_dir
        self.seed = seed
        self.image_path = image_path
        self.question = question
        self.image_context = image_context
        self.provider = provider
        self.cliagent_template_path = cliagent_template_path
        self.verification_passes = verification_passes
        self.stop_on_pass = stop_on_pass
        self.max_iters = max_iters
        self.phase_timeout_seconds = phase_timeout_seconds
        self.pid = pid
        self.generated_ground_truth = generated_ground_truth

    def run(self):
        # Create run directory
        run_dir = self.runs_dir / self.pid
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created run directory: {run_dir}")

        # Step 1: Generate code using code.py
        generate_script_path = run_dir / "generate.py"
        print(f"Generating code to: {generate_script_path}")
        
        # Construct command for code.py
        # python src/code.py <image_path> <question> <output_path>
        code_cmd = [
            "python",
            str(self.project_root / "src/code.py"),
            self.image_path,
            self.question,
            str(generate_script_path)
        ]
        
        try:
            subprocess.run(code_cmd, check=True, cwd=self.project_root)
            print("Code generation successful.")
        except subprocess.CalledProcessError as e:
            print(f"Error generating code: {e}")
            return

        # Step 2: Format cliagent.md and run cliagent.py
        prompt_path = run_dir / "prompt.md"
        print(f"Formatting prompt to: {prompt_path}")
        
        try:
            with open(self.cliagent_template_path, "r") as f:
                template_content = f.read()
            
            # Format the template
            # Assuming template has {output_dir} and {max_iters} placeholders
            # We can add more if needed, but based on request these seem relevant
            formatted_prompt = template_content.format(
                output_dir=str(run_dir),
                max_iters=self.max_iters,
                reference_image_path=self.image_path,
                question=self.question,
                generated_ground_truth=self.generated_ground_truth,
                verify_script_path=str(self.project_root / "src/verify.py")
            )
            
            with open(prompt_path, "w") as f:
                f.write(formatted_prompt)
                
        except Exception as e:
            print(f"Error formatting prompt: {e}")
            # If formatting fails (e.g. missing keys), we might want to just write the raw content 
            # or handle it gracefully. For now, let's try to write raw if format fails 
            # or just proceed. 
            pass

        # Run cliagent.py
        # python src/cliagent.py --prompt {prompt_path} --cwd {run_dir} --provider {provider}
        print("Running CLI agent...")
        cli_cmd = [
            "python",
            str(self.project_root / "src/cliagent.py"),
            "--prompt", str(prompt_path),
            "--cwd", str(run_dir),
            "--provider", self.provider
        ]
        
        try:
            subprocess.run(cli_cmd, check=True, cwd=self.project_root)
            print("CLI agent execution successful.")
        except subprocess.CalledProcessError as e:
            print(f"Error running CLI agent: {e}")
            # Continue to save metadata even if agent fails? 
            # Maybe.

        # Step 3: Save metadata
        metadata_path = run_dir / "metadata.json"
        metadata = {
            "question": self.question,
            "generated_ground_truth": self.generated_ground_truth,
            "image_context": self.image_context,
            "pid": self.pid,
            "seed": self.seed,
            "provider": self.provider,
            "max_iters": self.max_iters
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata to: {metadata_path}")
