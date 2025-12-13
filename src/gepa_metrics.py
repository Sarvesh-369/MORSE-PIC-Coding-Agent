
import dspy
import torch
import numpy as np
from PIL import Image
import os
import subprocess
import sys
import traceback
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

class GEPAMetrics:
    def __init__(self, vision_encoder_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m", similarity_threshold: float = 0.6):
        self.vision_encoder_model = vision_encoder_model
        self.similarity_threshold = similarity_threshold
        
        # Initialize model and processor
        print(f"Loading vision encoder: {self.vision_encoder_model}...")
        self.processor = AutoImageProcessor.from_pretrained(self.vision_encoder_model)
        self.model = AutoModel.from_pretrained(
            self.vision_encoder_model,
            device_map="auto",
            attn_implementation="sdpa"
        )
        # Note: device_map="auto" handles device placement, so explicit .to(device) might be redundant 
        # but inputs need to be on model.device.
            
    def _get_embedding(self, image):
        # Convert common wrappers to str path/URL if needed (e.g., dspy.Image)
        if not isinstance(image, (str, Image.Image)):
            for attr in ("path", "filepath", "file_path", "url"):
                value = getattr(image, attr, None)
                if isinstance(value, str) and value:
                    image = value
                    break

        # Convert str path or URL to image if needed
        if isinstance(image, str):
            try:
                # load_image handles both local paths and URLs
                image = load_image(image)
            except Exception:
                pass
        
        if not isinstance(image, Image.Image):
             return None

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        # Use pooler_output as requested
        return outputs.pooler_output.cpu().numpy().flatten()

    def compute_similarity(self, image1, image2):
        """Computes cosine similarity between two images."""
        emb1 = self._get_embedding(image1)
        emb2 = self._get_embedding(image2)
        
        if emb1 is None or emb2 is None:
            return 0.0, "Failed to process one or both images."
            
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0, "Zero norm embedding encountered."
            
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(similarity), "Success"

    def _extract_pid(self, example, default: str = "unknown") -> str:
        if example is None:
            return default

        for key in ("pid", "id", "example_id"):
            if hasattr(example, "get"):
                try:
                    value = example.get(key)
                except Exception:
                    value = None
                if value not in (None, ""):
                    return str(value)

            value = getattr(example, key, None)
            if value not in (None, ""):
                return str(value)

        store = getattr(example, "_store", None)
        if isinstance(store, dict):
            for key in ("pid", "id", "example_id"):
                value = store.get(key)
                if value not in (None, ""):
                    return str(value)

        return default

    def _write_generate_scripts(self, run_dir: str, code: str) -> str:
        os.makedirs(run_dir, exist_ok=True)

        raw_path = os.path.join(run_dir, "generate_raw.py")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(code.rstrip() + "\n")

        generate_path = os.path.join(run_dir, "generate.py")
        with open(generate_path, "w", encoding="utf-8") as f:
            f.write(code.rstrip() + "\n")

        return generate_path

    def _run_generate_script(self, run_dir: str, timeout_s: int = 60):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        result = subprocess.run(
            [sys.executable, "generate.py"],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        return result

    def metric(self, example, prediction, trace=None, pred_name=None, pred_trace=None):
        program_code = getattr(prediction, 'program', '')
        
        # Extract code block
        code = ""
        if "```python" in program_code:
            code = program_code.split("```python")[1].split("```")[0].strip()
        elif "```" in program_code:
            code = program_code.split("```")[0].strip()
        else:
            code = program_code
            
        if not code:
            return dspy.Prediction(score=0, feedback="No code found in the prediction.")

        # Save artifacts (generate.py first), then execute it to produce image.png
        pid = self._extract_pid(example, default="unknown")
        run_dir = os.path.join("runs", str(pid))
        os.makedirs(run_dir, exist_ok=True)

        try:
            self._write_generate_scripts(run_dir, code)
        except Exception as e:
            return dspy.Prediction(score=0, feedback=f"Failed to write generate.py: {e}\nTraceback: {traceback.format_exc()}")

        try:
            exec_result = self._run_generate_script(run_dir)
        except subprocess.TimeoutExpired as e:
            return dspy.Prediction(
                score=0,
                feedback=f"generate.py timed out after {e.timeout}s.",
            )
        except Exception as e:
            return dspy.Prediction(score=0, feedback=f"Failed to run generate.py: {e}\nTraceback: {traceback.format_exc()}")

        if exec_result.returncode != 0:
            return dspy.Prediction(
                score=0,
                feedback=(
                    "generate.py failed.\n"
                    f"Return code: {exec_result.returncode}\n"
                    f"STDOUT:\n{exec_result.stdout}\n"
                    f"STDERR:\n{exec_result.stderr}"
                ),
            )

        generated_image_path = os.path.join(run_dir, "image.png")
        if not os.path.exists(generated_image_path):
            return dspy.Prediction(
                score=0,
                feedback=(
                    "generate.py ran but did not produce runs/<pid>/image.png.\n"
                    "Make sure your code saves the output image as image.png in the current working directory.\n"
                    f"STDOUT:\n{exec_result.stdout}\n"
                    f"STDERR:\n{exec_result.stderr}"
                ),
            )

        similarity, msg = self.compute_similarity(example.image, generated_image_path)
        print(f"PID: {pid} | Similarity score: {similarity}")
            
        score = 1 if similarity >= self.similarity_threshold else 0
        
        feedback = f"Similarity Score: {similarity:.4f}"
        if score == 0:
            feedback += f"\nThreshold is {self.similarity_threshold}. Evaluation failed. Reason: {msg}"
        else:
            feedback += "\nSuccess!"
            
        return dspy.Prediction(score=score, feedback=feedback)
