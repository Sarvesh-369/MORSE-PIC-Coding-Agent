
import dspy
import torch
import numpy as np
from PIL import Image
import io
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

        # Execute code
        local_scope = {}
        
        try:
            exec(code, {}, local_scope)
        except Exception as e:
            return dspy.Prediction(score=0, feedback=f"Error executing code: {str(e)}\nTraceback: {traceback.format_exc()}")
        
        # Find image in local_scope
        generated_image = None
        # Priority to 'image' or 'img' variables
        if 'image' in local_scope and isinstance(local_scope['image'], Image.Image):
            generated_image = local_scope['image']
        elif 'img' in local_scope and isinstance(local_scope['img'], Image.Image):
            generated_image = local_scope['img']
        else:
            # excessive search
            for val in reversed(list(local_scope.values())):
                if isinstance(val, Image.Image):
                    generated_image = val
                    break
        
        if not generated_image:
             return dspy.Prediction(score=0, feedback="Code executed but no PIL Image object was found in the variables.")
        
        similarity, msg = self.compute_similarity(example.image, generated_image)
        print("Similarity score:", similarity)
            
        score = 1 if similarity >= self.similarity_threshold else 0
        
        feedback = f"Similarity Score: {similarity:.4f}"
        if score == 0:
            feedback += f"\nThreshold is {self.similarity_threshold}. Evaluation failed. Reason: {msg}"
        else:
            feedback += "\nSuccess!"
            
        return dspy.Prediction(score=score, feedback=feedback)