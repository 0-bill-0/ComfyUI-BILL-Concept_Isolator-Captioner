import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
import os

class BillLoRACaptionNode:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"

    @classmethod
    def INPUT_TYPES(s):
        installed_models = BillLoRACaptionNode.get_ollama_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (installed_models, ), 
                "excluded_details": ("STRING", {"multiline": True, "default": "a vintage red sports car"}), 
                "trigger_word": ("STRING", {"default": "MyCar"}), 
            },
            "optional": {
                "filename": ("STRING", {"default": "caption_output"}),
                "save_folder": ("STRING", {"default": "comfyui/output/captions"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "caption_image"
    CATEGORY = "BillNodes/Training"

    @staticmethod
    def get_ollama_models():
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                return models if models else ["gemma4"]
        except:
            pass
        return ["gemma4"]

    def image_to_base64(self, image_tensor):
        try:
            # ComfyUI images are [B, H, W, C]. We take the first image in the batch.
            img_np = image_tensor[0].cpu().numpy()
            # Scale 0-1 to 0-255
            img_np = (img_np * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Image conversion failed: {str(e)}")

    def caption_image(self, image, model, excluded_details, trigger_word, filename="caption_output", save_folder="comfyui/output/captions"):
        # 1. Handle Image Conversion
        try:
            img_b64 = self.image_to_base64(image)
        except Exception as e:
            return (f"CRITICAL ERROR: {str(e)}",)

        # 2. Construct the Prompt
        master_prompt = (
            f"You are a professional LoRA dataset captioner. Describe the image while isolating a specific entity.\n\n"
            f"TARGET ENTITY: '{excluded_details}'\n"
            f"TRIGGER WORD: '{trigger_word}'\n\n"
            f"RULE: Replace all physical descriptions of the TARGET ENTITY with the TRIGGER WORD. "
            f"Describe the environment, lighting, and pose in high-fidelity cinematic detail. "
            f"Example: If target is 'red car' and trigger is 'MyCar', write 'MyCar parked on a rainy street' NOT 'A red car parked on a rainy street'.\n\n"
            f"Output one fluid paragraph. Start directly. No intro."
        )

        payload = {
            "model": model,
            "prompt": master_prompt,
            "images": [img_b64],
            "stream": False,
            "options": { "temperature": 0.1 }
        }

        # 3. API Request
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            final_caption = response.json().get("response", "").strip()
            
            if not final_caption:
                final_caption = "Error: Ollama returned an empty response."
                
        except requests.exceptions.ConnectionError:
            final_caption = "Error: Could not connect to Ollama. Is it running?"
        except Exception as e:
            final_caption = f"API Error: {str(e)}"

        # 4. Robust Saving Logic
        if filename:
            try:
                # Create folder if it doesn't exist
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=True)
                
                # Clean filename to avoid path errors
                safe_filename = "".join([c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')]).rstrip()
                file_path = os.path.join(save_folder, f"{safe_filename}.txt")
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(final_caption)
            except Exception as e:
                # We don't return here because we still want the caption string in ComfyUI
                final_caption += f" | (File Save Error: {str(e)})"

        return (final_caption,)
