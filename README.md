\# 🚀 Bill's LoRA Detail Captioner (Gemma 4 Edition)



A professional-grade image captioning node for ComfyUI, designed to generate high-fidelity datasets for LoRA training using next-generation multimodal LLMs.



\## 🧠 The Power of Gemma 4

This node is optimized for \*\*Gemma 4\*\* (and similar high-parameter Vision models like Llama 3.2 Vision). Unlike older models, Gemma 4 possesses superior reasoning capabilities, allowing it to follow complex constraints while maintaining a cinematic, verbose vocabulary.



\## 🌟 The Logic: Concept Isolation

To train a high-quality LoRA, you must isolate the subject's unique identity from the scene. This node uses \*\*Concept Isolation\*\*:



1\. \*\*Identify:\*\* You provide the "Excluded Details" (e.g., "man with grey hair and blue eyes").

2\. \*\*Isolate:\*\* The AI is forbidden from describing those specific traits.

3\. \*\*Trigger:\*\* The AI replaces those traits with a unique \*\*Trigger Word\*\* (e.g., "Bill").



\*\*Result:\*\* The LoRA learns that "grey hair and blue eyes" = `Bill`, rather than thinking they are just general features of the image.



\## ✨ Key Features

\- \*\*Cinematic Analysis:\*\* Generates high-fidelity descriptions focusing on chromatic hues, textures, and volumetric lighting.

\- \*\*Single-Pass Logic:\*\* Leverages the intelligence of Gemma 4 to observe and filter the image in one pass, eliminating the "invention" problems found in multi-pass systems.

\- \*\*Auto-Save Workflow:\*\* Automatically saves captions as `.txt` files matched to image filenames for instant training.

\- \*\*VRAM Optimized:\*\* Purpose-built for users with \*\*24GB - 32GB VRAM\*\* to run heavy, high-intelligence models.



\## 🛠️ Installation



1\. \*\*Install Ollama:\*\* Download \[Ollama](https://ollama.ai/).

2\. \*\*Pull a Vision Model:\*\* Run the following in your terminal:

&#x20;  - `ollama run gemma4` (Recommended)

&#x20;  - `ollama run llava` (Alternative)

3\. \*\*Install Node:\*\*

&#x20;  - Clone this repository into your `ComfyUI/custom\_nodes/` folder:

&#x20;    ```bash

&#x20;    git clone https://github.com/YOUR\_USERNAME/ComfyUI-BillLoRA-Captioner.git

&#x20;    ```

4\. \*\*Install Dependencies:\*\*

&#x20;  ```bash

&#x20;  pip install -r requirements.txt



