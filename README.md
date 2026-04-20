# BILL-Concept_Isolator-Captioner

## Overview
The **BILL-Concept_Isolator-Captioner** is a specialized ComfyUI node designed to automate the creation of high-precision captioning datasets. By integrating multimodal LLMs—specifically the Gemma 4 architecture—it bridges the gap between raw image data and the descriptive metadata required for high-quality LoRA (Low-Rank Adaptation) training.

## Technical Approach: Concept Isolation
To ensure maximum subject fidelity, the node employs a logic gate known as **Concept Isolation**. This process prevents the model from overfitting by decoupling the subject's unique identifiers from the general scene description.

### The Workflow Logic:
*   **Input Filtering:** The user specifies the subject's inherent traits (Excluded Details).
*   **Constraint Application:** The LLM is prompted to describe the image while treating the excluded details as "invisible" to the general description.
*   **Token Injection:** The identified unique traits are condensed into a single **Trigger Word**, ensuring the LoRA associates the trigger token with the visual identity.

## System Capabilities
*   **High-Fidelity Descriptive Analysis:** Focuses on lighting, composition, and texture rather than simple object labeling.
*   **Single-Pass Inference:** Reduces token drift by utilizing a single, complex prompt pass rather than iterative refinement.
*   **Integrated File Management:** Streamlined `.txt` export mapped to image filenames for immediate use in training scripts (e.g., Kohya_ss).
*   **Hardware Target:** Optimized for high-VRAM environments (24GB+), enabling the use of larger, more reasoned models.

## Installation Guide

### Backend Setup
1. Install the **Ollama** runtime environment.
2. Deploy the vision model via CLI:
   `ollama run gemma4`

### Node Integration
1. Clone the repository to the ComfyUI custom nodes directory:
   `git clone https://github.com/0-bill-0/ComfyUI-BILL-Concept_Isolator-Captioner.git`
2. Install required Python dependencies:
   `pip install -r requirements.txt`

## Hardware Requirements
| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **VRAM** | potato | 24GB - 32GB |
| **Model** | potato | Gemma 4 |
| **OS** | potato | Linux |
