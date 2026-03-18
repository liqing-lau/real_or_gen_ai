## real_or_gen_ai

### 1. Project Overview

- **Project Name**: `real_or_gen_ai`  
- **Task Type**: Image binary classification – determine whether an input image is **AI‑generated** or **real images**.  
- **Project Goal**: Build an engineering‑oriented small project suitable for a 5‑person team, including:
  - Data loading and preprocessing
  - Model training and evaluation
  - Inference scripts
  - (Optional) Web demo for interactive use


---

### 2. Dataset

- **Primary Data Source**: Kaggle dataset “AI generated images vs real images”.  
- **Expected Directory Layout (after preparation)**:
  - `data/train/real/`
  - `data/train/ai/`
  - `data/val/real/`
  - `data/val/ai/`
  - (optional) `data/test/...`
- **Label Convention**:
  - `0 = real`
  - `1 = ai`

#### 2.1 Preprocessing

- Resize all images to `224 × 224`.  
- Convert to RGB if needed.  
- Normalize using ImageNet statistics:
  - mean = `[0.485, 0.456, 0.406]`
  - std = `[0.229, 0.224, 0.225]`

#### 2.2 Data Augmentation

Applied on the training set only (examples, will be tuned in experiments):

- Random horizontal flip  
- Small random rotations  
- Light color jitter (brightness/contrast/saturation)  

Validation (and test) sets only use deterministic transforms: resize, center crop/resize, normalization.

---

### 3. Technical Stack & Architecture

- **Framework**: PyTorch + torchvision (optionally `timm` for more backbones).  
- **Training Paradigm**: Transfer learning / fine‑tuning from ImageNet‑pretrained models.  
- **Language**: Python (3.10+ recommended).

#### 3.1 Planned Project Structure

```text
real_or_gen_ai/
  data/                  # Raw and processed datasets
  checkpoints/           # Saved model weights
  src/
    dataset.py           # Dataset and DataLoader definitions
    model.py             # Model construction and loading utilities
    train.py             # Training & validation loop
    inference.py         # Inference utilities / CLI entry
  notebooks/             # EDA and experimental notebooks
  requirements.txt       # Python dependencies
  README.md              # Project documentation (this file)
```


---

### 4. Model Design

- **Baseline Backbone**: `ResNet18` pretrained on ImageNet.  
- **Output Layer**: A fully connected layer with `num_classes = 2` (for `real` vs `ai`).  
- **Why ResNet18 + ImageNet Pretraining?**
  - Kaggle dataset size is moderate; training from scratch is likely to overfit and converge slowly.
  - ImageNet‑pretrained models provide strong generic visual features (edges, textures, shapes), which work well for transfer learning.
  - ResNet18 is lightweight and easy to train on common GPUs (4–8 GB VRAM), suitable for a teaching / course project.

#### 4.1 Model Interface (Planned)

In `src/model.py`:

```python
def build_model(backbone: str = "resnet18", num_classes: int = 2):
    """Construct a classification model with the given backbone."""
    ...
```

This design makes it easy to:

- Start with `resnet18` as the baseline.
- Later switch to other backbones (e.g. `resnet50`, `efficientnet_b0`, `convnext_tiny`, ViT, etc.) for comparison by changing only a configuration parameter.

---

### 5. Training and Evaluation

- **Data Split**:
  - Train / Validation ≈ 80% / 20% (either random split or as provided by the dataset).
- **Loss Function**:
  - `CrossEntropyLoss` for 2‑class classification.
- **Optimizer & Scheduler**:
  - Optimizer: `AdamW` with learning rate around `1e-4 ~ 3e-4` (to be tuned).  
  - Learning rate scheduler: `ReduceLROnPlateau` based on validation loss.  
  - Early stopping: stop training after several epochs without improvement on validation metrics.

#### 5.1 Metrics

Key evaluation metrics:

- **Accuracy**  
- **F1‑score**  
- **Confusion matrix** (to analyze typical misclassifications)  
- (Optional) ROC‑AUC, precision, recall

Training logs and curves (loss/metrics vs. epochs) will be tracked via:

- TensorBoard and/or  
- Weights & Biases (`wandb`) or simple CSV logging.

---

### 6. Inference & Usage

#### 6.1 Command‑Line Inference

Planned CLI entry point in `src/inference.py`:

```bash
python src/inference.py --model-path checkpoints/model-resnet18-v1.pt --image path/to/image.jpg
```

Expected output:

- Predicted label: `ai` or `real`  
- Probabilities: `P(ai)` and `P(real)`

Planned Python function signature:

```python
def predict_image(model_path: str, image_path: str) -> dict:
    """
    Returns a dictionary like:
    {
        "label": "ai" or "real",
        "prob_ai": float,
        "prob_real": float,
    }
    """
```

#### 6.2 (Optional) Web Demo

Using Gradio or Streamlit (to be decided), we plan to provide a simple web interface:

- Upload an image.  
- Show predicted label and confidence scores.  
- Optionally display a few example images and typical failure cases.

---
ai->0, real->1
source .venv/bin/activate
python3 src/train.py --stage1-epochs 3 --stage2-epochs 7 --sample-ratio 0.5
python3 src/train.py --stage1-epochs 3 --stage2-epochs 7