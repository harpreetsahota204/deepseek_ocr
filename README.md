# DeepSeek-OCR FiftyOne Zoo Model

DeepSeek-OCR is a vision-language model designed for optical character recognition with a focus on "contextual optical compression." Unlike traditional OCR engines, it uses a dual-encoder architecture (SAM + CLIP) to process documents and convert them to structured text formats like Markdown.

![image](deepseekocr_fo.gif)

**Key Features:**
- Supports multiple resolution modes for different document types
- Can process documents with complex layouts, tables, and formulas
- Outputs structured Markdown with bounding box annotations
- Handles multi-page PDFs and various image formats

---

## Requirements

**Important:** This model requires specific versions of transformers and tokenizers:

```bash
pip install transformers==4.46.3
pip install tokenizers==0.20.3
pip install addict
pip install fiftyone
pip install torch
pip install torchvision
```

**Optional (for GPU acceleration):**
```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

---

## Installation and Setup

### Register the Model Source

```python
import fiftyone.zoo as foz

# Register the model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/deepseek_ocr",
    overwrite=True  # This will make sure you're always using the latest implementation
)
```

### Load the Model

```python
# Load the model
model = foz.load_zoo_model("deepseek-ai/DeepSeek-OCR")
```

---

## Usage Examples

### Load a Dataset

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the dataset
# Note: other available arguments include 'max_samples', etc
dataset = load_from_hub("Voxel51/document-haystack-10pages")
```

### Grounding Mode - Extract Text with Bounding Boxes

```python
# Grounding Mode - Extract text with bounding boxes
model.resolution_mode = "gundam"
model.operation = "grounding"

dataset.apply_model(model, label_field="text_detections")
```

### Free OCR - Text Extraction Only

```python
# Free OCR
model.operation = "ocr"
dataset.apply_model(model, label_field="text_extraction")
```

### Describe Mode - Document Description

```python
# Describe mode
model.operation = "describe"
dataset.apply_model(model, label_field="doc_description")
```

### Custom Prompt

```python
# Custom prompt
model.operation = "grounding"
model.prompt = "<image>\n<|grounding|>Locate <|ref|>The secret<|/ref|> in the image."
dataset.apply_model(model, label_field="custom_detections")
```

---

## Resolution Modes

DeepSeek-OCR provides five predefined resolution modes optimized for different document types:

### Single-View Modes (`crop_mode=False`)

These modes process the entire image as **one single view** at a fixed resolution:

| Mode | `base_size` | `image_size` | `crop_mode` | Vision Tokens | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| **Tiny** | 512 | 512 | False | 64 | Fastest, for very simple documents |
| **Small** | 640 | 640 | False | 100 | Fast, for simple receipts/forms |
| **Base** | 1024 | 1024 | False | 256 | Balanced, for standard documents |
| **Large** | 1280 | 1280 | False | 400 | Highest quality, slower |

**How it works:**
```
Your image (any size) → Resized/padded to [N×N] → Single view → Fixed token count
```

### Multi-View Mode: Gundam (`crop_mode=True`)

**Gundam mode** is the **recommended default** for complex documents. It processes images using **two complementary views**:

| Mode | `base_size` | `image_size` | `crop_mode` | Vision Tokens | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| **Gundam** | 1024 | 640 | True | Variable | Multi-view for complex layouts |

**How it works:**
```
Your image (any size) → [1024×1024 global view]  (overall structure)
                      + [640×640 patches × N]     (fine details)
                      → 256 + (N × 100) tokens
```

The model automatically determines how many 640×640 patches are needed based on your image dimensions.

**Visual Example:**

For a 2400×3200 pixel image with Gundam mode:
```
Global View:           Local Patches:
┌──────────┐          ┌────┬────┬────┬────┐
│          │          │ 1  │ 2  │ 3  │ 4  │
│ 1024×1024│    +     ├────┼────┼────┼────┤
│          │          │ 5  │ 6  │ 7  │ 8  │
│          │          ├────┼────┼────┼────┤
└──────────┘          │ 9  │ 10 │ 11 │ 12 │
                      └────┴────┴────┴────┘
                      (each patch is 640×640)
```

---

## Key Parameters

### `resolution_mode`

Controls the processing resolution and strategy. Options: `"gundam"` (default), `"base"`, `"small"`, `"large"`, `"tiny"`.

### `operation`

Determines the task type and output format:
- `"grounding"` - Returns `fo.Detections` with bounding boxes
- `"ocr"` - Returns text string
- `"describe"` - Returns description text string

### Custom Prompts

You can create custom prompts to guide the model toward specific extraction tasks. The model automatically infers the output type based on the prompt content.

**Guidelines:**
- Always include the `<image>` placeholder
- Include `<|grounding|>` for detection output with bounding boxes
- Omit `<|grounding|>` for text-only output

**Examples:**

```python
# Grounding with bounding boxes - returns fo.Detections
model.prompt = "<image>\n<|grounding|>Extract all table content."
model.prompt = "<image>\n<|grounding|>Find all headers and section titles."
model.prompt = "<image>\n<|grounding|>Locate all monetary amounts."

# Text-only output - returns string
model.prompt = "<image>\nExtract only phone numbers and email addresses."
model.prompt = "<image>\nSummarize the main points in bullet format."
model.prompt = "<image>\nTranslate the document text to Spanish."
```

When using custom prompts, the model automatically determines the output format based on whether `<|grounding|>` is present.

---

## Best Practices and Recommendations

### Choosing the Right Resolution Mode

| Document Type | Recommended Settings | Rationale |
|:---|:---|:---|
| **Complex PDFs, academic papers** | Gundam (1024/640/True) | Captures both structure and details |
| **Multi-column layouts** | Gundam (1024/640/True) | Handles complex spatial relationships |
| **Tables and forms** | Gundam (1024/640/True) | Preserves table structure |
| **Standard single-page docs** | Base (1024/1024/False) | Balanced speed and quality |
| **Simple receipts** | Small (640/640/False) | Fast processing |
| **Quick testing/preview** | Tiny (512/512/False) | Fastest option |
| **High-res scans** | Large (1280/1280/False) | Maximum quality |

---

## Visualization

For viewing extracted text and captions, install the caption viewer plugin:

```bash
fiftyone plugins download https://github.com/mythrandire/caption-viewer
```

---

## Additional Resources

- **Official Repository:** https://github.com/deepseek-ai/DeepSeek-OCR
- **Model Card:** https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **Paper:** DeepSeek-OCR: Contexts Optical Compression (arXiv, 2025)

---

## Citation

```bibtex
@article{wei2024deepseek-ocr,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```
