# RUKOPYS Project Tracker

## Goal

Build a working handwritten-to-data pipeline for the Kaggle RUKOPYS task, while keeping each step understandable.

Full target pipeline:

```text
page image
-> detect text/region boxes
-> crop handwritten lines
-> OCR model reads each crop
-> assemble prediction
-> create Kaggle submission
```

## Current Status

### Done

- Created a line-level OCR notebook: `notebook90c3ab57ce.ipynb`.
- Loads `UkrainianCatholicUniversity/rukopys`, config `gt_only`, split `train`.
- Extracts legible `handwritten` regions into OCR samples: `crop -> text`.
- Normalizes punctuation variants for OCR labels.
- Builds alphabet with CTC blank token.
- Implements CTC encode/decode helpers.
- Preprocesses line crops as grayscale, fixed-height images.
- Implements `OCRDataset` and `collate_fn` with width padding.
- Uses page-level train/validation split before extracting line crops.
- Implements CRNN + BiLSTM + CTC loss OCR model.
- Implements CER metric.
- Adds training and validation loops.
- Adds checkpoint save/load helpers.
- Saves model metadata needed for inference:
  - `model_state_dict`
  - `idx2char`
  - `char2idx`
  - `IMAGE_HEIGHT`
  - model config
  - normalization note
  - training history
  - optional optimizer state
- Demo prediction uses `loaded_model`, so checkpoint restore is covered in the notebook flow.

### What This Can Do Now

The project can train and evaluate OCR for already-cropped handwritten line images.

Current supported flow:

```text
handwritten line crop
-> CRNN OCR model
-> predicted text
```

### What This Cannot Do Yet

- It cannot process raw Kaggle test pages end-to-end.
- It cannot find line bounding boxes on new pages.
- It cannot create a Kaggle submission yet.
- It cannot locally score Kaggle test data, because Kaggle test labels are hidden.

## Important Concept

Kaggle test data is not the same as local validation.

- Local validation: use held-out pages from the labeled train split and compute CER.
- Kaggle test: run inference on unlabeled pages, submit predictions, and let Kaggle compute the hidden score.

The OCR model is only one component. For Kaggle test, we still need a detector:

```text
test page
-> detector finds handwritten line boxes
-> OCR reads each crop
-> submission formatter writes output
```

## Next Steps

### Step 1: Run the OCR Notebook

Run `notebook90c3ab57ce.ipynb` in Kaggle or another environment with:

- `datasets`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `torch`
- GPU recommended

Check:

- training loss decreases
- validation CER decreases
- random validation predictions become readable after enough epochs
- `crnn_rukopys.pt` is created

### Step 2: Improve OCR Training

Small improvements before detector work:

- Set `DEBUG = False`.
- Increase `EPOCHS`.
- Save best checkpoint by lowest validation CER, not only final checkpoint.
- Add validation examples after every epoch.
- Consider filtering out non-text-heavy samples if the first model is unstable.

### Step 3: Export Detector Dataset

Create YOLO-style dataset from train page annotations.

Needed output:

```text
detector_data/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```

Initial detector classes can be:

- `handwritten`
- `printed`
- `formula`
- `table`
- `annotation`
- `image`
- `graph`

Minimum useful detector for OCR:

- detect `handwritten` regions only

### Step 4: Train Page Detector

Train a detector such as YOLO on the exported dataset.

Validation should check:

- boxes roughly cover handwritten lines
- false positives are manageable
- reading order can be recovered by sorting boxes

### Step 5: Page-Level Inference

Build an inference function:

```text
input: full page image
output: list of predicted handwritten lines
```

Pseudo-flow:

```python
boxes = detector.predict(page)
boxes = sort_boxes_reading_order(boxes)

texts = []
for box in boxes:
    crop = page.crop(box)
    text = predict_text(ocr_model, crop)
    texts.append(text)
```

### Step 6: Kaggle Submission

Before implementing this, inspect Kaggle `sample_submission.csv`.

Need to determine:

- required columns
- whether predictions are page-level text, region-level text, or structured JSON
- ordering expectations
- escaping/newline rules

Then create:

```text
submission.csv
```

and submit to Kaggle for the hidden test score.

## Open Questions

- What exact format does Kaggle require for `sample_submission.csv`?
- Should the first detector handle only `handwritten`, or all region types?
- Should OCR preserve all symbols, or should it be limited to Ukrainian text plus common punctuation?
- Should we use only `gt_only` or add silver-labeled data after the baseline works?

## Recommended Next Implementation Task

Add best-checkpoint saving by validation CER in the OCR notebook.

Reason: before building the detector, it is useful to make OCR training repeatable and avoid losing the best epoch.
