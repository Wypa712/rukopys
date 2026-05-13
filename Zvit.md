## 2026-05-13

- First implementation step for the RUKOPYS notebook: local validation now splits pages first, then extracts OCR line crops separately for train and validation.
- Reason: the previous row-level split could leak lines from the same page/handwriting style into both train and validation.
- Scope intentionally stopped here: no detector, no Kaggle submission generation yet. Next step should be either model checkpoint saving/loading or page-level detector data export.
- Second implementation step: added OCR checkpoint save/load helpers to the notebook.
- The checkpoint stores `model_state_dict`, `idx2char`, `char2idx`, `IMAGE_HEIGHT`, model config, normalization note, training history, and optional optimizer state.
- The random validation prediction now uses `loaded_model`, so the notebook demonstrates inference from a saved checkpoint.
- Added `RUKOPYS_TRACKER.md` as the project tracker/roadmap with current status, limitations, and next tasks.
- Reviewed current Kaggle/RUKOPYS state from tracker and notebook. The project currently has a line-level CRNN+BiLSTM+CTC OCR baseline for already-cropped `handwritten` regions, with page-level train/validation split, CER, checkpoint save/load, and demo inference from the loaded checkpoint.
- Main remaining gap for Kaggle is the full page pipeline: inspect Kaggle `sample_submission.csv`, export region annotations for detector training, train/evaluate a page detector, sort detected handwritten boxes in reading order, run OCR over crops, and generate `submission.csv`.
- Immediate useful notebook improvements before detector work: set `DEBUG = False`, increase `EPOCHS`, save best checkpoint by validation CER, and keep validation prediction examples after each epoch.
