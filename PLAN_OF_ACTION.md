# Plan of action

Author: Abhay Kumar, 23115002

1. Freeze the data contract. Use Oxford-IIIT Pet human trimaps, binary background/pet labels, ignored boundary pixels, and breed-balanced manifests. Take 40 training and 10 validation examples per breed from official trainval, and 20 per breed from official test. Save source hashes and split IDs before training.
2. Fix the baseline. Train a compact encoder–decoder from scratch at 96 × 96 using seed 23115002, paired horizontal flips, Adam at 0.001, batch 16, 20 epochs, CPU and four threads. Select the best validation global mIoU, breaking ties by earliest epoch.
3. Make one controlled change. Enable additive encoder-to-decoder skip fusion at two matching resolutions. Preserve parameter shapes, initialization, sample order, augmentation draws, optimizer and update budget. Do not tune on test results.
4. Evaluate both saved checkpoints on exactly the same untouched 740-image official-test subset. Aggregate the two-class confusion matrix over all valid pixels; report per-class IoU and arithmetic mean IoU. Record parameters and CPU forward-only latency with batch 1, 30 warmups and 200 timed iterations.
5. Produce evidence. Save raw epoch logs, checkpoints, metric JSON, per-image scores, loss and validation plots, original images next to human masks and predictions, representative failures, architecture and data-contract figures. Capture actual rendered experiment evidence pages as screenshots.
6. Write the report. Cover introduction, background study, detailed methodology and block parameters, results with interpretation, limitations, conclusion, references and exact reproduction instructions. Name the report 23115002.
7. Verify and package. Re-evaluate checkpoints, test metric edge cases and split integrity, write the final report in your own words as required by the assignment, inspect all pages, and package source, environment versions, logs and evidence. Publish to GitHub only after the user provides the intended repository/account; do not send email without an explicit request.

## Expected submission

- `report/23115002.docx` and a PDF copy
- `src/`, pinned requirements, reproducible download and execution commands
- `artifacts/` with split manifest, environment, training histories, checkpoint files, measured metrics and figures
- Screenshot evidence included in the report
- GitHub-ready project and submission instructions
