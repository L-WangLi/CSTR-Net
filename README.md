# CSTR-Net: Multi-Scale Temporal Refinement with Cross-Sensor Interaction for Turbofan Engine RUL Prediction

CSTR-Net predicts turbofan engine remaining useful life through multi-scale gated temporal convolutions and in-branch cross-sensor attention. This repository contains the final model, data preprocessing, training, and evaluation code for C-MAPSS and N-CMAPSS.
（The manuscript is currently under review. If you find this work useful, please consider citing it using the citation information below or the Cite this repository button.）

## Files

`models/cstr_net.py` defines CSTR-Net 
`models/layers.py` contains its temporal and prediction layers
`models/checkpoint.py` loads trained checkpoints.

`dataset/` contains the C-MAPSS preprocessor, the N-CMAPSS phase-token preprocessor, and the DS02 downsampling preprocessor. 
`training/` contains model construction and training loops. 
`configs/` stores the final dataset and training settings.

`train.py` trains on C-MAPSS, `run_ncmapss.py` trains on N-CMAPSS, and `evaluate.py` evaluates a trained checkpoint. 
Dependencies are listed in `requirements.txt`.

## Data preparation

Download [C-MAPSS](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip) and [N-CMAPSS](https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip) from the [NASA repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

Place C-MAPSS `train_FD00*.txt`, `test_FD00*.txt`, and `RUL_FD00*.txt` in `dataset/raw/cmapss/`. Place the N-CMAPSS `.h5` files in `dataset/raw/ncmapss/`.

Use Python 3.10 and run commands from the repository root:

```bash
pip install -r requirements.txt

python -m dataset.cmapss
python -m dataset.ncmapss
python -m dataset.ncmapss_downsample
```

C-MAPSS preprocessing reads the subset settings from `configs/cmapss/`. N-CMAPSS uses phase-token windows of length 60 for eight subsets and downsampled windows of length 50 for DS02. Processed arrays and their metadata are saved under `dataset/processed/`.

## Training and evaluation

```bash
python train.py --dataset FD001 --seed 7
python run_ncmapss.py --datasets DS01 DS02 --seeds 7
```

Configurations are loaded automatically. Use `--data-dir` to reuse processed data, `--device cpu` to run on CPU, or `--dry-run` to inspect settings. Training writes checkpoints and metrics to `runs/`.

Evaluate a checkpoint produced by training:

```bash
python evaluate.py --checkpoint /path/to/best_checkpoint.pt --data-dir /path/to/FD001
```

Here `--data-dir` points to the subset folder containing `X_test.npy` and `y_test.npy`. Evaluation outputs are saved to `outputs/`.

## Citation

The manuscript describing CSTR-Net is currently under review.
If you find this work useful, please consider citing:

```bibtex
@unpublished{wang2026cstrnet,
  author = {Wang, Li and Chaw, Jun Kit and Ang, Mei Choo and
            Wang, Hongqing and Mahmoud, Moamin A. and
            Badioze Zaman, Halimah and
            Gunasekaran, Saraswathy Shamini and Cheng, Xiang},
  title  = {{CSTR-Net}: Multi-Scale Temporal Refinement with
            Cross-Sensor Interaction for Turbofan Engine {RUL} Prediction},
  year   = {2026},
  note   = {Manuscript under review}
}
```

The citation will be updated when a public preprint or the published
article becomes available.
