# SEAL
 Structured Encoder for Ancient Logograms

## Requirements

```
pip install -r requirements.txt
```

## Launch

To train or evaluate the model:

```
cd model
python launch.py [options...]
```

Check `launch.py` or call it with `--help` to inspect the available options.


### Checkpoint

The model checkpoint must be placed under the `model` folder, next to `launch.py`

## Dataset

The model was trained on [SEAL 5864](https://www.kaggle.com/datasets/emanuelemessina/seal-5684/data).

This repo also contains the dataset generation code for "SEAL _N_".

The dataset (`test` and `train` folders, `image_stats.txt`) must be placed inside `dataset/output`.