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

## Dataset

The model was trained on [SEAL 5864](https://www.kaggle.com/datasets/emanuelemessina/seal-5684/data).

This repo also contains the dataset generation code for "SEAL _N_".

### Credits

- Fonts from:
  - freefonts.top
  - fonts.net.cn
  - font.chinaz.com
  - diyiziti.com

- Kangxi dictionary: [github.com/7468696e6b/kangxiDictText]()
- Classical chinese frequency list: [lingua.mtsu.edu]()

- Background images from Pinterest