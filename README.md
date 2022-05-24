## TRASA
The code for TRASA: *Transition Relation Aware Self-Attention for Session-based Recommendation*

## Dataset Preprocess
- The datasets and code for data preprocessing can refer to [LESSR](https://github.com/twchen/lessr).
- After preprocessing the data, place it to the floder `data/`

## Train Model
Use the following commands to run the code.
```shell
# python main.py --dataset-dir data/<dataset_name> --batch-size <xxx>
python main.py --dataset-dir data/sample --batch-size 32
```

## Acknowledgements
[GTOS](https://github.com/jcyk/gtos) and [LESSR](https://github.com/twchen/lessr)