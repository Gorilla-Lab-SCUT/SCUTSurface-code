# Evaluation Metrics

## Vanilla Metrics

Including Chamfer-L2 Distance, F-score and Normal Consistency Score.

Run the script:
```
    cd vanilla_metric
    python eval_two_folder.py --eval_type syn_obj --in_dir DIR_to_THE_INPUT --gt_dir DIR_to_GROUND_TRUTH --samplepoints 200000 --out_csv test.csv
```

The results could be recorded in `test.csv`



## Neural Metrics

Use the pretrained model to evaluate
```
    cd vanilla_metric
    python eval_two_folder.py --eval_type syn_obj --in_dir DIR_to_THE_INPUT --gt_dir DIR_to_GROUND_TRUTH --model_dir DIR_to_MODEL --out_csv test.csv
```

Train the network

```
    cd vanilla_metric
    python Train.py --name MODEL_NAME_to_SAVE
```

The results could be recorded in `test.csv`
