# CATRAN (CAusal TRANsformer)
This is a repository with my solution to the [CausalBench challenge](https://www.gsk.ai/causalbench-challenge/).

In short the idea of this method is to learn the relation between genes (the gene-regulatory network) as attention weights which are used for downstream tasks of predicting gene expression.

The solution itself can be found in ```catran.py``` file. To test it, one should first install the requirements of this repository with:

```pip install -r requirements.txt```. 

Then download [causalbench-starter repository](https://github.com/causalbench/causalbench-starter), copy ```catran.py``` to ```causalbench-starter/src``` and run:

```
causalbench_run \
    --dataset_name weissmann_rpe1 \
    --output_directory /path/to/output/ \
    --data_directory /path/to/data/storage \
    --training_regime partial_interventional \
    --partial_intervention_seed 0 \
    --fraction_partial_intervention $FRACTION \
    --model_name catran \
    --inference_function_file_path /path/to/custom_inference_function.py \
    --subset_data 1.0 \
    --model_seed 0 \
    --do_filter
```
