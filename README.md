## SCALA: Towards Imperceptible and Efficient Black-box Textual Adversarial Perturbations - Transactions on Information Forensics & Security (TIFS)

### Efficient Word-level Black-box Adversarial Textual Generation Based on Hamming Distance

#### Instructions for running the attack from this repository.

#### Requirements
-  Numpy == 1.19.5  
-  Pytorch == 1.11.0    
-  Python >= 3.6
-  Tensorflow == 1.15.2
-  TensorflowHub == 0.11.0 
-  textattack == 0.3.3

#### Download Dependencies

- Download pretrained target models for each dataset [bert](XXXX), [lstm](XXXX), [cnn](XXXX) unzip it.

- Download the counter-fitted-vectors from [here](XXXX) and place it in the main directory.

- Download top 50 synonym file from [here](XXXX) and place it in the main directory.

- Download the glove 200 dimensional vectors from [here](XXXX) unzip it.

#### Training target models

To train BERT on a particular dataset use the commands provided in the `.\BERT\` directory. 

For training LSTM and CNN models run the `train_classifier.py --<model_name> --<dataset>`.

For finetuning Llama-3.2-1b and Llama-3.2-3b, run `finetune_binary_classfication.py` and `finetune_multi_classification.py` for various datasets in the `.\llama\` directory.
 
#### How to Run:

After training or finetuning the models, use the following command to get the attack results. 

For BERT model

```
python classification_attack.py \
        --target_model Type_of_taget_model (bert,cnn,lstm) \
        --target_dataset Dataset Name (mr, imdb, yelp, ag, snli, mnli)\
        --target_model_path pre-trained target model path \
        --dataset_dir directory_to_data_samples_to_attack  \
        --output_dir  directory_to_save_results \
        --word_embeddings_path path_to_embeddings \
        --counter_fitting_cos_sim_path path_to_synonym_file \
        --nclasses  how many classes for classification


```
Example of attacking BERT on IMDB dataset.

```

python3 classification_attack.py \
        --target_model bert \
        --target_dataset imdb \
        --target_model_path pretrained_models/bert/imdb \
        --dataset_dir data/ \
        --output_dir  final_results/ \
        --word_embeddings_path embedding/glove.6B.200d.txt \
        --counter_fitting_cos_sim_path counter-fitted-vectors.txt \
        --nclasses 2 \


```

Example of attacking BERT on SNLI dataset. 

```

python3 entailment.py \
        --target_model bert \
        --target_dataset snli \
        --target_model_path ../pretrained_models/bert/snli \
        --dataset_dir ../data/ \
        --output_dir  ../final_results/ \
        --word_embeddings_path ../embedding/glove.6B.200d.txt \
        --counter_fitting_cos_sim_path ../counter-fitted-vectors.txt \


```

Example of attacking Llama-3.2-1b on MR dataset.

```
python llm.py \
    --target_model llama-3.2-1b \
    --target_dataset mr \
    --target_model_path ./models/llama/mr/ \
    --dataset_dir ./data/mr.txt \
    --output_dir ./outputs/ \
    --word_embeddings_path ./embedding/glove.6B.200d.txt \
    --counter_fitting_embeddings_path ./counter_fitted/counter-fitted-vectors.txt \
    --counter_fitting_cos_sim_path ./counter_fitted/mat.txt \
    --USE_cache_path ./embedding/use \
    --theta 1 \
    --nclasses 2
```


#### Results
The results will be available in **final_results/classification/** directory for classification task and in **final_results/entailment/** for entailment tasks.
For attacking other target models look at the ```commands``` folder.

