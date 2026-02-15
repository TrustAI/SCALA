import torch 
import criteria
from collections import defaultdict
import argparse
import dataloader
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
import numpy as np
import pickle
from pathlib import Path
import csv
import sys
import random
csv.field_size_limit(sys.maxsize)
np.random.seed(1234)
random.seed(0)
import csv
import logging
import torch
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import os
import torch
import torch.nn.functional as F

class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        
        module_url = "XXXX"
        self.embed = hub.KerasLayer(module_url, trainable=False)
        
        # Build graph first (before session)
        self.build_graph()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        # Initialize session after graph is completely defined
        with self.sess.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.sess.run(tf.compat.v1.tables_initializer())

    def build_graph(self):
        # Define placeholders and operations BEFORE running the session
        self.sts_input1 = tf.compat.v1.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.compat.v1.placeholder(tf.string, shape=(None))

        # Normalize embeddings
        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)

        # Cosine similarity calculation
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            self.sim_scores,
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores



def get_token_log_likelihoods(
    text: str,
    model,
    tokenizer,
    return_tokens: bool = True,
    return_mean: bool = False
):
    """
    Compute token-level log-likelihoods of a given sentence using a causal LM.
    
    Args:
        text (str): The input sentence to evaluate.
        model (AutoModelForCausalLM): A Hugging Face decoder-only model (e.g., GPT-2, LLaMA).
        tokenizer (AutoTokenizer): Corresponding tokenizer for the model.
        return_tokens (bool): If True, also return the tokens.
        return_mean (bool): If True, also return the average log-likelihood over the sentence.

    Returns:
        list of float: Log-likelihood per token.
        (optional) list of str: Corresponding tokens.
        (optional) float: Mean log-likelihood.
    """
    model.eval()
    device = model.device

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Compute logits
    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compute log-probabilities over vocab
    log_probs = F.log_softmax(logits, dim=-1)

    # Align targets (predict token[i+1] given token[:i])
    token_log_probs = log_probs[0, :-1, :]
    target_token_ids = input_ids[0, 1:]
    log_likelihoods = token_log_probs.gather(1, target_token_ids.unsqueeze(1)).squeeze(1)

    result = [l.item() for l in log_likelihoods]

    outputs = (result,)
    if return_tokens:
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])[1:]  # Skip first token (no prediction)
        outputs += (tokens,)
    if return_mean:
        avg = sum(result) / len(result)
        outputs += (avg,)

    return outputs if len(outputs) > 1 else outputs[0]


def get_last_token_embedding(
    outputs,
    model,
    tokenizer,
    input_ids,
    n_input_token: int,
    n_generated: int,
    token_stop_index: int,
    full_answer: str,
    sliced_answer: str = "",
    token_layer: int = -1  # -1 = last layer
):
    """
    Extract the embedding of the last generated token from a model output.

    Args:
        outputs: Model outputs from generate() with output_hidden_states=True.
        model: The model (used to infer decoder vs decoder-only).
        tokenizer: Corresponding tokenizer.
        input_ids (torch.Tensor): Tokenized input prompt (1D tensor).
        n_input_token (int): Length of input prompt.
        n_generated (int): Number of generated tokens.
        token_stop_index (int): Index at which the generation stopped.
        full_answer (str): Full generated text (for logging/debug).
        sliced_answer (str): Optionally sliced/generated part only.
        token_layer (int): Which layer's output to use (default: last).

    Returns:
        last_token_embedding (torch.Tensor): Shape (batch_size=1, hidden_size)
    """

    # Get hidden states (decoder vs decoder-only)
    if hasattr(outputs, "decoder_hidden_states"):
        hidden = outputs.decoder_hidden_states
    else:
        hidden = outputs.hidden_states

    # Handle weird edge cases where hidden has only one item
    if len(hidden) == 1:
        logging.warning(
            'Taking first and only generation for hidden! '
            'n_generated: %d, n_input_token: %d, token_stop_index %d, '
            'last_token: %s, generation was: %s',
            n_generated, n_input_token, token_stop_index,
            tokenizer.decode(outputs.sequences[0][-1]),
            full_answer
        )
        last_input = hidden[0]

    elif (n_generated - 1) >= len(hidden):
        # More tokens than hidden states — use last available
        logging.error(
            'Taking last state because n_generated is too large. '
            'n_generated: %d, n_input_token: %d, token_stop_index %d, '
            'last_token: %s, generation was: %s, slice_answer: %s',
            n_generated, n_input_token, token_stop_index,
            tokenizer.decode(outputs.sequences[0][-1]),
            full_answer, sliced_answer
        )
        last_input = hidden[-1]

    else:
        # Normal case: take the (n_generated - 1)th hidden state
        last_input = hidden[n_generated - 1]

    # Pick the desired layer's output
    middle_layer_embedding = last_input[token_layer]  # shape: (batch, seq_len, hidden_size)

    # Take the last token embedding
    last_token_embedding = middle_layer_embedding[:, -1, :].cpu()  # shape: (1, hidden_size)

    return last_token_embedding



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.
    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=64):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data, batch_size=32):
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
        return eval_dataloader


import torch

def llama_get_label_and_confidence(text, labels, tokenizer, model):
    # # Create the prompt with the text and labels
    input = " ".join(text)

    # Tokenize the input
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move inputs to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Move model to device
    model = model.to(device)

    # Perform inference without gradient calculation
    with torch.no_grad():
        output = model(**inputs)

    # Extract logits (batch_size, num_labels)
    logits = output.logits  # Shape: (batch_size, num_labels)

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Since you're classifying a single text, squeeze the batch dimension
    probs = probs.squeeze(0)  # Shape: (num_labels)

    # Map probabilities to provided labels
    label_probs = {label: probs[i].item() for i, label in enumerate(labels)}

    # Find the label with the highest probability
    best_label = max(label_probs, key=label_probs.get)
    confidence = label_probs[best_label]

    return best_label, confidence, probs

def llama_get_confidence_score(text, labels, tokenizer, model, target_label):
    # # Create the prompt with the text and labels
    input = " ".join(text)

    # Tokenize the input
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move inputs to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Move model to device
    model = model.to(device)

    # Perform inference without gradient calculation
    with torch.no_grad():
        output = model(**inputs)

    # Extract logits (batch_size, num_labels)
    logits = output.logits  # Shape: (batch_size, num_labels)

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Since you're classifying a single text, squeeze the batch dimension
    probs = probs.squeeze(0)  # Shape: (num_labels)

    # Map probabilities to provided labels
    label_probs = {label: probs[i].item() for i, label in enumerate(labels)}
    if target_label is not None:
        if target_label in label_probs:
            # Return the probability of the specific label
            label_probs[target_label]
        else:
            raise ValueError(f"Target label '{target_label}' not found in provided labels")
    # If no target label is specified, return the best one
    best_label = max(label_probs, key=label_probs.get)
    confidence = label_probs[target_label]
    return best_label, confidence

def get_attack_result(new_text, labels, orig_label, tokenizer, model):
    # Get the predicted label and confidence from the model
    best_label, confidence, _ = llama_get_label_and_confidence(new_text, labels, tokenizer, model)
    
    # Compare the predicted label with the original label
    pr = (orig_label != best_label)
    
    return pr
    
def get_attack_prob(new_text, predictor, batch_size):  
    new_probs = predictor(new_text, batch_size=batch_size)
    new_prob = new_probs.max().data.cpu().numpy()
    return new_prob

def attack(text_ls, predictor, model, tokenizer, label_ls, true_label, word2idx, idx2word, cos_sim, use, sim_threshold, top_k_words, batch_size, synonym_num, theta):
    # get label and confidence score
    orig_label, orig_prob, orig_probs = llama_get_label_and_confidence(text_ls, label_ls, tokenizer, model)
    true_label = int(true_label)
    orig_label = int(orig_label)
    if true_label != orig_label:
        print('Original classifier fail')
        return '', 0, 0, orig_label, orig_label, 0, 0
    else:
        # step-1:pos filter 
        pos_ls = criteria.get_pos(text_ls)
        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos and len(text_ls[i]) > 2:
                    words_perturb.append((i, text_ls[i]))

        # find synonyms and make a dict of synonyms of each word.
        words_perturb = words_perturb[:top_k_words]
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, synonym_values=[],[]
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp=[]
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            temp=[]
            for ii in res[0]:
                temp.append(ii)
            synonym_values.append(temp)
        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms

       # STEP 2: Find a reasonable sort 
        orig_qrs = 0
        flag = 0
        temp = 0
        n = 0
        new_texts = [None] * int(theta * synonym_num) * len(synonyms_all) 
        orig_probs = [[orig_prob] * int(theta * synonym_num)] * len(synonyms_all) 
        

        for i in range(len(synonyms_all)):
            idx = synonyms_all[i][0]
            syn = synonyms_all[i][1] 
            k = 0 
            syn_index = np.random.permutation(synonym_num)[:int(synonym_num*theta)]
            for j in range(int(synonym_num*theta)): 
                new_text = text_ls[:]  
                new_text[idx] = syn[syn_index[j]]
                new_texts[n] = new_text
                n += 1   
                k += 1       
                orig_qrs += 0       
                
        # new_probs = predictor(new_texts, batch_size).squeeze()
        new_probs = []
        for new_text in new_texts:
            # print(llama_get_confidence_score(new_text, label_ls, tokenizer, model, target_label=str_label))
            _, new_prob = llama_get_confidence_score(new_text, label_ls, tokenizer, model, target_label=orig_label)
            new_probs.append(new_prob)
        orig_prob = torch.tensor(orig_prob)
        new_probs = torch.tensor(new_probs)
        saliency_scores = torch.sub(orig_prob.detach().clone(), new_probs)

        # Reshape only if size matches
        
        if saliency_scores.numel() == len(synonyms_all) * int(theta * synonym_num):
            saliency_scores = saliency_scores.reshape([len(synonyms_all), int(theta * synonym_num)])
        # Use correct dim based on shape
        if len(saliency_scores.shape) == 2:
            saliency_scores_values = torch.max(saliency_scores, dim=1)[0]
            saliency_scores_indices = torch.argmax(saliency_scores, dim=1)
        else:
            saliency_scores_values = torch.max(saliency_scores, dim=0)[0]
            saliency_scores_indices = torch.argmax(saliency_scores, dim=0)

        # If saliency_scores_indices is a scalar, convert it to a list
        if saliency_scores_indices.dim() == 0:
            saliency_scores_indices = [saliency_scores_indices.item()]  # Convert scalar to list

        # Sort the scores
        argsort_saliency_scores = torch.argsort(saliency_scores_values, descending=True).tolist()

        # Only sort if indices are not empty
        if len(saliency_scores_indices) > 1:
            saliency_scores_indices = [saliency_scores_indices[i] for i in argsort_saliency_scores]


        # step 3: replace with synonyms based on the sort
        orig_changed = 0  
        current_text = text_ls[:]
        replace_indices = []
        
        for i in range(len(argsort_saliency_scores)):

            current_probs = []
            semantic_sims = []

            idx = synonyms_all[argsort_saliency_scores[i]][0]
            syn = synonyms_all[argsort_saliency_scores[i]][1]
            current_texts = [None]* synonym_num
            n = 0
            for j in range(len(syn)):
                current_text[idx] = syn[j]  
                current_texts[n] = current_text[:]
                n+=1
            for current_text in current_texts:
                _, current_prob = llama_get_confidence_score(current_text, label_ls, tokenizer, model, target_label=orig_label)
                current_probs.append(current_prob)
                semantic_sim = use.semantic_sim([' '.join(text_ls)], [' '.join(current_text)])[0]
                semantic_sims.append(semantic_sim)

 
            if isinstance(current_probs, list):
                current_probs_tensor = torch.tensor(current_probs, device=orig_prob.device)
            current_saliency_scores = torch.sub(orig_prob.detach().clone(), current_probs_tensor)

            # Apply the constraint: semantic similarity must be greater than the threshold
            sorted_indices = torch.argsort(current_saliency_scores, descending=True).tolist()
            selected_index = None
            for idx_candidate in sorted_indices:
                if semantic_sims[idx_candidate] > sim_threshold:
                    selected_index = idx_candidate
                    break
            if selected_index is not None:
                current_text[idx] = syn[selected_index]
                replace_indices.append(idx)
                pr = get_attack_result(current_text, label_ls, orig_label, tokenizer, model)
                orig_qrs += 1
                orig_changed += 1
                if np.sum(pr)>0:
                    flag = 1
                    print('original changed',orig_changed)
                    break
            else:
                print(f"No suitable synonym found for position {idx} due to semantic similarity threshold.")
        
        # step 4: replace back based on sort
        one_word_probs = []
        if flag == 1:
            new_label, confidence, new_probs = llama_get_label_and_confidence(current_text, label_ls, tokenizer, model)
            if orig_changed >= 2:
                replace_back_qrs = 0
                replace_back = 0
                one_word_texts = [None] * orig_changed
                m = 0
                j = 0  
                # new_probs = predictor([current_text], batch_size).squeeze()
                new_label = torch.argmax(new_probs)
                new_prob = confidence
                for index in range(len(replace_indices)):
                    one_word_text = current_text[:]
                    one_word_text[index] = text_ls[index]
                    one_word_texts[m] = one_word_text[:]
                    m += 1
                    replace_back_qrs += 1
                replace_back_indices = replace_indices[:]
                # one_word_probs = predictor(one_word_texts, batch_size).squeeze()
                for one_word_text in one_word_texts:
                    _, one_word_prob = llama_get_confidence_score(one_word_text, label_ls, tokenizer, model, target_label=int(new_label))
                    one_word_probs.append(one_word_prob)
                one_word_probs = torch.tensor(one_word_probs)
                new_prob = torch.tensor(new_prob)
                saliency_scores_back = torch.sub(new_prob.detach().clone(), one_word_probs)
                argsort_saliency_scores_back = torch.argsort(saliency_scores_back, descending=False)
                replace_back_indices = [replace_back_indices[i] for i in argsort_saliency_scores_back.tolist()]
                adv_text = current_text[:]
                for i in range(len(replace_back_indices)):
                    adv_text[replace_back_indices[i]] = text_ls[replace_back_indices[i]]
                    # pr = get_attack_result([adv_text], predictor, orig_label, batch_size)
                    pr = get_attack_result(adv_text, label_ls, orig_label, tokenizer, model)
                    replace_back_qrs += 1
                    replace_back += 1
                    if np.sum(pr) == 0:
                        adv_text[replace_back_indices[i]] = current_text[replace_back_indices[i]]
                        replace_back -= 1
            else:
                replace_back = 0
                adv_text = current_text[:]
                replace_back_qrs = 0
            final_changed = orig_changed - replace_back
            qrs = orig_qrs + replace_back_qrs
            final_semantic_sim = use.semantic_sim([' '.join(text_ls)], [' '.join(adv_text)])[0]
            print('final_semantic_sim', final_semantic_sim)
            print('attack success')
            return ' '.join(adv_text), final_changed, orig_changed, \
                  orig_label, new_label, qrs, final_semantic_sim
        else:
            print("attack fail")
            return '', 0, 0, orig_label, orig_label, 0, 0

        
def main():
    
    # create parser
    parser = argparse.ArgumentParser()

    ###### add parameters
    parser.add_argument("--target_model",
                        default="llama-3.2-1b",
                        type=str,
                        required=True,
                        choices=['llama-3.2-1b','llama-3.2-3b'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--target_dataset",
                        default="mr",
                        type=str,
                        required=True,
                        help="Dataset Name: mr, imdb, yelp, ag, snli, mnli, MR-GSM8K")
    parser.add_argument("--target_model_path",
                        type=str,
                        default="../pretrained_models/bert/mr",
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--dataset_dir",
                        default="../data/",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--output_dir",
                        default="../final_results/classification/",
                        type=str,
                        required=True,
                        help="Which directory to save results.")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='../embedding/glove.6B.200d.txt',
                        required=True,
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        default="/counter-fitted-vectors.txt",
                        required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='../mat.txt',
                        required=True,
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")

    parser.add_argument("--USE_cache_path",
                        type=str,
                        default='./embedding/use',
                        required=True,
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    ## Model hyperparameters
    parser.add_argument("--data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--theta",
                        default=0.5,
                        type=float,
                        help="A parameter to control the number of synonyms to extract")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--batch_size",
                        default=64,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        # required=True,
                        help="How many classes for classification.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")
    parser.add_argument("--top_k_words",
                        default=1000000,
                        type=int,
                        help="Top K Words")
    parser.add_argument("--compute_predictive_entropy",
                        default=False,
                        type=int,
                        help="compute_predictive_entropy")
    


    # parser paremeters
    args = parser.parse_args()


    # get data to attack
    # texts, labels = dataloader.read_corpus(args.dataset_dir+args.target_dataset,csvf=False)
    texts, labels = dataloader.read_corpus(args.dataset_dir,csvf=False)
    data = list(zip(texts, labels))
    data = data[:args.data_size] 
    print("Data import finished!")

    if args.target_dataset in ['mr', 'boolq', 'mr-gsm8k_answer', 'mr-gsm8k_solution', 'imdb', 'yelp']:
        label_ls = [0, 1]
        num_label = 2
    elif args.target_dataset == 'ag':
        label_ls = [0, 1, 2, 3]
        num_label = 4

    # construct the model (target model: wordLSTM, wordCNN, bert)
    print("Building Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if args.target_model == 'lstm':
    #     model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
    #     checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
    #     model.load_state_dict(checkpoint)
    # elif args.target_model == 'cnn':
    #     model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=150, cnn=True).cuda()
    #     checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
    #     model.load_state_dict(checkpoint)
    # elif args.target_model == 'bert':
    #     model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length).cuda()
    if args.target_model == 'llama-3.2-1b':
        if args.target_dataset == 'imdb':
            model_name = "yash3056/Llama-3.2-1B-imdb"
        else: 
            model_name = os.path.join("./models/llama_3.2_1b", f"{args.target_dataset}_finetuned_model", "final")
    elif args.target_model == 'llama-3.2-3b':
        if args.target_dataset == 'imdb':
            model_name = "yash3056/Llama-3.2-3B-imdb"
        else: 
            model_name = os.path.join("./models/llama_3.2_3b", f"{args.target_dataset}_finetuned_model", "final")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = args.max_seq_length
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_label)
    model = model.to(device)
    

    # predictor = model.text_pred
    predictor = llama_get_label_and_confidence
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}
    sim_lis=[]
    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1


    # load top 50 synonym file of counter-fitted-vectors
    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        with open(args.counter_fitting_cos_sim_path, "rb") as fp:
            sim_lis = pickle.load(fp)
    else:
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        cos_sim = np.dot(embeddings, embeddings.T)
    print("Cos sim import finished!")

    # use is used to compute the semantic similarity score
    use = USE(args.USE_cache_path)
    

    # start attacking
    orig_failures = 0.    
    adv_failures = 0.     
    num_attackable_samples = 0              
    final_changed_rates = []      
    nums_queries = []      
    orig_texts = []         
    adv_texts = []
    true_labels = []
    new_labels = []
    s_queries=[]          
    f_queries=[]           
    success=[]
    results=[]
    fails=[]
    final_sims = []
    orig_changed_rates = []
    final_semantic_sims = []


    # create directory for saving results
    orig_sent_dir = args.output_dir+ 'orig_sent/'+args.target_model+"/"+args.target_dataset
    adv_sent_dir = args.output_dir+'/adv_sent/'+args.target_model+"/"+args.target_dataset
    orig_and_adv_dir =args.output_dir+'orig_and_adv_sent/'+args.target_model+"/"+args.target_dataset
    log_results_dir = args.output_dir+'log_results/'+args.target_model+"/"+args.target_dataset
    csv_results_dir = args.output_dir+'csv_result/'+args.target_model+"/"+args.target_dataset
    Path(orig_sent_dir).mkdir(parents=True, exist_ok=True)
    Path(adv_sent_dir).mkdir(parents=True, exist_ok=True)
    Path(orig_and_adv_dir).mkdir(parents=True, exist_ok=True)
    Path(log_results_dir).mkdir(parents=True, exist_ok=True)
    Path(csv_results_dir).mkdir(parents=True, exist_ok=True)

    
    print('Start attacking!')
    with open(log_results_dir + '/' + args.target_dataset + '_result.txt', 'a') as logfile, \
     open(csv_results_dir + '/' + args.target_dataset + '_csv_result.txt', 'w', newline='') as csvfile, \
     open(orig_and_adv_dir + '/' + args.target_dataset + '.txt', 'w') as origadvfile, \
     open(orig_sent_dir + '/' + args.target_dataset + '.txt', 'w') as origfile, \
     open(adv_sent_dir + '/' + args.target_dataset + '.txt', 'w') as advfile, \
     open('results.txt', 'a') as result_file:  # This is to store intermediate results for failure tracking.

        # Prepare the CSV writer once to write all results in a batch
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Index", "Original Label", "New Label", "Original Text", "Adversarial Text", "Num Queries", "Final Changed Rate (%)", "Original Changed Rate (%)"])

        # Process each sample in the data
        for idx, (text, true_label) in enumerate(data):
            print("{} Samples Done".format(int(idx)))
            
            # Perform attack (modify as needed)
            new_text, num_changed, orig_changed, \
            orig_label, new_label, \
            num_queries, final_semantic_sim = attack(text, predictor, model, tokenizer, label_ls, true_label, word2idx, idx2word, sim_lis, use,
                                sim_threshold = args.sim_score_threshold, 
                                top_k_words=args.top_k_words,
                                batch_size=args.batch_size,
                                synonym_num=args.synonym_num,
                                theta=args.theta)
            
            # Failure tracking
            orig_label = int(orig_label)
            true_label = int(true_label)
            new_label = int(new_label)
            if true_label != orig_label:
                orig_failures += 1
            else:
                nums_queries.append(num_queries)

            if true_label != new_label:
                adv_failures += 1

            final_changed_rate = 1.0 * num_changed / len(text)
            orig_changed_rate = 1.0 * orig_changed / len(text)
            
            # Save the results incrementally
            if true_label == orig_label and true_label != new_label:
                temp = [idx, orig_label, new_label, ' '.join(text), new_text, num_queries, final_changed_rate * 100, orig_changed_rate * 100]
                csvwriter.writerow(temp)
                result_file.write('{}\n'.format(temp))  # Store in a result file line by line
                
                orig_texts.append(' '.join(text))
                adv_texts.append(new_text)
                true_labels.append(true_label)
                new_labels.append(new_label)
                orig_changed_rates.append(orig_changed_rate)
                final_changed_rates.append(final_changed_rate)
                final_semantic_sims.append(final_semantic_sim)
            
            if true_label == orig_label and true_label == new_label:
                temp1 = [idx, ' '.join(text), new_text, num_queries]
                fails.append(temp1)

                # Write adversarial failures incrementally
                result_file.write('{}\n'.format(temp1))
                
                f_queries.append(num_queries)
            if true_label == orig_label:
                num_attackable_samples += 1
            # Write the original and adversarial texts immediately to their respective files
            origfile.write('{}\n'.format(' '.join(text)))  # Write the original text
            advfile.write('{}\n'.format(new_text))  # Write the adversarial text
            origadvfile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(orig_label, ' '.join(text), new_label, new_text))


        # Final logging after all samples are processed
        message = 'theta is :{} For target model using TFIDF {} on dataset {} top words {} ' \
                'original accuracy: {:.3f}%, adv accuracy: {:.3f}%, original avg change: {:.3f}% ' \
                'avg changed rate: {:.3f}%, final_semantic_sims:{:.1f}, num of queries: {:.1f}'.format(args.theta,
                    args.target_model,
                    args.target_dataset,
                    args.top_k_words,
                    (1-orig_failures/args.data_size)*100,
                    # (1-(adv_failures/(args.data_size - orig_failures)))*100,
                    (1 - adv_failures/args.data_size) * 100,
                    np.mean(orig_changed_rates)*100,
                    np.mean(final_changed_rates)*100,
                    np.mean(final_semantic_sims),
                    np.mean(nums_queries))
        
        # Write final message to log file
        logfile.write(message)

        # Write all results to CSV file (this is done only once at the end)
        with open(csv_results_dir + '/' + args.target_dataset + '_csv_result.txt', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(results) 

if __name__ == "__main__":
    main()
