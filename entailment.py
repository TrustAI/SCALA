import sys
import pickle
import argparse
import os
from pathlib import Path
import numpy as np
from torch._C import device
np.random.seed(1234)
from scipy.special import softmax
import fnmatch
import criteria
import string
import pickle
import random
random.seed(0)
import csv
from InferSent.models import NLINet, InferSent, BLSTMEncoder
from esim.model import ESIM
from esim.data import Preprocessor
from esim.utils import correct_predictions
from collections import defaultdict
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig
import time



class NLI_infer_InferSent(nn.Module):
    def __init__(self,
                 pretrained_file,
                 embedding_path,
                 data,
                 batch_size=32):
        super(NLI_infer_InferSent, self).__init__()
        config_nli_model = {
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'n_enc_layers': 1,
            'dpout_model': 0.,
            'dpout_fc': 0.,
            'fc_dim': 512,
            'bsize': batch_size,
            'n_classes': 3,
            'pool_type': 'max',
            'nonlinear_fc': 0,
            'encoder_type': 'InferSent',
            'use_cuda': True,
            'use_target': False,
            'version': 1,
        }
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}


        print("\t* Building model...")
        self.model = NLINet(config_nli_model).cuda()

        for name, param in self.model.named_parameters():
            print(name, param.shape)
        print("Reloading pretrained parameters...")
        self.model.load_state_dict(torch.load(pretrained_file))

        print('Building vocab and embeddings...')
        self.dataset = NLIDataset_InferSent(embedding_path, data=data, batch_size=batch_size)

    def text_pred(self, text_data):
        self.model.eval()
        data_batches = self.dataset.transform_text(text_data)
        probs_all = []
        with torch.no_grad():
            for batch in data_batches:
                (s1_batch, s1_len), (s2_batch, s2_len) = batch
                s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)
        return torch.cat(probs_all, dim=0)

class NLI_infer_ESIM(nn.Module):
    def __init__(self,
                 pretrained_file,
                 worddict_path,
                 local_rank=-1,
                 batch_size=32):
        super(NLI_infer_ESIM, self).__init__()

        self.batch_size = batch_size
        self.device = torch.device("cuda:{}".format(local_rank) if local_rank > -1 else "cuda")
        checkpoint = torch.load(pretrained_file)
        vocab_size = checkpoint['model']['_word_embedding.weight'].size(0)
        embedding_dim = checkpoint['model']['_word_embedding.weight'].size(1)
        hidden_size = checkpoint['model']['_projection.0.weight'].size(0)
        num_classes = checkpoint['model']['_classification.4.weight'].size(0)

        print("\t* Building model...")
        self.model = ESIM(vocab_size,
                          embedding_dim,
                          hidden_size,
                          num_classes=num_classes,
                          device=self.device).to(self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.dataset = NLIDataset_ESIM(worddict_path)

    def text_pred(self, text_data):
        self.model.eval()
        device = self.device
        self.dataset.transform_text(text_data)
        dataloader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size)

        probs_all = []
        with torch.no_grad():
            for batch in dataloader:
                premises = batch['premise'].to(device)
                premises_lengths = batch['premise_length'].to(device)
                hypotheses = batch['hypothesis'].to(device)
                hypotheses_lengths = batch['hypothesis_length'].to(device)

                _, probs = self.model(premises,
                                      premises_lengths,
                                      hypotheses,
                                      hypotheses_lengths)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=64):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=3).cuda()
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data):
        self.model.eval()
        dataloader = self.dataset.transform_text(text_data)

        probs_all = []
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)
        return torch.cat(probs_all, dim=0)


def read_data(filepath, data_size, target_model='infersent', lowercase=False, ignore_punctuation=False, stopwords=[]):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """
    if target_model == 'bert':
        labeldict = {"contradiction": 0,
                      "entailment": 1,
                      "neutral": 2}
    else:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}
    with open(filepath, 'r', encoding='utf8') as input_data:
        premises, hypotheses, labels = [], [], []

        punct_table = str.maketrans({key: ' '
                                     for key in string.punctuation})

        for idx, line in enumerate(input_data):
            if idx >= data_size:
                break

            line = line.strip().split('\t')

            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)
            premises.append([w for w in premise.rstrip().split()
                             if w not in stopwords])
            hypotheses.append([w for w in hypothesis.rstrip().split()
                               if w not in stopwords])
            labels.append(labeldict[line[0]])

        return {"premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


class NLIDataset_ESIM(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 worddict_path,
                 padding_idx=0,
                 bos="_BOS_",
                 eos="_EOS_"):
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
        self.bos = bos
        self.eos = eos
        self.padding_idx = padding_idx
        with open(worddict_path, 'rb') as pkl:
            self.worddict = pickle.load(pkl)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
            "premise": self.data["premises"][index],
            "premise_length": min(self.premises_lengths[index],
                                  self.max_premise_length),
            "hypothesis": self.data["hypotheses"][index],
            "hypothesis_length": min(self.hypotheses_lengths[index],
                                     self.max_hypothesis_length)
        }

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                index = self.worddict['_OOV_']
            indices.append(index)
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"premises": [],
                            "hypotheses": []}

        for i, premise in enumerate(data['premises']):

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def transform_text(self, data):
        data = self.transform_to_indices(data)

        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {
            "premises": torch.ones((self.num_sequences,
                                    self.max_premise_length),
                                   dtype=torch.long) * self.padding_idx,
            "hypotheses": torch.ones((self.num_sequences,
                                      self.max_hypothesis_length),
                                     dtype=torch.long) * self.padding_idx}

        for i, premise in enumerate(data["premises"]):
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])



class NLIDataset_InferSent(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 embedding_path,
                 data,
                 word_emb_dim=300,
                 batch_size=32,
                 bos="<s>",
                 eos="</s>"):
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
        self.bos = bos
        self.eos = eos
        self.word_emb_dim = word_emb_dim
        self.batch_size = batch_size

        # build word dict
        self.word_vec = self.build_vocab(data['premises']+data['hypotheses'], embedding_path)

    def build_vocab(self, sentences, embedding_path):
        word_dict = self.get_word_dict(sentences)
        word_vec = self.get_embedding(word_dict, embedding_path)
        print('Vocab size : {0}'.format(len(word_vec)))
        return word_vec

    def get_word_dict(self, sentences):
        # create vocab of words
        word_dict = {}
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<oov>'] = ''
        return word_dict

    def get_embedding(self, word_dict, embedding_path):
        # create word_vec with glove vectors
        word_vec = {}
        word_vec['<oov>'] = np.random.normal(size=(self.word_emb_dim))
        with open(embedding_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with embedding vectors'.format(
            len(word_vec), len(word_dict)))
        return word_vec

    def get_batch(self, batch, word_vec, emb_dim=300):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        embed = np.zeros((max_len, len(batch), emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] in word_vec:
                    embed[j, i, :] = word_vec[batch[i][j]]
                else:
                    embed[j, i, :] = word_vec['<oov>']

        return torch.from_numpy(embed).float(), lengths

    def transform_text(self, data):
        premises = data['premises']
        hypotheses = data['hypotheses']

        # add bos and eos
        premises = [['<s>'] + premise + ['</s>'] for premise in premises]
        hypotheses = [['<s>'] + hypothese + ['</s>'] for hypothese in hypotheses]

        batches = []
        for stidx in range(0, len(premises), self.batch_size):
            s1_batch, s1_len = self.get_batch(premises[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            s2_batch, s2_len = self.get_batch(hypotheses[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            batches.append(((s1_batch, s1_len), (s2_batch, s2_len)))

        return batches


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
                 batch_size=32):
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

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, (text_a, text_b)) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            tokens_b = None
            if text_b:
                tokens_b = tokenizer.tokenize(' '.join(text_b))
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
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

    def transform_text(self, data):
        eval_features = self.convert_examples_to_features(list(zip(data['premises'], data['hypotheses'])),
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

        return eval_dataloader



def get_attack_result(hypotheses, premise, predictor, orig_label):

    new_probs = predictor({'premises': [premise] * len(hypotheses), 'hypotheses': hypotheses})
    pr=(orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
    return pr



def get_attack_prob(hypotheses, premise, predictor):
    
    new_probs = predictor({'premises': [premise] * len(hypotheses), 'hypotheses': hypotheses})
    new_prob = new_probs.max().data.cpu().numpy()
    return new_prob



def attack(hypotheses, premise, predictor, true_label, word2idx, idx2word, cos_sim, qrs, top_k_words, batch_size, synonym_num,theta):
 
    orig_probs = predictor({'premises': [premise], 'hypotheses': [hypotheses]}).squeeze() #predictor(premise,hypothese).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()


    if true_label != orig_label:
        print('Original classifier fail')
        return '', 0, 0, orig_label, orig_label, 0
    else:
        text_ls = hypotheses[:]
        pos_ls = criteria.get_pos(text_ls)


        # get the pos and verb tense info
        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        # pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        # for pos in pos_pref:
        for i in range(len(pos_ls)):
            words_perturb.append((i, text_ls[i]))
    
    
        # find synonyms and make a dict of synonyms of each word.
        words_perturb = words_perturb[:top_k_words]
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words,synonym_values=[],[]
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

        # Find a reasonable sort 
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
                new_texts[n] = new_text[:]
                n += 1          
                orig_qrs += 0   
        new_probs = predictor({'premises':[premise]*int(theta * synonym_num)*len(synonyms_all), 'hypotheses':new_texts}).squeeze() 
        saliency_scores = torch.sub(orig_prob, new_probs[:, orig_label])
        saliency_scores = saliency_scores.reshape([len(synonyms_all), int(theta * synonym_num)])
        saliency_scores_values = torch.max(saliency_scores, dim=1)[0]
        saliency_scores_indices = torch.argmax(saliency_scores, dim=1)
        argsort_saliency_scores = torch.argsort(saliency_scores_values, descending=True).tolist()
        saliency_scores_indices = saliency_scores_indices[argsort_saliency_scores].tolist()


        # replace with synonyms based on the sort
        orig_changed = 0  
        current_text = text_ls[:]
        replace_indices = []
        for i in range(len(argsort_saliency_scores)):
            idx = synonyms_all[argsort_saliency_scores[i]][0]
            syn = synonyms_all[argsort_saliency_scores[i]][1]
            current_texts = [None]* synonym_num
            n = 0
            for j in range(len(syn)):
                current_text[idx] = syn[j]  
                current_texts[n] = current_text[:]
                n+=1
            current_probs = predictor({'premises':[premise]*synonym_num, 'hypotheses':current_texts}).squeeze()
            current_saliency_scores = torch.sub(orig_prob.detach().clone(), current_probs[:, orig_label])
            current_max_index = torch.argmax(current_saliency_scores).tolist()
            current_text[idx] = syn[current_max_index] 
            replace_indices.append(idx)
            pr = get_attack_result([current_text], premise, predictor, orig_label)
            orig_qrs += 1
            orig_changed += 1
            if np.sum(pr)>0:
                flag = 1
                break


        # replace back based on sort
        if flag == 1:
            if orig_changed >= 2:
                replace_back_qrs = 0
                replace_back = 0
                one_word_texts = [None] * orig_changed
                m = 0
                j = 0  
                new_probs = predictor({'premises':[premise], 'hypotheses':current_text}).squeeze()
                new_label = torch.argmax(new_probs)
                new_prob = new_probs.max()
                for index in range(len(replace_indices)):
                    one_word_text = current_text[:]
                    one_word_text[index] = text_ls[index]
                    one_word_texts[m] = one_word_text[:]
                    m += 1
                    replace_back_qrs += 1
                replace_back_indices = replace_indices[:]
                one_word_probs = predictor({'premises':[premise]*len(replace_indices), 'hypotheses':one_word_texts}).squeeze()
                saliency_scores_back = torch.sub(new_prob.detach().clone(), one_word_probs[:, new_label])
                argsort_saliency_scores_back = torch.argsort(saliency_scores_back, descending=False)
                replace_back_indices = [replace_back_indices[i] for i in argsort_saliency_scores_back.tolist()]
                adv_text = current_text[:]
                for i in range(len(replace_back_indices)):
                    adv_text[replace_back_indices[i]] = text_ls[replace_back_indices[i]]
                    pr = get_attack_result([adv_text], premise, predictor, orig_label)
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
            print('attack success')
            return ' '.join(adv_text), final_changed, orig_changed, \
                orig_label,  torch.argmax(predictor({'premises':[premise], 'hypotheses': [adv_text]})),\
                qrs
        else:
            print("attack fail")
            return '', 0, 0, orig_label, orig_label, 0
        

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--target_dataset",
                        default="snli",
                        type=str,
                        required=True,
                        help="Dataset Name")
    parser.add_argument("--target_model",
                        type=str,
                        default="bert",
                        required=True,
                        choices=['infersent', 'esim', 'bert'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--target_model_path",
                        type=str,
                        default="../pretrained_models/bert/snli",
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--dataset_dir",
                        default="../data/",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--output_dir",
                        default="../final_results/entailment/",
                        type=str,
                        required=True,
                        help="Which directory to save results.")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default="../embedding/glove.6B.200d.txt",
                        required=True,
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        default="../counter-fitted-vectors.txt",
                        required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='../mat.txt',
                        required=True,
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")


    ## Model hyperparameters
    parser.add_argument("--theta",
                        default=0.1,
                        type=int,
                        help="parameter")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--top_k_words",
                        default=1000000,
                        type=int,
                        help="Top K Words")
    parser.add_argument("--allowed_qrs",
                        default=1000000,
                        type=int,
                        help="Allowerd qrs")

    args = parser.parse_args()

    # get data to attack
    data = read_data(args.dataset_dir+args.target_dataset, data_size=args.data_size, target_model=args.target_model)
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'esim':
        model = NLI_infer_ESIM(args.target_model_path,
                                args.word_embeddings_path,
                               batch_size=args.batch_size)
    elif args.target_model == 'infersent':
        model = NLI_infer_InferSent(args.target_model_path,
                                    args.word_embeddings_path,
                                    data=data,
                                    batch_size=args.batch_size)
    else:
        model = NLI_infer_BERT(args.target_model_path)
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    print("Building vocab...")
    idx2word = {}
    word2idx = {}
    sim_lis=[]
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    # for cosine similarity matrix
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
        print(embeddings.T.shape)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        cos_sim = np.dot(embeddings, embeddings.T)
    print("Cos sim import finished!")

    whole_time1 = time.time()

    # start attacking
    orig_failures = 0.    
    adv_failures = 0.     
    final_changed_rates = []      
    nums_queries = []      
    orig_texts = []         
    adv_texts = []
    true_labels = []
    new_labels = []
    s_queries=[]         
    success=[]
    results=[]
    orig_changed_rates = []
    adv_hypotheses = []
    true_labels = []
    new_labels = []
    orig_labels=[]
    premises = []

    # create directory for saving results
    orig_sent_dir = args.output_dir+'orig_sent/'+args.target_model+"/"+args.target_dataset
    adv_sent_dir = args.output_dir+'adv_sent/'+args.target_model+"/"+args.target_dataset
    orig_and_adv_dir = args.output_dir+'orig_and_adv_sent/'+args.target_model+"/"+args.target_dataset
    log_results_dir = args.output_dir+'log_results/'+args.target_model+"/"+args.target_dataset
    csv_results_dir = args.output_dir+'csv_result/'+args.target_model+"/"+args.target_dataset
    time_dir = args.output_dir+'time/'+args.target_model+"/"+args.target_dataset
    Path(orig_sent_dir).mkdir(parents=True, exist_ok=True)
    Path(adv_sent_dir).mkdir(parents=True, exist_ok=True)
    Path(orig_and_adv_dir).mkdir(parents=True, exist_ok=True)
    Path(log_results_dir).mkdir(parents=True, exist_ok=True)
    Path(csv_results_dir).mkdir(parents=True, exist_ok=True)
    Path(time_dir).mkdir(parents=True, exist_ok=True)
    open(time_dir + '/parrallel_time.txt', "w").close()

    whole_time1 = time.time()
    for idx, premise in enumerate(data['premises']):
        if idx % 10 == 0:
            print('{} samples out of {} have been finished!'.format(idx, args.data_size))
        single_time1 = time.time()
        hypothese, true_label = data['hypotheses'][idx], data['labels'][idx]
        new_text, num_changed, orig_changed, orig_label, \
        new_label, num_queries = attack(hypothese, premise, predictor, true_label,
                                        word2idx, idx2word, sim_lis ,
                                        qrs = args.allowed_qrs, top_k_words = args.top_k_words, 
                                        batch_size = args.batch_size,
                                        synonym_num = args.synonym_num,
                                        theta = args.theta)
        single_time2 = time.time()
        single_time = single_time2 - single_time1
        with open(time_dir + '/parrallel_time.txt', 'a') as f:
            f.write('{}\n'.format(single_time))

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)
        if true_label != new_label:
            adv_failures += 1

        final_changed_rate = 1.0 * num_changed / len(hypothese)
        orig_changed_rate = 1.0 * orig_changed / len(hypothese)
        if true_label == orig_label and true_label != new_label:
            temp=[]
            s_queries.append(num_queries)
            success.append(idx)
            final_changed_rates.append(final_changed_rate)
            premises.append(premise)
            orig_texts.append(' '.join(hypothese))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)
            orig_changed_rates.append(orig_changed_rate)
            temp.append(orig_label)
            temp.append(' '.join(premise))
            temp.append(' '.join(hypothese))
            temp.append(new_text)
            orig_labels.append(orig_label)
            adv_hypotheses.append(''.join(hypothese))
            results.append(temp)

    
    whole_time2 = time.time()


    message = 'theta={}For target model {}   top words {} qrs {} : ' \
              'original accuracy: {:.3f}%, adv accuracy: {:.3f}%, random avg  change: {:.3f}% ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}, It costs {} seconds\n'.format(args.theta,
                                                                    args.target_model,
                                                                      args.top_k_words,args.allowed_qrs,
                                                                     (1-orig_failures/args.data_size)*100,
                                                                     (1-adv_failures/args.data_size)*100,
                                                                     np.mean(orig_changed_rates)*100,
                                                                     np.mean(final_changed_rates)*100,
                                                                     np.mean(nums_queries),
                                                                     whole_time2-whole_time1)

    print(message)
    
    if args.target_model == 'bert':
            labeldict = {0: "contradiction",
                     1: "entailment",
                     2:  "neutral"}
    else:
        labeldict = {0: "entailment",
                     1: "neutral",
                     2: "contradiction"}
    with open(log_results_dir+'/'+args.target_dataset+'_result.txt','a') as logfile:
        logfile.write(message)

    with open(csv_results_dir+'/'+args.target_dataset+'_csv_result.txt','w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)

    with open(orig_and_adv_dir+'/'+args.target_dataset+'.txt','w') as origadvfile:
        for premise, orig_text, adv_text, true_label, new_label in zip(premises, orig_texts, adv_texts, true_labels, new_labels):
            origadvfile.write('premise:{}\norig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(premise, labeldict[true_label], orig_text, labeldict[int(new_label)], adv_text))

    with open(orig_sent_dir+'/'+args.target_dataset+'.txt','w') as origfile:
        for orig_text in orig_texts:
            origfile.write('{}\n'.format(orig_text))

    with open(adv_sent_dir+'/'+args.target_dataset+'.txt','w') as advfile:
        for adv_text in adv_texts:
            advfile.write('{}\n'.format(adv_text))

if __name__ == "__main__":
    main()