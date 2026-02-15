import torch 
import criteria
from collections import defaultdict
import argparse
import dataloader
from train_classifier import Model
import torch.nn as nn
from BERT.modeling import BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from BERT.tokenization import BertTokenizer
import numpy as np
import pickle
from pathlib import Path
import csv
import sys
import random
csv.field_size_limit(sys.maxsize)
np.random.seed(1234)
random.seed(0)
import time



class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        self.model.eval()
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)
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



def get_attack_result(new_text, predictor, orig_label, batch_size):
    new_probs = predictor(new_text, batch_size=batch_size)
    pr=(orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
    return pr
    
def get_attack_prob(new_text, predictor, batch_size):  
    new_probs = predictor(new_text, batch_size=batch_size)
    new_prob = new_probs.max().data.cpu().numpy()
    return new_prob

def attack(text_ls, predictor, true_label, word2idx, idx2word, cos_sim, top_k_words, batch_size, synonym_num, theta):
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        print('Original classifier fail')
        return '', 0, 0, orig_label, orig_label, 0
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
        new_probs = predictor(new_texts, batch_size).squeeze()
        new_labels = torch.argmax(new_probs, dim=1)
        
        saliency_scores = torch.sub(orig_prob.detach().clone(), new_probs[:, orig_label])
        saliency_scores = saliency_scores.reshape([len(synonyms_all), int(theta*synonym_num)])
        saliency_scores_values = torch.max(saliency_scores, dim=1)[0]
        saliency_scores_indices = torch.argmax(saliency_scores, dim=1)
        argsort_saliency_scores = torch.argsort(saliency_scores_values, descending=True).tolist()
        saliency_scores_indices = saliency_scores_indices[argsort_saliency_scores].tolist()

        # step 3: replace with synonyms based on the sort
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
            current_probs = predictor(current_texts, batch_size).squeeze()
            current_saliency_scores = torch.sub(orig_prob.detach().clone(), current_probs[:, orig_label])
            current_max_index = torch.argmax(current_saliency_scores).tolist()
            print(current_max_index)
            current_text[idx] = syn[current_max_index] 
            replace_indices.append(idx)
            pr = get_attack_result([current_text], predictor, orig_label, batch_size)
            orig_qrs += 1
            orig_changed += 1
            if np.sum(pr)>0:
                flag = 1
                print('current label',torch.argmax(predictor([current_text])))
                print('original changed',orig_changed)
                break
   
        
        # step 4: replace back based on sort
        if flag == 1:
            if orig_changed >= 2:
                replace_back_qrs = 0
                replace_back = 0
                one_word_texts = [None] * orig_changed
                m = 0
                j = 0  
                new_probs = predictor([current_text], batch_size).squeeze()
                new_label = torch.argmax(new_probs)
                new_prob = new_probs.max()
                for index in range(len(replace_indices)):
                    one_word_text = current_text[:]
                    one_word_text[index] = text_ls[index]
                    one_word_texts[m] = one_word_text[:]
                    m += 1
                    replace_back_qrs += 1
                replace_back_indices = replace_indices[:]
                one_word_probs = predictor(one_word_texts, batch_size).squeeze()
                saliency_scores_back = torch.sub(new_prob.detach().clone(), one_word_probs[:, new_label])
                argsort_saliency_scores_back = torch.argsort(saliency_scores_back, descending=False)
                replace_back_indices = [replace_back_indices[i] for i in argsort_saliency_scores_back.tolist()]
                adv_text = current_text[:]
                for i in range(len(replace_back_indices)):
                    adv_text[replace_back_indices[i]] = text_ls[replace_back_indices[i]]
                    pr = get_attack_result([adv_text], predictor, orig_label, batch_size)
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
                  orig_label, torch.argmax(predictor([adv_text])), qrs
        else:
            print("attack fail")
            return '', 0, 0, orig_label, orig_label, 0

        


def main():
    
    # create parser
    parser = argparse.ArgumentParser()

    ###### add parameters
    parser.add_argument("--target_model",
                        default="bert",
                        type=str,
                        required=True,
                        choices=['lstm', 'bert', 'cnn'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--target_dataset",
                        default="mr",
                        type=str,
                        required=True,
                        help="Dataset Name: mr, imdb, yelp, ag, snli, mnli")
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
                        default="../counter-fitted-vectors.txt",
                        required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='../mat.txt',
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
                        default=0.2,
                        type=int,
                        help="Number of synonyms to extract")
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


    # parser paremeters
    args = parser.parse_args()


    # get data to attack
    texts, labels = dataloader.read_corpus(args.dataset_dir+args.target_dataset,csvf=False)
    data = list(zip(texts, labels))
    data = data[:args.data_size] 
    print("Data import finished!")


    # construct the model (target model: wordLSTM, wordCNN, bert)
    print("Building Model...")
    if args.target_model == 'lstm':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'cnn':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=150, cnn=True).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length).cuda()
    predictor = model.text_pred
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
    f_queries=[]           
    success=[]
    results=[]
    fails=[]
    final_sims = []
    orig_changed_rates = []


    # create directory for saving results
    orig_sent_dir = args.output_dir+ 'orig_sent/'+args.target_model+"/"+args.target_dataset
    adv_sent_dir = args.output_dir+'/adv_sent/'+args.target_model+"/"+args.target_dataset
    orig_and_adv_dir =args.output_dir+'orig_and_adv_sent/'+args.target_model+"/"+args.target_dataset
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

    
    print('Start attacking!')
    for idx, (text, true_label) in enumerate(data):
        print("{} Samples Done".format(int(idx)))
        single_time1 = time.time()
        new_text, num_changed, orig_changed, \
        orig_label, new_label, \
        num_queries = attack(text, predictor, true_label, word2idx, idx2word, sim_lis, 
                                                top_k_words=args.top_k_words,
                                                batch_size=args.batch_size,
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
 

        final_changed_rate = 1.0 * num_changed / len(text)
        orig_changed_rate = 1.0 * orig_changed / len(text)
        if true_label == orig_label and true_label != new_label:
            temp=[]
            s_queries.append(num_queries)
            success.append(idx)
            final_changed_rates.append(final_changed_rate)
            orig_texts.append(' '.join(text))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)
            orig_changed_rates.append(orig_changed_rate)
            temp.append(idx)
            temp.append(orig_label)
            temp.append(new_label)
            temp.append(' '.join(text))
            temp.append(new_text)
            temp.append(num_queries)
            temp.append(final_changed_rate * 100)
            temp.append(orig_changed_rate * 100)
            results.append(temp)
        if true_label == orig_label and true_label == new_label:
            f_queries.append(num_queries)
            temp1=[]
            temp1.append(idx)
            temp1.append(' '.join(text))
            temp1.append(new_text)
            temp1.append(num_queries)
            fails.append(temp1)
    whole_time2 = time.time()

    message = 'theta is :{} For target model using TFIDF {} on dataset {} top words {} ' \
              'original accuracy: {:.3f}%, adv accuracy: {:.3f}%, original avg  change: {:.3f}% ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}'\
              ' It costs {} seconds\n'.format(args.theta,
                args.target_model,
                args.target_dataset,
                args.top_k_words,
                (1-orig_failures/args.data_size)*100,
                (1-adv_failures/args.data_size)*100,
                np.mean(orig_changed_rates)*100,
                np.mean(final_changed_rates)*100,
                np.mean(nums_queries),
                whole_time2-whole_time1)
    print(message)
    with open(log_results_dir+'/'+args.target_dataset+'_result.txt','a') as logfile:
        logfile.write(message)

    with open(csv_results_dir+'/'+args.target_dataset+'_csv_result.txt','w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)

    with open(orig_and_adv_dir+'/'+args.target_dataset+'.txt','w') as origadvfile:
        for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
            origadvfile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))

    with open(orig_sent_dir+'/'+args.target_dataset+'.txt','w') as origfile:
        for orig_text in orig_texts:
            origfile.write('{}\n'.format(orig_text))

    with open(adv_sent_dir+'/'+args.target_dataset+'.txt','w') as advfile:
        for adv_text in adv_texts:
            advfile.write('{}\n'.format(adv_text))

if __name__ == "__main__":
    main()
