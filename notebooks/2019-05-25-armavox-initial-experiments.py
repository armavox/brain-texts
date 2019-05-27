#!/usr/bin/env python
# coding: utf-8

# In[6]:


import re
from bert_embedding import BertEmbedding


# In[70]:


import pandas as pd


# In[3]:


text = """On MR images of the sub-and supratentorial structures of the brain, supplemented by intravenous contrast (Gadovist 10 ml) obtained the following data:
• in the brain substance numerous formations of specific genesis in both hemispheres of the brain were revealed:
o - at least 7 in the left hemisphere - from 2mm to 4.5mm in diameter;
o - at least 4 in the right cerebral hemisphere from 2mm to 8x7mm in the supratentorial regions of the frontal-parietal region, 14x17mm in the basal regions of the temporal lobe;
o - in the left hemisphere of the cerebellum two formations 9x7.5mm and 9.2x8.3mm"""


# In[11]:


text


# In[9]:


pattern = re.compile(r'[a-zA-Z0-9]+')


# In[12]:


wordlist = pattern.findall(text)


# In[13]:


be = BertEmbedding()


# In[15]:


embs = be(wordlist)


# In[92]:


train_df = pd.read_csv('/Users/artem/pyprojects/brain-tumor/data/TRAIN_SET/ozerki_annotated_not_segmented/brain-text1.csv', names=['name', 'annot', 'label'])


# In[419]:


train_df['label'].to_csv('labels.txt', sep='\n', header=False, index=False)


# In[95]:


X = train_df['annot']


# In[68]:


import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

batch_size = 32
X_train, y_train = samples_from_file('train.csv') # Put your own data loading function here
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train = [tokenizer.tokenize('[CLS] ' + sent + ' [SEP]') for sent in X_train] # Appending [CLS] and [SEP] tokens - this probably can be done in a cleaner way
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model = bert_model.cuda()

X_train_tokens = [tokenizer.convert_tokens_to_ids(sent) for sent in X_train]
results = torch.zeros((len(X_test_tokens), bert_model.config.hidden_size)).long()
with torch.no_grad():
    for stidx in range(0, len(X_test_tokens), batch_size):
        X = X_test_tokens[stidx:stidx + batch_size]
        X = torch.LongTensor(X).cuda()
        _, pooled_output = bert_model(X)
        results[stidx:stidx + batch_size,:] = pooled_output.cpu()


# In[87]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[100]:


bert_model = BertModel.from_pretrained('bert-base-uncased')


# In[147]:


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
    
    def __repr__(self):
        return self.text_a


# In[248]:


def read_sequence(input_sentences):
    examples = []
    unique_id = 0
    for sentence in input_sentences:
        examples.append(InputExample(unique_id=unique_id, text_a=sentence))
        unique_id += 1
    return examples

examples = read_sequence(X)


# In[127]:


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


# In[247]:


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length, f'{input_ids, len(input_ids)}'
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# In[249]:


MAX_SEQ_LENGTH = 256
features = convert_examples_to_features(examples, seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer)


# In[214]:


def get_features(input_text, dim=768):
    layer_indexes = LAYERS
  
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  
    examples = read_sequence(input_text)
  
    features = convert_examples_to_features(
        examples=examples, seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer)
  
    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature
  
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=INIT_CHECKPOINT,
        layer_indexes=layer_indexes,
        use_tpu=True,
        use_one_hot_embeddings=True)
  
    input_fn = input_fn_builder(
        features=features, seq_length=MAX_SEQ_LENGTH)
  
    # Get features
    for result in estimator.predict(input_fn, yield_single_examples=True):
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]
        output = collections.OrderedDict()
        for (i, token) in enumerate(feature.tokens):
            layers = []
            for (j, layer_index) in enumerate(layer_indexes):
                layer_output = result["layer_output_%d" % j]
                layer_output_flat = np.array([x for x in layer_output[i:(i + 1)].flat])
                layers.append(layer_output_flat)
            output[token] = sum(layers)[:dim]
    
    return output


# In[265]:


bert_model(torch.LongTensor([31, 51, 99]))


# In[263]:


batch_size = 2
results = torch.zeros((256, bert_model.config.hidden_size)).long()
with torch.no_grad():
    for stidx in range(0, 256, batch_size):
        X = X_test_tokens[stidx:stidx + batch_size]
        X = torch.LongTensor(X).cuda()
        _, pooled_output = bert_model(X)
        results[stidx:stidx + batch_size,:] = pooled_output.cpu()

bert_model(torch.LongTensor(features[0].input_ids))


# In[320]:


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


# In[270]:


ex = read_examples('../../brain-tumor/data/TRAIN_SET/ozerki_annotated_not_segmented/brain-text.txt')


# In[274]:


with open('../../brain-tumor/data/TRAIN_SET/ozerki_annotated_not_segmented/annotations2.txt') as f:
    text = f.read()


# In[279]:


patients = text.split('\n\n')


# In[291]:


for i, patient in enumerate(patients):
    patients[i] = patient.replace('\n', ' ')


# In[293]:


patients[0]


# In[311]:


with open('annotations3.txt', 'w') as f:
    for item in patients:
        f.write('%s\n' % item)


# In[312]:


with open('annotations3.txt') as f:
    text = f.read()
text.split('\n').__len__()


# In[316]:


patients.__len__()


# In[322]:


ex = read_examples('annotations3.txt')


# In[327]:


ex[8].unique_id


# In[328]:


import torch


# In[329]:


get_ipython().system('python -V')


# In[332]:


import json


# In[334]:


get_ipython().system('ls')


# In[347]:


inp_ids = torch.LongTensor(features[0].input_ids)
inp_mask = torch.LongTensor(features[0].input_mask)


# In[355]:


from torch.utils.data import TensorDataset, SequentialSampler, DistributedSampler, DataLoader


# In[44]:


all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
if -1 == -1:
    eval_sampler = SequentialSampler(eval_data)
else:
    eval_sampler = DistributedSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)


# In[556]:


with torch.no_grad():
    emb_dataset = []
    for inp_id, inp_mask, ex_index in eval_dataloader:
        layers, _ = bert_model(inp_id, token_type_ids=None, attention_mask=inp_mask)
        stack_last_4_layers = torch.stack(layers[-4:])
        embedding = torch.sum(stack_last_4_layers, dim=0)
        emb_dataset.append(embedding)
    dataset = TensorDataset(torch.cat(emb_dataset, dim=0))


# In[557]:


dataset[0]


# In[552]:


TensorDataset(torch.cat(emb_dataset, dim=0))[0]


# In[374]:


_.shape


# In[382]:


eval_data[0]


# In[390]:


list(iter(eval_dataloader))[-1]


# In[410]:


torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        ).unsqueeze(1).shape


# In[398]:


bert_model(inp_ind, token_type_ids=None, attention_mask=inp_mask)


# In[404]:


list(iter(eval_dataloader))[-1][0].shape


# In[405]:


list(eval_data[-1])[0].shape


# In[420]:


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


# In[423]:


read_examples('annotations_eng.txt')[-1]


# In[428]:


with open('labels.txt') as f:
    labels = []
    for line in f:
        labels.append(float(line.strip('\n')))


# In[429]:


labels


# In[431]:


eval_data[:4]


# In[433]:


from torch.utils.data import Dataset


# In[439]:


def read_labels(input_file):
    with open(input_file) as f:
        labels = []
        for line in f:
            labels.append(float(line.strip('\n')))
        return labels


# In[45]:


class BertFeaturesDataset(Dataset):
    """
    Parameters
    ----------
    input_file : str
        Path to input file with strings
    bert_model : str
        Bert pre-trained model selected in the list: bert-base-uncased,
        bert-large-uncased, bert-base-cased, bert-base-multilingual,
        bert-base-chinese.
    """
    def __init__(self, input_file, labels_file, bert_model, max_seq_length=256):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True
        )
        examples = read_examples(input_file)
        features = convert_examples_to_features(
            examples=examples, seq_length=max_seq_length, tokenizer=tokenizer
        )

        self.model = BertModel.from_pretrained(bert_model)
        self.model.to(device)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_example_index = torch.arange(
            self.all_input_ids.size(0), dtype=torch.long
        )
        self.dataset = TensorDataset(self.all_input_ids,
                                     self.all_input_mask,
                                     self.all_example_index)
        self.labels = read_labels(labels_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


# In[46]:


bert = BertFeaturesDataset('annotations_eng.txt', 'labels.txt', 'bert-base-uncased')


# In[47]:


bert_loader = DataLoader(bert, batch_size=1)


# In[51]:


(inp_id, inp_mask, ex_index), _  = next(iter(bert_loader))


# In[53]:


model = BertModel.from_pretrained('bert-base-uncased')


# In[55]:


layers, pool = model(inp_id, token_type_ids=None,attention_mask=inp_mask)


# In[73]:


t = TensorDataset(torch.stack(layers[-4:]))[0]


# In[74]:


type(t)


# In[43]:


bert[0][0]


# In[85]:


# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import re

import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


def read_labels(input_file):
    with open(input_file) as f:
        labels = []
        for line in f:
            labels.append(float(line.strip('\n')))
        return labels


class BertFeaturesDataset(Dataset):
    """
    Parameters
    ----------
    input_file : str
        Path to input file with strings
    bert_model : str
        Bert pre-trained model selected in the list: bert-base-uncased,
        bert-large-uncased, bert-base-cased, bert-base-multilingual,
        bert-base-chinese.
    """

    def __init__(self, input_file, labels_file, bert_model,
                 max_seq_length=256, batch_size=4):

        self.input_file = input_file
        self.labels_file = labels_file
        self.bert_model = bert_model
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        self.tensor_dataset = self.init_bert_dataset(
            self.input_file, self.labels_file,
            self.bert_model, self.max_seq_length
        )
        self.dataset = self.get_bert_embeddings(self.tensor_dataset,
                                                self.bert_model,
                                                self.batch_size)

        self.labels = read_labels(self.labels_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = {
            'embedding': self.dataset[idx][0],
            'label': self.labels[idx]
        }
        return sample

    def init_bert_dataset(self, input_file, labels_file, bert_model,
                          max_seq_length):

        tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True
        )
        examples = read_examples(input_file)
        features = convert_examples_to_features(
            examples=examples, seq_length=max_seq_length, tokenizer=tokenizer
        )

        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_example_index = torch.arange(
            self.all_input_ids.size(0), dtype=torch.long
        )
        return TensorDataset(self.all_input_ids,
                             self.all_input_mask,
                             self.all_example_index)

    def get_bert_embeddings(self, tensor_dataset, bert_model, batch_size):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        model = BertModel.from_pretrained(bert_model)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler,
                                batch_size=self.batch_size)

        embedding_dataset = []
        with torch.no_grad():
            for inp_id, inp_mask, ex_index in dataloader:
                layers, _ = model(inp_id, token_type_ids=None,
                                  attention_mask=inp_mask)
                stack_last_4_layers = torch.stack(layers[-4:])
                embedding = torch.sum(stack_last_4_layers, dim=0)
                embedding_dataset.append(embedding)

        return TensorDataset(torch.cat(embedding_dataset, dim=0))


# In[86]:


bert = BertFeaturesDataset('annotations_eng.txt', 'labels.txt', 'bert-base-uncased')


# In[91]:


dl = DataLoader(bert, batch_size=3)


# In[92]:


for batch in dl:
    emb = batch['embedding']


# In[97]:


bert[:4]['embedding'].shape


# In[84]:


bert[0]['embedding']


# In[41]:





# In[100]:


import pandas as pd


# In[106]:



df = pd.read_csv('../../brain-tumor/data/TRAIN_SET/ozerki_annotated_not_segmented/brain-labels.csv', header=None, dtype={0:str, 1:int})


# In[107]:


df.info()


# In[116]:


df.iloc[:,0].to_list()


# In[18]:


import os
path_to_img_folder = '../../brain-tumor/data/TRAIN_SET/ozerki_annotated_not_segmented/'
patient_name = 'G10'


# In[121]:


path = os.path.join(path_to_img_folder, patient_name)


# In[123]:


path


# In[125]:


import glob


# In[22]:


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


# In[139]:


l = list(listdir_nohidden(path))


# In[140]:


l


# In[179]:




def read_images(path_to_img_folder, patient_name):
    path = os.path.join(path_to_img_folder, patient_name)
    mod_list = list(listdir_nohidden(path))
    filelist = list(listdir_nohidden(os.path.join(path, mod_list[0])))
    filelist = sorted(filelist, key=lambda x: int(x[1:]))
    for file in filelist:
        yield glob.glob(os.path.join(path_to_img_folder, patient_name, mod_list[0], file))[0]
    


# In[180]:


for file in read_image(path_to_img_folder, patient_name):
    print(file)


# In[17]:


path_to_img_folder


# In[170]:


sorted(list(read_image(path_to_img_folder, patient_name)), key=lambda x: int(x[1:]))


# In[19]:


import pydicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[81]:


img = pydicom.dcmread('../../brain-tumor/data/TRAIN_SET/ozerki_annotated_not_segmented/G4/S140/I210').pixel_array


# In[82]:


img.shape


# In[13]:


plt.imshow(img, cmap=cm.gray)


# In[20]:


def images_path(path_to_img_folder, patient_name):
    path = os.path.join(path_to_img_folder, patient_name)
    mod_list = list(listdir_nohidden(path))  # TODO: handle several modalities
    filelist = list(listdir_nohidden(os.path.join(path, mod_list[0])))
    filelist = sorted(filelist, key=lambda x: int(x[1:]))
    for file in filelist:
        yield glob.glob(os.path.join(path_to_img_folder,
                        patient_name, mod_list[0], file))[0]


def read_image(path_to_image_file):
    return pydicom.dcmread(path_to_image_file).pixel_array


def stack_images(path_to_img_folder, patient_name):
    image = []
    for img_path in images_path(path_to_img_folder, patient_name):
        image.append(read_image(img_path))
    return np.array(image)


# In[26]:


import glob
import numpy as np


# In[28]:


stack_images(path_to_img_folder, patient_name)


# In[3]:


# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import re
import os
import numpy as np
import pandas as pd
import pydicom
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


def read_labels(input_file):
    with open(input_file) as f:
        labels = []
        for line in f:
            labels.append(float(line.strip('\n')))
        return labels


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def images_path(path_to_img_folder, patient_name):
    path = os.path.join(path_to_img_folder, patient_name)
    mod_list = list(listdir_nohidden(path))  # TODO: handle several modalities
    filelist = list(listdir_nohidden(os.path.join(path, mod_list[0])))
    filelist = sorted(filelist, key=lambda x: int(x[1:]))
    for file in filelist:
        yield glob.glob(os.path.join(path_to_img_folder,
                        patient_name, mod_list[0], file))[0]


def read_image(path_to_image_file):
    p_array = np.array(pydicom.dcmread(path_to_image_file).pixel_array)
    pil_img = Image.fromarray(p_array)
    pil_img = pil_img.resize((256, 256))
    p_array = np.array(pil_img, dtype='int64')
    return p_array


# TODO: Eliminate of bones
def stack_images(path_to_img_folder, patient_name):
    image = []
    for img_path in images_path(path_to_img_folder, patient_name):
        image.append(read_image(img_path))
    return np.array(image)


class BertFeaturesDataset(Dataset):
    """
    Parameters
    ----------
    input_file : str
        Path to input file with strings
    bert_model : str
        Bert pre-trained model selected in the list: bert-base-uncased,
        bert-large-uncased, bert-base-cased, bert-base-multilingual,
        bert-base-chinese.
    """

    def __init__(self, imgs_folder, input_text_file, labels_file, bert_model,
                 max_seq_length=256, batch_size=4):

        self.imgs_folder = imgs_folder
        self.input_file = input_text_file
        self.labels_file = labels_file
        self.bert_model = bert_model
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        self.tensor_dataset = self.init_bert_dataset(
            self.input_file, self.labels_file,
            self.bert_model, self.max_seq_length
        )
        self.dataset = self.get_bert_embeddings(self.tensor_dataset,
                                                self.bert_model,
                                                self.batch_size)
        df = pd.read_csv(self.labels_file, header=None, dtype={0: str, 1: int})
        self.labels = df.iloc[:, 1].values
        self.names = df.iloc[:, 0].to_list()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = stack_images(self.imgs_folder, name)

        sample = {
            'image': img,
            'embedding': self.dataset[idx],
            'label': self.labels[idx],
            'name': self.names[idx]
        }
        return sample

    def init_bert_dataset(self, input_file, labels_file, bert_model,
                          max_seq_length):

        tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True
        )
        examples = read_examples(input_file)
        features = convert_examples_to_features(
            examples=examples, seq_length=max_seq_length, tokenizer=tokenizer
        )

        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_example_index = torch.arange(
            self.all_input_ids.size(0), dtype=torch.long
        )
        return TensorDataset(self.all_input_ids,
                             self.all_input_mask,
                             self.all_example_index)

    def get_bert_embeddings(self, tensor_dataset, bert_model, batch_size):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        model = BertModel.from_pretrained(bert_model)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler,
                                batch_size=self.batch_size)

        embedding_dataset = []
        with torch.no_grad():
            for inp_id, inp_mask, ex_index in dataloader:
                layers, _ = model(inp_id, token_type_ids=None,
                                  attention_mask=inp_mask)
                stack_last_4_layers = torch.stack(layers[-4:])
                embedding = torch.sum(stack_last_4_layers, dim=0)
                embedding_dataset.append(embedding)

        return TensorDataset(torch.cat(embedding_dataset, dim=0))


# In[4]:


path_to_img_folder = '../../brain-tumor/data/TRAIN_SET/ozerki_annotated_not_segmented/'


# In[5]:


def stack_images(path_to_img_folder, patient_name):
    image = []
    for img_path in images_path(path_to_img_folder, patient_name):
        image.append(read_image(img_path))
    return np.array(image)


# In[ ]:


plt.imshow(imgs[160])


# In[5]:


bert = BertFeaturesDataset(path_to_img_folder, 'annotations_eng.txt', 'brain-labels.csv', 'bert-base-uncased')


# In[4]:


import glob
import numpy as np
from torch.utils.data import SequentialSampler
sampler = SequentialSampler(bert)


# In[5]:


dl = DataLoader(bert, sampler=sampler, batch_size=1)


# In[114]:


for i, batch in enumerate(dl):
    print(i)
    print(batch['image'].shape)


# In[6]:


import matplotlib.pyplot as plt
img = bert[2]['image'][160,:,:]


# In[9]:


im = np.array(img, dtype='uint16')


# In[ ]:


plt.imshow(bert[2]['image'][160,:,:])


# In[1]:


plt.imshow(img)


# In[10]:


import PIL
from PIL import Image


# In[11]:


im = Image.fromarray(im)


# In[12]:


im


# In[100]:


np.array(im.resize((256,256)), dtype='int64')


# In[ ]:




