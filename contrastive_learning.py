# -*- coding: utf-8 -*-

import functools
from tqdm import tqdm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel

# Load the files
FILENAME_ECB = "data/ecb_data_preprocessed.csv"
FILENAME_FED = "data/fed_data_preprocessed.csv"

ecb = pd.read_csv(FILENAME_ECB, index_col=0)
fed = pd.read_csv(FILENAME_FED, index_col=0)

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config file for parameters
class Arguments():
  def __init__(self):
    self.model_name_or_path = 'bert-base-uncased'
    self.max_seq_length = 512
    self.learning_rate = 3e-5 
    self.adam_epsilon = 1e-8
    self.warmup_proportion = 0.1
    self.weight_decay = 0.01
    self.num_train_epochs = 1
    self.gradient_accumulation_steps = 1
    self.pad_to_max_length = True
    self.batch_size = 2
    self.output_dir = 'model_outputs'
    self.overwrite = True
    self.local_rank = -1
    self.no_cuda = False

args = Arguments()

# Define the model
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

        
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        else:
            raise NotImplementedError

class PoolerArguments():
    def __init__(self): 
        self.temp = 0.05 # Temperature for softmax.
        self.pooler_type = 'cls' #What kind of pooler to use 
        # Number of sentences in one instance
        # 2: pair instance; 3: pair instance with a hard negative
        self.num_sent = 2

pooler_args = PoolerArguments()


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = pooler_args.pooler_type
    cls.pooler = Pooler(pooler_args.pooler_type)
    cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=pooler_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    #import ipdb; ipdb.set_trace();
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = int(input_ids.size(0)/2)
    

    # mlm_outputs = None
    # Flatten input for encoding
    # input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    # attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    # if token_type_ids is not None:
    #     token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False if pooler_args.pooler_type == 'cls' else True,
        return_dict=True,
    )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, pooler_args.num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )

# Custom model for Constrastive Learning task
class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config):
        super().__init__(config)
        #self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config)
        cl_init(self, config)
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

# Custom CSE Dataset
class CSEDataset(Dataset):
    def __init__(self, ecb, fed):
        self.ecb = ecb
        self.fed = fed    

        # Get the preprocessed text
        ecb_texts = list(self.ecb["text_"].values)
        fed_texts = list(self.fed["text_"].values)
        
        data_ecb = list(zip(ecb_texts,ecb_texts))
        data_fed = list(zip(fed_texts,fed_texts))

        self.data =  data_ecb + data_fed
        
    def __len__(self):
      return len(self.data)
    def __getitem__(self,idx):
      return self.data[idx]

def process_batch(txt_list, tokenizer, token_max_length = 512):
    source_ls = [source for source,target in txt_list]
    target_ls = [target for source,target in txt_list]
    source_tokens = tokenizer(source_ls, truncation=True, padding="max_length", max_length=token_max_length, 
                              return_token_type_ids = True)
    target_tokens = tokenizer(target_ls, truncation=True, padding="max_length", max_length=token_max_length,
                              return_token_type_ids = True)
    input_ids = []
    attention_mask = []
    token_type_ids = []
    for i in range(len(source_tokens["input_ids"])):
        input_ids.append(source_tokens["input_ids"][i])
        input_ids.append(target_tokens["input_ids"][i])
        attention_mask.append(source_tokens["attention_mask"][i])
        attention_mask.append(target_tokens["attention_mask"][i])
        token_type_ids.append(source_tokens["token_type_ids"][i])
        token_type_ids.append(target_tokens["token_type_ids"][i])
    return torch.tensor(input_ids),torch.tensor(attention_mask),torch.tensor(token_type_ids)

def get_train_dataloader(train_dataset, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    model_collate_fn = functools.partial(
      process_batch,
      tokenizer=tokenizer,
      token_max_length = args.max_seq_length
      )
    train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=train_sampler,
                                collate_fn=model_collate_fn)
    return train_dataloader

# Instantiate the model and tokenizer
from transformers import AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForCL.from_pretrained("bert-base-uncased").to(device)


# Instantiate the dataset and dataloader
train_data = CSEDataset(ecb=ecb, fed=fed)

train_dataloader = get_train_dataloader(train_data, tokenizer)


# Prepare the optimizer and scheduler
num_train_optimization_steps = int(len(train_data) / args.batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

param_optimizer = list(model.named_parameters())
no_decay = ['bias','LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_optimization_steps)


# Training of the model
for epoch in range(args.num_train_epochs):
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    #for input_ids,attention_mask,token_type_ids in train_dataloader:
      batch = tuple(t.to(device) for t in batch)
      input_ids,attention_mask,token_type_ids = batch
      #import ipdb; ipdb.set_trace();
      # zero the parameter gradients
      optimizer.zero_grad()
      outputs = model(input_ids,attention_mask,token_type_ids)
      loss = outputs["loss"]

      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
      loss.backward()
      running_loss += loss.item()
      if (step + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
      print("Step Loss", loss.item())
    print("Epoch Loss", running_loss)


# Save the model 
torch.save(model.state_dict(), "bert_simsce.pt")
