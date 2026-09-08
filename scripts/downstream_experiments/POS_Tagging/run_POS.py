from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from torch.utils.data import DataLoader
import torch
from transformers import Trainer, TrainingArguments, BertTokenizerFast, BertForTokenClassification, DataCollatorForTokenClassification
from scipy.stats import linregress
import json
import sys


random = sys.argv[1]
if random == "True":
    random = True
else:
    random = False

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = load_dataset("eriktks/conll2003")['train']
dataset = load_dataset("eriktks/conll2003")['test']

train_sents = train_dataset["tokens"]
train_tags = train_dataset["pos_tags"]

test_sents = dataset["tokens"]
test_tags = dataset["pos_tags"]

tagset = sorted({tag for sent in train_tags for tag in sent})

tag2id = {tag:i for i, tag in enumerate(tagset)}
id2tag = {i:tag for tag, i in tag2id.items()}


num_labels = len(tagset)

def pack_sentences(sents, tags):
    max_length = 200
    packed_sentences = []
    packed_tags = []
    current_words = []
    current_tags = []
    
    for sent_words, sent_tags in zip(sents, tags):
        if len(current_words) + len(sent_words) < max_length:
            current_words.extend(sent_words)
            current_tags.extend(sent_tags)
        
        else:
            packed_sentences.append(current_words)
            packed_tags.append(current_tags)
            
            current_words = list(sent_words)
            current_tags = list(sent_tags)
    
    if current_words:
        packed_sentences.append(current_words)
        packed_tags.append(current_tags)
    
    return packed_sentences, packed_tags

sentences, tags = pack_sentences(test_sents, test_tags)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
collator = DataCollatorForTokenClassification(tokenizer = tokenizer)

def tokenize_and_align_labels(sentences, tags):
    encodings = tokenizer(
        sentences,
        is_split_into_words = True,
        truncation = True,
    )
    
    labels = []
    
    for i, label in enumerate(tags):
        word_ids = encodings.word_ids(batch_index = i)
        label_ids = []
        
        previous_word = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            
            elif word_idx != previous_word:
                label_ids.append(tag2id[label[word_idx]])
            
            else:
                label_ids.append(-100)
            
            previous_word = word_idx
        
        labels.append(label_ids)
    
    encodings["labels"] = labels
    return encodings

encodings = tokenize_and_align_labels(sentences, tags)


class CONLLDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
dataset = CONLLDataset(encodings)

def add_hooks(model, layer):
    
    def remove_anchors(batched_embeddings):
        #embeddings are (B, seq_len, dim)
        
        
        full_embeddings = []
        randomly_flipped_embeddings = []
        
        for embeddings in batched_embeddings: #embeddings is (seq_len, dim)
            
            random_embeddings = embeddings.clone()
            embedding_norms = torch.norm(embeddings, p = 2, dim = -1).cpu()

            normed = torch.nn.functional.normalize(embeddings, p = 2, dim = -1) #(seq_len, dim)
            angles_matrix = (normed @ normed.T).cpu() #(seq_len, seq_len)
            
            n = angles_matrix.shape[1] #seq_len
            x = angles_matrix
            y = embedding_norms.unsqueeze(0).expand_as(x)
            
            x_mean = x.mean(dim = 1, keepdim = True) #(seq_len, 1)
            y_mean = embedding_norms.mean()
            
            x_centered = x - x_mean
            y_centered = y - y_mean
            
            cov_xy = (x_centered * y_centered).sum(dim = 1) / (n - 1)
            var_x = (x_centered ** 2).sum(dim = 1) / (n - 1)
            
            slopes = cov_xy / var_x
            intercepts = y_mean - slopes * x_mean.squeeze()
            
            y_pred = slopes.unsqueeze(1) * x + intercepts.unsqueeze(1)
            
            ss_res = ((y - y_pred) ** 2).sum(dim = 1)
            ss_tot = (y_centered ** 2).sum(dim = 1)
            
            r_squared = 1 - ss_res / ss_tot #(seq_len,)
            r_squared = torch.sign(slopes) * r_squared
            
            anchor_mask = r_squared > 0.5
            
            num_anchors = anchor_mask.sum()
            
            embeddings[anchor_mask] = torch.zeros_like(embeddings[anchor_mask])
            
            valid_indices = torch.where(~anchor_mask)[0]
            
            random_indices = valid_indices[torch.randperm(len(valid_indices))[:num_anchors]]
            
            random_embeddings[random_indices] = torch.zeros_like(random_embeddings[random_indices])
            
            full_embeddings.append(embeddings)
            randomly_flipped_embeddings.append(random_embeddings)
        
        full_embeddings = torch.stack(full_embeddings)
        randomly_flipped_embeddings = torch.stack(randomly_flipped_embeddings)
        
        
        if random:
            return randomly_flipped_embeddings
        else:
            return full_embeddings     

    def get_intervention_hook(fn):
        def hook(module, input, output):
            hidden_states = output[0]
            modified_hidden_states = fn(hidden_states)
            
            return (modified_hidden_states,) + output[1:]

        return hook

    handle = model.bert.encoder.layer[layer].register_forward_hook(
        get_intervention_hook(remove_anchors)
    )
    
    return handle  

def evaluate(model, dataset, id2label):
    model.eval()
    predictions = []
    true_labels = []
    
    dataloader = DataLoader(dataset, batch_size = 1, collate_fn = collator)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc = "Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask
            )
            
            logits = outputs.logits #(B, N, C)
            preds = torch.argmax(logits, dim=-1)
            
            for pred, label in zip(preds, labels):
                pred_list = pred.cpu().numpy()
                label_list = label.cpu().numpy()
                
                mask = label_list != -100
                predictions.extend(pred_list[mask])
                true_labels.extend(label_list[mask])
    
    
    print(accuracy_score(true_labels, predictions))

model = BertForTokenClassification.from_pretrained(
    "../../../models/bert-pos"
)

layer = int(sys.argv[2])

model.to(device)
handle = add_hooks(model, layer)
evaluate(model, dataset, id2tag)
