from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import sys
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForSequenceClassification, DataCollatorWithPadding

from torch.utils.data import DataLoader


is_random = sys.argv[1]

if is_random == "True":
    is_random = True
else:
    is_random = False

layer = int(sys.argv[2])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_dataset = load_dataset("stanfordnlp/sst2")["validation"]

def filter_short(example):
    return len(example['sentence'].split()) > 10

test_dataset = test_dataset.filter(filter_short)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('../../../models/bert-sentiment')
model.eval()
model.to(device)
collator = DataCollatorWithPadding(tokenizer = tokenizer)

def tokenize(example):
    return tokenizer(example['sentence'], truncation = True)

test_dataset = test_dataset.map(tokenize, batched = True)
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])

def evaluate(model, dataset):
    model.eval()
    predictions = []
    true_labels = []
    
    dataloader = DataLoader(dataset, batch_size = 1, collate_fn = collator)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask
            )
            
            logits = outputs.logits #(B, 2)
            preds = torch.argmax(logits, dim = -1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"{accuracy:.4f}")
    
def add_hooks(model, layer):
    
    def remove_anchors(batched_embeddings):
        #embeddings are (B, seq_len, dim)
        
        #let me try to vectorize this
        
        
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
        
            anchor_mask = r_squared < -0.5
            
            num_anchors = anchor_mask.sum()
            
            embeddings[anchor_mask] = torch.zeros_like(embeddings[anchor_mask])
            
            valid_indices = torch.where(~anchor_mask)[0]
            
            random_indices = valid_indices[torch.randperm(len(valid_indices))[:num_anchors]]
            
            random_embeddings[random_indices] = torch.zeros_like(random_embeddings[random_indices])
            
            full_embeddings.append(embeddings)
            randomly_flipped_embeddings.append(random_embeddings)
        
              
        
        full_embeddings = torch.stack(full_embeddings)
        randomly_flipped_embeddings = torch.stack(randomly_flipped_embeddings)

        if is_random:
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

handle = add_hooks(model, layer)

evaluate(model, test_dataset)

