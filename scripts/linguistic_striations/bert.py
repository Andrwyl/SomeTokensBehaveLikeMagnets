import torch
from transformers import AutoTokenizer, BertModel, BertForTokenClassification
import numpy as np
import re
import matplotlib.pyplot as plt
import json
from scipy.stats import pointbiserialr
from tqdm import tqdm

def get_embeddings(paragraph, layer, tokenizer, model):
    inputs = tokenizer(paragraph, return_tensors='pt', is_split_into_words = True, truncation = True)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    token_embeddings = outputs.hidden_states[layer].squeeze(0)
    return token_embeddings

def candidate_anchor_vectors(embeddings):
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
       
    return r_squared.tolist()

def collect_and_align(word_ids, scores):
    
    assert(len(word_ids) == len(scores))
    
    collected_scores = []
    
    prev_id = None
    for id, score in zip(word_ids, scores):
        if id != prev_id:
            collected_scores.append(score)
        
        prev_id = id
    
    return collected_scores

def get_anchor_strengths(paragraph, layer, tokenizer, model):
    
    
    #get the anchor scores
    embs = get_embeddings(paragraph, layer, tokenizer, model)
    anchor_strengths = candidate_anchor_vectors(embs)[1:-1]
    anchor_scores = anchor_strengths #[0 if strength < 0 else 1 for strength in anchor_strengths]
    
    encodings = tokenizer(paragraph, return_tensors='pt', is_split_into_words = True, truncation = True)
    word_ids = encodings.word_ids()[1:-1]
    
    assert(len(anchor_scores) == len(word_ids))
    collected_strengths = collect_and_align(word_ids, anchor_scores)
    return collected_strengths

def get_pos_tags(paragraph, tokenizer, model):
    
    class_names = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
            'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
            'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
            'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
            'WP': 44, 'WP$': 45, 'WRB': 46}
    class_names = {v:k for k,v in class_names.items()}
    
    content_tags = {
    'NN', 'NNS', 'NNP', 'NNPS', 'NN|SYM',   # nouns
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # verbs
    'JJ', 'JJR', 'JJS',                     # adjectives
    'RB', 'RBR', 'RBS',                    # adverbs
    }
    
    tagset = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    tag2id = {tag:i for i, tag in enumerate(tagset)}
    id2tag = {i:tag for tag, i in tag2id.items()}    

    inputs = tokenizer(paragraph, return_tensors='pt', is_split_into_words = True, truncation = True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        logits = outputs.logits.squeeze() #(seq_len, classes)
        preds = torch.argmax(logits, dim = -1).cpu().numpy() #(seq_len, )
        
        preds = [id2tag[pred] for pred in preds]
        preds = [class_names[pred] for pred in preds]
        
        preds = [1 if pred in content_tags else 0 for pred in preds]
    
    collected_POS = collect_and_align(inputs.word_ids()[1:-1], preds[1:-1])
    return collected_POS



tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased')
bert_model.eval()
POS_model = BertForTokenClassification.from_pretrained('../../models/bert-pos')
POS_model.eval()

scores = {}

with open('../../datasets/paragraphs.txt', 'r') as f:
    paragraphs = f.read().splitlines()
    
for layer in tqdm(range(bert_model.config.num_hidden_layers)):
    for paragraph in tqdm(paragraphs):
        paragraph = paragraph.strip()
        paragraph = paragraph.replace("’", "'")
        paragraph = paragraph.replace('“', '"')
        paragraph = paragraph.replace('”', '"')
        paragraph = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", paragraph)
        anchor_polarity = get_anchor_strengths(paragraph, layer, tokenizer, bert_model)
        pos_tags = get_pos_tags(paragraph, tokenizer, POS_model)   

        #score, _ = pointbiserialr(anchor_polarity, pos_tags)
        assert len(anchor_polarity) == len(pos_tags)

        if layer not in scores:
            scores[layer] = [anchor_polarity, pos_tags]
        else:
            scores[layer][0].extend(anchor_polarity)
            scores[layer][1].extend(pos_tags)


r_values = []
for layer in range(bert_model.config.num_hidden_layers):
    r_values.append(pointbiserialr(scores[layer][0], scores[layer][1])[0])

with open("results/bert.json", "w") as f:
    json.dump(scores, f)


plt.plot(np.arange(bert_model.config.num_hidden_layers), r_values)
plt.xlabel("Layers")
plt.ylabel("PB Correlation")
plt.title("Repelling Magnets Track Function Words in Early Layers")
plt.savefig("results/bert.pdf")

