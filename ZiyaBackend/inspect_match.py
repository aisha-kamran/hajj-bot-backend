import json
import re
from sentence_transformers import SentenceTransformer, util
import torch

def normalize_urdu(text):
    text = text or ''
    text = text.replace('ي','ی').replace('ے','ی').replace('ة','ہ').replace('ه','ہ')
    text = re.sub(r'[\u064B-\u0652]','', text)
    return re.sub(r'\s+',' ', text).strip()

with open('product_files/product_catalog.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
model = SentenceTransformer('intfloat/multilingual-e5-small')
questions = [normalize_urdu(item.get('Question','')) for item in data]
emb = model.encode(questions, convert_to_tensor=True)
query='سعی کہاں ہوتی ہے؟'
qnorm = normalize_urdu(query)
qemb = model.encode(qnorm, convert_to_tensor=True)
scores = util.cos_sim(qemb, emb)[0]
vals, idxs = torch.topk(scores, k=min(10, len(data)))
print('TOP MATCHES:')
for score, idx in zip(vals, idxs):
    item = data[int(idx)]
    print(f'{float(score):.6f} | Q: {item.get("Question")}')
print('\nKEYWORD HITS:')
for item in data:
    if 'سعی' in item.get('Question','') or 'سعی' in item.get('Answer',''):
        print(item.get('Question'))
