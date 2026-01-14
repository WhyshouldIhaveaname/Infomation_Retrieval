import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data'))
import gzip
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss  # 必须安装: pip install faiss-cpu

# ==========================================
# 1. 配置参数 (针对 3060 Laptop 6GB 优化)
# ==========================================
class Config:
    # 示例文件名 (请根据实际下载的文件名修改)
    # 2023版通常后缀是 .jsonl.gz
    DATA_PATH = 'Health_and_Personal_Care.jsonl.gz'       
    META_PATH = 'meta_Health_and_Personal_Care.jsonl.gz'    
    
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    
    MAX_LEN = 64  
    HISTORY_LEN = 10 
    BATCH_SIZE = 64
    EPOCHS = 10     
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    EMBED_DIM = 384
    RANK = 32         
    TEMPERATURE = 0.05

config = Config()
print(f"🚀 Device: {config.DEVICE} | GPU Mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else "CPU")

# ==========================================
# 2. 数据处理 (序列化)
# ==========================================
# ==========================================
# 通用 Metadata 处理工具函数 (新增)
# ==========================================
def format_metadata_to_text(meta_item, max_len_chars=300):
    """
    针对 Amazon Reviews 2023 数据集的文本序列化函数 (修复 NoneType 报错版)
    """
    parts = []
    
    # 辅助函数：安全获取字符串，处理 None 和 'nan'
    def safe_get(key):
        val = meta_item.get(key)
        if val is None: return ""
        val = str(val).strip()
        if val.lower() == 'nan': return ""
        return val

    # 1. 核心字段 (优先级最高)
    title = safe_get('title')
    if title:
        parts.append(f"Title: {title}")

    # 2. 品牌/店铺信息
    store = safe_get('store')
    if store:
        parts.append(f"Brand: {store}")

    # 3. 主要分类
    cat = safe_get('main_category')
    if cat and cat.lower() != 'all categories':
        parts.append(f"Category: {cat}")

    # 4. 详细参数 (Details)
    details = meta_item.get('details')
    if isinstance(details, dict):
        # 挑选一些通用的高价值 Key
        valid_keys = ['author', 'artist', 'brand', 'format', 'color', 'genre', 'label']
        for k, v in details.items():
            if not k or not v: continue # 跳过空键值
            k_lower = str(k).lower()
            # 模糊匹配 key
            if any(vk in k_lower for vk in valid_keys):
                seg = f"{k}: {str(v).strip()}"
                if len(parts) < 8: # 防止太长
                    parts.append(seg)
    
    # 5. 特性列表 (Features)
    features = meta_item.get('features')
    if isinstance(features, list) and features:
        # 只取前 2 个特性
        count = 0
        for feat in features:
            if count >= 2: break
            if feat:
                feat_str = str(feat).strip()
                if feat_str:
                    parts.append(f"Feature: {feat_str}")
                    count += 1

    # 6. 描述 (Description)
    desc = meta_item.get('description')
    desc_text = ""
    if isinstance(desc, list) and len(desc) > 0:
        desc_text = str(desc[0])
    elif isinstance(desc, str):
        desc_text = desc
    
    if desc_text:
        # 简单截断
        clean_desc = desc_text.strip()[:100]
        if clean_desc:
            parts.append(f"Desc: {clean_desc}...")

    # 7. 智能拼接与截断
    final_text = ""
    for part in parts:
        # 预估添加后的长度
        if len(final_text) + len(part) > max_len_chars:
            # 如果 Title 还没加进去，硬塞
            if "Title:" in part and "Title:" not in final_text:
                remaining = max_len_chars - len(final_text)
                if remaining > 10:
                    final_text += part[:remaining]
            break
        final_text += part + " ; "
    
    return final_text.strip(" ; ")

def load_and_process_data(review_path, meta_path, limit=None):
    print(f"Loading 2023 Dataset...")
    print(f"Meta: {meta_path}")
    print(f"Review: {review_path}")
    
    # --- 1. 加载 Metadata ---
    # 2023 版 Key: parent_asin
    asin2text = {} 
    
    meta_count = 0
    with gzip.open(meta_path, 'r') as f:
        for l in tqdm(f, desc="Reading Meta"):
            try:
                # 2023 数据集是标准的 JSONL，直接 json.loads 即可，不需要 eval
                line = json.loads(l.strip())
                
                # 使用 parent_asin 作为唯一标识 (聚合变体)
                # 如果没有 parent_asin，尝试用 asin
                item_id = line.get('parent_asin', line.get('asin'))
                
                if not item_id: continue
                
                processed_text = format_metadata_to_text(line)
                if processed_text:
                    asin2text[item_id] = processed_text
                    meta_count += 1
            except json.JSONDecodeError:
                continue

    print(f"✅ Loaded {len(asin2text)} items metadata.")

    # --- 2. 加载 Review 数据 ---
    # 2023 版 Key: user_id, parent_asin, timestamp
    data = []
    hit_meta_count = 0 
    
    with gzip.open(review_path, 'r') as f:
        for i, l in enumerate(tqdm(f, desc="Reading Reviews")):
            if limit and i >= limit: break
            try:
                line = json.loads(l.strip())
                
                # ID 映射
                user_id = line.get('user_id')
                item_id = line.get('parent_asin', line.get('asin'))
                timestamp = line.get('timestamp', 0)
                
                if not user_id or not item_id: continue

                # 优先使用 Meta
                if item_id in asin2text:
                    final_text = asin2text[item_id]
                    hit_meta_count += 1
                else:
                    # 兜底：使用 Review 里的 title
                    # 2023 版 review 里有 'title' 和 'text'
                    parts = []
                    if 'title' in line: parts.append(line['title'])
                    if 'text' in line: parts.append(line['text'][:100])
                    final_text = " ".join(parts)
                
                if len(final_text) < 5: continue

                data.append({
                    'user': user_id,
                    'item': item_id,
                    'text': final_text, 
                    'time': timestamp
                })
            except json.JSONDecodeError:
                continue
    
    hit_rate = hit_meta_count / len(data) if len(data) > 0 else 0
    print(f"📊 Metadata Hit Rate: {hit_rate:.2%}")
    
    df = pd.DataFrame(data)
    # 2023 timestamp 可能是毫秒，排序逻辑不变
    df = df.sort_values(['user', 'time'])
    return df
# 加载数据 (限制 50k 条用于演示，跑全量可去掉 limit)
full_df = load_and_process_data(config.DATA_PATH, config.META_PATH, limit=100000)

print("构建 Item 映射...")
item_list = full_df['item'].unique()
item_map = {asin: i for i, asin in enumerate(item_list)}
id2item_text = {i: text for asin, text, i in zip(full_df['item'], full_df['text'], [item_map[a] for a in full_df['item']])}
# 这是一个简化的 text map，实际上一个 asin 可能有多条评论，这里随机取了一条作为该物品的代表文本
# 生产环境中应该建立专门的 Item Meta 表

full_df['item_idx'] = full_df['item'].map(item_map)
NUM_ITEMS = len(item_list)
print(f"Total Interactions: {len(full_df)}, Total Items: {NUM_ITEMS}")

# 构建序列数据
# 格式: ([history_item_texts], target_item_text)
train_samples = []
test_samples = []

print("构建用户历史序列...")
user_groups = full_df.groupby('user')
for uid, group in tqdm(user_groups):
    if len(group) < 3: continue # 交互太少无法构建序列
    
    items = group['item_idx'].tolist()
    texts = group['text'].tolist()
    
    # 简单的 Leave-one-out 划分
    # 倒数第1个是测试集目标，倒数第2个是测试集 Seed，倒数 2-N 是测试集历史
    
    # --- 构建测试样本 ---
    # XPERT 逻辑：Seed Event 是触发检索的事件。
    # 这里我们定义：Input History 用于生成 Morph，Last Item in History 作为 Seed Event
    
    # 测试集：用过去的所有数据预测最后一个
    if len(items) > config.HISTORY_LEN:
        hist_texts = texts[-(config.HISTORY_LEN+1):-1] # 取倒数第2个往前推N个
        target_text = texts[-1]
        target_item_idx = items[-1]
        test_samples.append((hist_texts, target_text, target_item_idx))
    
    # 训练集：滑动窗口
    # 比如序列 A, B, C, D, E
    # (A,B)->C, (A,B,C)->D...
    for i in range(1, len(items)-1):
        # 窗口截止到 i
        start = max(0, i - config.HISTORY_LEN)
        hist_window = texts[start:i+1] # 包含 seed event (第 i 个)
        target = texts[i+1]
        train_samples.append((hist_window, target))

print(f"Train Samples: {len(train_samples)}, Test Samples: {len(test_samples)}")

# ==========================================
# 3. Dataset & DataLoader
# ==========================================
class SeqRecDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len, is_test=False):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            hist_texts, target_text, target_idx = self.samples[idx]
        else:
            hist_texts, target_text = self.samples[idx]
            target_idx = -1

        # Seed Event 是历史里的最后一个
        seed_text = hist_texts[-1]
        # 用于生成 User Preference 的是历史 (这里简单起见，把 seed 也放进 history 编码)
        context_text = " [SEP] ".join(hist_texts) 

        return context_text, seed_text, target_text, target_idx

def collate_fn(batch):
    context_list, seed_list, target_list, idx_list = zip(*batch)
    
    # 统一 Tokenize
    def tokenize(text_list):
        return tokenizer(
            list(text_list), 
            padding='max_length', truncation=True, max_length=config.MAX_LEN, return_tensors='pt'
        )
    
    return {
        'context': tokenize(context_list), # 用于 LSTM 生成 Morph
        'seed': tokenize(seed_list),       # 用于被 Morph 作用 (基准点)
        'target': tokenize(target_list),   # 正样本
        'target_idx': torch.tensor(idx_list)
    }

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
train_dataset = SeqRecDataset(train_samples, tokenizer, config.MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0) # Windows下设0

# ==========================================
# 4. Pro版 模型定义 (Low-Rank + LSTM)
# ==========================================
class XpertPro(nn.Module):
    def __init__(self, model_name, embed_dim, rank, hidden_dim=256):
        super(XpertPro, self).__init__()
        
        # 1. 文本编码器 (Freeze 冻结以节省显存)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        #设为False时冻结参数
        for param in self.text_encoder.parameters():
            param.requires_grad = True
            
        # 2. 偏好提取器 (LSTM)
        # 输入是 Generic Text Embedding, 输出是 User State
        self.preference_rnn = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # 3. 低秩矩阵生成器 (Low-Rank Generator)
        # 生成矩阵 A (D x r) 和 B (D x r)
        # 输出维度 = D * r
        self.head_A = nn.Linear(hidden_dim, embed_dim * rank)
        self.head_B = nn.Linear(hidden_dim, embed_dim * rank)
        
        self.embed_dim = embed_dim
        self.rank = rank

    def get_generic_embedding(self, inputs):
        # 仅推理，不计算梯度
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
        # Mean Pooling
        emb = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(emb, p=2, dim=1)

    def generate_morph_operators(self, context_emb):
        # context_emb: [batch, dim] (这里简化了，直接把 concat 的 history 作为一个 embedding 喂给 LSTM 的一步)
        # 如果追求更精细，应该把 history 分开 tokenize，得到 [batch, seq, dim]，然后喂给 LSTM
        # 为了速度和显存，这里输入是 [batch, 1, dim]
        
        _, (h_n, _) = self.preference_rnn(context_emb.unsqueeze(1))
        user_state = h_n.squeeze(0) # [batch, hidden_dim]
        
        # 生成 A 和 B
        batch_size = user_state.size(0)
        
        # A: [batch, D, r]
        mat_A = self.head_A(user_state).view(batch_size, self.embed_dim, self.rank)
        # B: [batch, D, r]
        mat_B = self.head_B(user_state).view(batch_size, self.embed_dim, self.rank)
        
        return mat_A, mat_B

    def forward(self, context_inputs, seed_inputs, target_inputs=None):
        # 1. 获取所有 Generic Embeddings
        # context: 用户历史文本拼接
        # seed: 触发检索的那个物品
        # target: 真实点击的下一个物品
        
        context_emb = self.get_generic_embedding(context_inputs)
        seed_emb = self.get_generic_embedding(seed_inputs)
        
        # 2. 生成 Low-Rank Morph Operators
        # P_u = I + A @ B.T
        mat_A, mat_B = self.generate_morph_operators(context_emb)
        
        # 3. 应用 Morph Operator (核心优化)
        # 我们需要计算 v_pers = P_u @ v_seed
        # v_pers = (I + A @ B.T) @ v_seed = v_seed + A @ (B.T @ v_seed)
        # 这样计算复杂度从 O(D^2) 降到 O(D*r)
        
        # seed_emb: [batch, D, 1]
        v_seed = seed_emb.unsqueeze(2) 
        
        # step 1: temp = B.T @ v_seed -> [batch, r, 1]
        temp = torch.bmm(mat_B.transpose(1, 2), v_seed)
        
        # step 2: delta = A @ temp -> [batch, D, 1]
        delta = torch.bmm(mat_A, temp).squeeze(2)
        
        # step 3: res = v_seed + delta
        personalized_query = F.normalize(seed_emb + delta, p=2, dim=1)
        
        if target_inputs is not None:
            target_emb = self.get_generic_embedding(target_inputs)
            return personalized_query, target_emb
        else:
            return personalized_query

model = XpertPro(config.MODEL_NAME, config.EMBED_DIM, config.RANK)
model.to(config.DEVICE)

# ==========================================
# 5. 训练 (Training)
# ==========================================
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR)
scaler = torch.cuda.amp.GradScaler()

print(">>> Start Training (Freeze BERT, Train Adapter Only)...")

for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        # Move to device
        ctx = {k: v.to(config.DEVICE) for k, v in batch['context'].items()}
        seed = {k: v.to(config.DEVICE) for k, v in batch['seed'].items()}
        tgt = {k: v.to(config.DEVICE) for k, v in batch['target'].items()}
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            # query: 个性化后的向量, key: 目标物品的通用向量
            query_emb, key_emb = model(ctx, seed, tgt)
            
            # InfoNCE Loss (Contrastive)
            # 同样 batch 内的其他 target 作为负样本
            logits = torch.matmul(query_emb, key_emb.T) / config.TEMPERATURE
            labels = torch.arange(logits.size(0)).long().to(config.DEVICE)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

# ==========================================
# 6. FAISS 全量评估 (Recall@K)
# ==========================================
print("\n>>> Building Index for Evaluation (Recall@50)...")
model.eval()

# 6.1 计算所有 Item 的 Embedding (Generic) 建立索引
all_item_texts = [id2item_text[i] for i in range(NUM_ITEMS)]
item_embs = []

# 分批计算 Item Embedding
BATCH_EVAL = 128
with torch.no_grad():
    for i in tqdm(range(0, NUM_ITEMS, BATCH_EVAL), desc="Encoding Items"):
        batch_texts = all_item_texts[i : i + BATCH_EVAL]
        inputs = tokenizer(batch_texts, padding='max_length', truncation=True, max_length=config.MAX_LEN, return_tensors='pt').to(config.DEVICE)
        emb = model.get_generic_embedding(inputs)
        item_embs.append(emb.cpu().numpy())

item_matrix = np.concatenate(item_embs, axis=0) # [Num_Items, 384]

# 6.2 建立 FAISS 索引 (Inner Product)
index = faiss.IndexFlatIP(config.EMBED_DIM)
index.add(item_matrix)
print(f"FAISS Index Built: {index.ntotal} items.")

# 6.3 计算测试集用户的 Personalized Query
test_dataset = SeqRecDataset(test_samples, tokenizer, config.MAX_LEN, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

hits_10 = 0
hits_50 = 0
total_test = 0

print(">>> Running Retrieval Evaluation...")
with torch.no_grad():
    for batch in tqdm(test_loader):
        ctx = {k: v.to(config.DEVICE) for k, v in batch['context'].items()}
        seed = {k: v.to(config.DEVICE) for k, v in batch['seed'].items()}
        target_indices = batch['target_idx'].numpy()
        
        # 生成个性化查询向量
        query_vecs = model(ctx, seed).cpu().numpy() # [batch, 384]
        
        # 搜索 Top-K
        D, I = index.search(query_vecs, 50) # I: [batch, 50]
        
        # 计算 Hit
        for rank, target_idx in zip(I, target_indices):
            if target_idx in rank[:10]:
                hits_10 += 1
            if target_idx in rank[:50]:
                hits_50 += 1
        
        total_test += len(target_indices)

print("="*40)
print(f"📊 Final Evaluation Results (Test Set Size: {total_test})")
print(f"Recall@10: {hits_10 / total_test:.4f}")
print(f"Recall@50: {hits_50 / total_test:.4f}")
print("="*40)

# ==========================================
# 7. 案例展示 (Qualitative)
# ==========================================
print("\n>>> Showing a Personalized Case...")
# 找一个测试样本
sample_idx = 0
sample_ctx, sample_seed, sample_target, _ = test_dataset[sample_idx]

print(f"👤 User History Context: {sample_ctx[:80]}...")
print(f"🌱 Seed Item: {sample_seed[:50]}...")
print(f"🎯 True Target: {sample_target[:50]}...")

# 模拟推理
inputs_ctx = tokenizer([sample_ctx], padding='max_length',truncation=True, max_length=config.MAX_LEN, return_tensors='pt').to(config.DEVICE)
inputs_seed = tokenizer([sample_seed], padding='max_length', max_length=config.MAX_LEN, return_tensors='pt').to(config.DEVICE)

with torch.no_grad():
    # 1. 通用向量检索结果
    gen_emb = model.get_generic_embedding(inputs_seed).cpu().numpy()
    _, I_gen = index.search(gen_emb, 5)
    print("\n[Generic Retrieval (Without Morph)]:")
    for idx in I_gen[0]:
        print(f" - {id2item_text[idx][:60]}")
        
    # 2. 个性化向量检索结果
    pers_emb = model(inputs_ctx, inputs_seed).cpu().numpy()
    _, I_pers = index.search(pers_emb, 5)
    print("\n[XPERT Retrieval (With Low-Rank Morph)]:")
    for idx in I_pers[0]:
        print(f" - {id2item_text[idx][:60]}")