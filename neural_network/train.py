import torch
from torch.utils.data import DataLoader
import math
from graph_transformer import GraphAwareTransformer
from neural_network.data.read_data import GraphSLIDataset
import torch.nn as nn
def train():
    # 超参数配置
    config = {
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 20,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'save_path': 'best_model.pth'
    }

    # 初始化数据集
    train_dataset = GraphSLIDataset(
        sli_file="../data/training_data.json",
        graph_file="../data/k3_graph.json",
        max_seq_length=512
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, train_dataset.tokenizer)
    )

    # 初始化模型
    model = GraphAwareTransformer(
        vocab_size=len(train_dataset.tokenizer.vocab),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    )
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()  # 根据任务需求可以换成CrossEntropyLoss

    # 训练循环
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据转移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播 [seq_len, batch_size]
            outputs = model(
                input_ids.transpose(0, 1),  # Transformer需要[seq_len, batch_size]
                graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(device),
                src_key_padding_mask=(attention_mask == 0)
            )
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}')
        
        # 保存最佳模型
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config['save_path'])
        
        print(f'Epoch {epoch} | Average Loss: {avg_loss:.4f}')

def collate_fn(batch, tokenizer):
    """自定义批次处理函数"""
    max_len = max(len(item['input_ids']) for item in batch)
    
    return {
        'input_ids': torch.stack([
            torch.nn.functional.pad(
                item['input_ids'],
                (0, max_len - len(item['input_ids'])),
                value=tokenizer.special_tokens['[PAD]']
            ) for item in batch
        ]),
        'attention_mask': torch.stack([
            torch.nn.functional.pad(
                item['attention_mask'],
                (0, max_len - len(item['attention_mask'])),
                value=0
            ) for item in batch
        ]),
        'graph_mask': torch.stack([
            torch.nn.functional.pad(
                item['graph_mask'],
                (0, max_len - len(item['graph_mask'])),
                value=0
            ) for item in batch
        ]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'reward': torch.stack([item['reward'] for item in batch])
    }

def generate_square_subsequent_mask(sz):
    """生成Transformer的因果掩码"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == "__main__":
    train()