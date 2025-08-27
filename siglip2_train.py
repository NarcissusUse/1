import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoModel, AutoProcessor
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

# --- 引入 torchvision transforms ---
from torchvision import transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 是否使用离线特征缓存进行训练（不改模型结构，只提速）
USE_CACHED_FEATURES = True

# --- Mixup 配置 ---
USE_MIXUP = True  # 是否使用mixup
MIXUP_ALPHA = 0.2  # mixup的alpha参数，控制混合程度

# --- Label Smoothing 配置 ---
USE_LABEL_SMOOTHING = True  # 是否使用label smoothing
LABEL_SMOOTHING = 0.1  # label smoothing参数，通常在0.1-0.3之间

MODEL_NAME = "google/siglip2-giant-opt-patch16-384"
CSV_PATH = "bondee_101008.csv"
IMAGE_DIR = "101008"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 2e-4
VAL_SPLIT_SIZE = 0.1
WARMUP_EPOCHS = 2
MIN_LR = 1e-5
MODEL_SAVE_PATH = "siglip2_classifier_101008_bondee.pth"

print(f"Using device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Classifier LR: {LEARNING_RATE}")
print(f"Using Mixup: {USE_MIXUP}, Alpha: {MIXUP_ALPHA}")
print(f"Using Label Smoothing: {USE_LABEL_SMOOTHING}, Smoothing: {LABEL_SMOOTHING}")

df = pd.read_csv(CSV_PATH)
df['label'] = df['label'].astype(str)

unique_labels = sorted(df['label'].unique())
num_classes = len(unique_labels)
print(f"Found {num_classes} unique classes.")
print("Class distribution in original dataset:")
print(df['label'].value_counts().sort_index())

label_to_id = {label: i for i, label in enumerate(unique_labels)}
id_to_label = {i: label for i, label in enumerate(unique_labels)}
df['encoded_label'] = df['label'].map(label_to_id)

train_df, val_df = train_test_split(
    df, test_size=VAL_SPLIT_SIZE, random_state=42, stratify=df['label']
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

class ImageClassificationDataset(Dataset):
    def __init__(self, dataframe, image_dir, processor):
        self.df = dataframe
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        label = row['encoded_label']
        try:
            image = Image.open(image_path).convert("RGB")
            processed_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = processed_inputs['pixel_values'].squeeze(0)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None
        return pixel_values, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

train_dataset = ImageClassificationDataset(train_df, IMAGE_DIR, processor)
val_dataset = ImageClassificationDataset(val_df, IMAGE_DIR, processor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class Siglip2Classifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(Siglip2Classifier, self).__init__()
        self.vision_encoder = AutoModel.from_pretrained(model_name)

        # 首先冻结所有层
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        embedding_dim = self.vision_encoder.config.vision_config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        with torch.no_grad():
            image_embeddings = self.vision_encoder.get_image_features(pixel_values=pixel_values)
        logits = self.classifier(image_embeddings)
        return logits

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_lr=0, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup阶段：线性增长，从 warmup_lr 到 base_lr
            if self.warmup_epochs == 1:
                lr = self.base_lr
            else:
                progress = (epoch + 1) / self.warmup_epochs
                lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * progress
        else:
            # Cosine decay阶段
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# --- Label Smoothing 损失函数 ---
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# --- Mixup 相关函数 ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    生成mixup后的数据
    Args:
        x: 输入特征 [batch_size, ...]
        y: 标签 [batch_size]
        alpha: Beta分布参数
        device: 设备
    Returns:
        mixed_x: 混合后的特征
        y_a, y_b: 原始标签对
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    计算mixup损失
    Args:
        criterion: 损失函数
        pred: 模型预测 [batch_size, num_classes]
        y_a, y_b: 混合前的两个标签
        lam: 混合系数
    Returns:
        loss: 混合损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def precompute_features(dataloader, vision_encoder, device):
    vision_encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for images, lbs in tqdm(dataloader, desc="Precomputing features"):
            if images is None:
                continue
            images = images.to(device)
            emb = vision_encoder.get_image_features(pixel_values=images)
            feats.append(emb.cpu())
            labels.append(lbs)
    return torch.cat(feats, 0), torch.cat(labels, 0)

model = Siglip2Classifier(MODEL_NAME, num_classes).to(DEVICE)

# 根据配置选择损失函数
if USE_LABEL_SMOOTHING:
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    print(f"Using Label Smoothing CrossEntropy with smoothing={LABEL_SMOOTHING}")
else:
    criterion = nn.CrossEntropyLoss()
    print("Using standard CrossEntropy loss")

# 设置差分学习率：只训练分类头
optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# 创建学习率调度器
scheduler = WarmupCosineScheduler(
    optimizer=optimizer,
    warmup_epochs=WARMUP_EPOCHS,
    total_epochs=EPOCHS,
    warmup_lr=LEARNING_RATE * 0.1,  # warmup从10%的基础学习率开始
    min_lr=MIN_LR
)

# 可选：预提取并缓存特征（仅训练分类头时使用）
if USE_CACHED_FEATURES:
    print("Caching features for training/validation...")
    train_feats, train_lbls = precompute_features(train_loader, model.vision_encoder, DEVICE)
    val_feats, val_lbls = precompute_features(val_loader, model.vision_encoder, DEVICE)

    train_feat_loader = DataLoader(
        TensorDataset(train_feats, train_lbls),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_feat_loader = DataLoader(
        TensorDataset(val_feats, val_lbls),
        batch_size=BATCH_SIZE, shuffle=False
    )

# --- 训练循环 ---
print("Starting training with Fine-tuning, Data Augmentation, Mixup, and Label Smoothing...")
train_losses, val_losses, val_accuracies, learning_rates = [], [], [], []

for epoch in range(EPOCHS):
    current_lr = scheduler.step(epoch)
    learning_rates.append(current_lr)
    
    # 训练
    running_loss = 0.0
    if USE_CACHED_FEATURES:
        data_iter = train_feat_loader
    else:
        data_iter = train_loader
    train_loop = tqdm(data_iter, desc="Epoch {}/{} [Training] LR: {:.2e}".format(epoch+1, EPOCHS, current_lr))

    if USE_CACHED_FEATURES:
        model.classifier.train()
    else:
        model.train()

    for x, labels in train_loop:
        if x is None: 
            continue
        
        x = x.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        
        # 应用Mixup和Label Smoothing
        if USE_MIXUP and USE_CACHED_FEATURES:
            # 对预计算特征应用mixup
            mixed_x, labels_a, labels_b, lam = mixup_data(x, labels, MIXUP_ALPHA, DEVICE)
            outputs = model.classifier(mixed_x)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        elif USE_MIXUP and not USE_CACHED_FEATURES:
            # 对原始图像特征应用mixup
            with torch.no_grad():
                image_features = model.vision_encoder.get_image_features(pixel_values=x)
            mixed_features, labels_a, labels_b, lam = mixup_data(image_features, labels, MIXUP_ALPHA, DEVICE)
            outputs = model.classifier(mixed_features)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            # 不使用mixup，但可能使用label smoothing
            if USE_CACHED_FEATURES:
                outputs = model.classifier(x)
            else:
                outputs = model(x)
            loss = criterion(outputs, labels)
        
        loss.backward() 
        
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
        
        optimizer.step()

        batch_size_now = x.size(0)
        running_loss += loss.item() * batch_size_now
        train_loop.set_postfix(loss=loss.item(), lr=current_lr)

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    print("\nEpoch {} Training Loss: {:.4f}, LR: {:.2e}".format(epoch+1, epoch_loss, current_lr))

    # --- 验证（验证时不使用mixup和label smoothing） ---
    if USE_CACHED_FEATURES:
        data_iter = val_feat_loader
        model.classifier.eval()
    else:
        data_iter = val_loader
        model.eval()

    val_loss = 0.0
    correct = 0
    total = 0
    label_0_true_positive = 0 
    label_0_false_negative = 0 
    label_0_false_positive = 0 
    
    # 验证时使用标准的CrossEntropy损失
    val_criterion = nn.CrossEntropyLoss()
    
    val_loop = tqdm(data_iter, desc="Epoch {}/{} [Validation]".format(epoch+1, EPOCHS))
    with torch.no_grad():
        for x, labels in val_loop:
            if x is None: 
                continue
            labels = labels.to(DEVICE)
            
            if USE_CACHED_FEATURES:
                outputs = model.classifier(x.to(DEVICE))
            else:
                outputs = model(x.to(DEVICE))
            loss = val_criterion(outputs, labels)  # 验证时使用标准损失
            val_loss += loss.item() * x.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                if true_label == 0 and pred_label == 0:
                    label_0_true_positive += 1
                elif true_label == 0 and pred_label != 0:
                    label_0_false_negative += 1
                elif true_label != 0 and pred_label == 0:
                    label_0_false_positive += 1

    epoch_val_loss = val_loss / len(val_dataset)
    accuracy = 100 * correct / total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(accuracy)
    
    if (label_0_true_positive + label_0_false_negative) > 0:
        label_0_recall = label_0_true_positive / (label_0_true_positive + label_0_false_negative)
    else:
        label_0_recall = 0.0
    
    total_non_label_0 = total - (label_0_true_positive + label_0_false_negative)
    if total_non_label_0 > 0:
        label_0_false_positive_rate = label_0_false_positive / total_non_label_0
    else:
        label_0_false_positive_rate = 0.0
    print("Epoch {} Validation Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch+1, epoch_val_loss, accuracy))
    print("Label 0 Recall: {:.4f} ({:.2f}%)".format(label_0_recall, label_0_recall*100))
    print("Label 0 False Positive Rate: {:.4f} ({:.2f}%)".format(label_0_false_positive_rate, label_0_false_positive_rate*100))
    print("-" * 50)

print("Training finished.")

# --- 保存模型和结果（保持不变：保存完整模型参数） ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Training finished. Model saved to {MODEL_SAVE_PATH}")

training_history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
    'learning_rates': learning_rates,
    'config': {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'base_lr': LEARNING_RATE,
        'min_lr': MIN_LR,
        'warmup_epochs': WARMUP_EPOCHS,
        'use_cached_features': USE_CACHED_FEATURES,
        'use_mixup': USE_MIXUP,
        'mixup_alpha': MIXUP_ALPHA,
        'use_label_smoothing': USE_LABEL_SMOOTHING,
        'label_smoothing': LABEL_SMOOTHING
    }
}
with open('label_map.json', 'w') as f:
    json.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f)
print("Label map saved to label_map.json")

with open('training_history.json', 'w') as f:
    json.dump(training_history, f, indent=2)
print("Training history saved to training_history.json")