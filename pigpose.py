import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import timm
import random
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. Configuration & Reproducibility
# ==========================================
class Config:
    SEED = 42
    IMG_SIZE = 384
    MODEL_NAME = 'tf_efficientnet_b3_ns'
    NUM_CLASSES = 5
    BATCH_SIZE = 16        # 根据显存调整，B3在16G显存通常可以开16-32
    EPOCHS = 8
    LEARNING_RATE = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(Config.SEED)

# ==========================================
# 2. Helper Functions (File Parsing)
# ==========================================
def get_data_root():
    """
    自动判断运行环境：
    - Kaggle Notebook: /kaggle/input/xxx/
    - 云 GPU / 本地: 手动指定路径
    """
    # Kaggle 环境
    if os.path.exists("/kaggle/input"):
        cand = glob.glob("/kaggle/input/*/train.csv")
        if len(cand) == 0:
            raise FileNotFoundError("train.csv not found in /kaggle/input")
        return os.path.dirname(cand[0])

    # ===== 非 Kaggle 环境（云 GPU / 本地）=====
    # ⚠️ 改成你服务器上的真实路径
    return "/root/pig_posture_recognition"


def resolve_path(img_dir, image_name):
    """
    Kaggle 官方数据中 image_id 就在 train_images / test_images 下
    不需要递归搜索（云服务器上会非常慢）
    """
    p = os.path.join(img_dir, str(image_name))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Image not found: {p}")
    return p


def parse_bbox(row_bbox_str):
    """
    解析 bbox 字符串 '[x, y, w, h]'
    """
    try:
        s = str(row_bbox_str).strip().strip("[]")
        parts = [float(p) for p in s.split(",") if p.strip() != ""]
        if len(parts) == 4:
            x, y, w, h = parts
            return x, y, x + w, y + h
    except Exception:
        pass
    return 0, 0, 0, 0


# ==========================================
# 3. Data Loading & Splitting
# ==========================================
DATA_ROOT = get_data_root()

TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV_PATH  = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMG_DIR  = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR   = os.path.join(DATA_ROOT, "test_images")

print("Using DATA_ROOT:", DATA_ROOT)
print("Train CSV:", TRAIN_CSV_PATH)
print("Train Images:", TRAIN_IMG_DIR)

# 读取数据
df_train = pd.read_csv(TRAIN_CSV_PATH)
df_test  = pd.read_csv(TEST_CSV_PATH)

# 为了防止同一张图里的猪泄露到验证集，我们需要按 Image ID 进行划分
# 既然要求“所有数据一起训练”，我们这里只划分极少部分(例如5%)做验证，或者你可以设置为0
splitter = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=Config.SEED)
train_idxs, val_idxs = next(splitter.split(df_train, groups=df_train['image_id']))

train_data = df_train.iloc[train_idxs].reset_index(drop=True)
valid_data = df_train.iloc[val_idxs].reset_index(drop=True)

print(f"Train Size: {len(train_data)}, Valid Size: {len(valid_data)}")

# ==========================================
# 4. Custom Dataset with "Letterbox" Preprocessing
# ==========================================
class PigDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. 读取图片
        img_path = resolve_path(self.img_dir, row['image_id'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape

        # 2. 解析 BBox
        x1, y1, x2, y2 = parse_bbox(row['bbox'])
        
        # 3. Context Padding (增加上下文，防止切太紧)
        # 向外扩展 15% 的边距，有助于模型看清猪的轮廓
        w_box = x2 - x1
        h_box = y2 - y1
        pad = 0.15 * max(w_box, h_box)
        
        x1_p = max(0, int(x1 - pad))
        y1_p = max(0, int(y1 - pad))
        x2_p = min(W, int(x2 + pad))
        y2_p = min(H, int(y2 + pad))
        
        # 4. Crop (裁剪)
        crop = image[y1_p:y2_p, x1_p:x2_p]
        
        # 5. Letterbox Resize (关键步骤：保持长宽比，不拉伸)
        h_c, w_c, _ = crop.shape
        max_dim = max(h_c, w_c)
        
        # 创建正方形黑色画布
        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        
        # 计算居中位置
        start_y = (max_dim - h_c) // 2
        start_x = (max_dim - w_c) // 2
        
        # 将裁剪图贴到画布中心
        square_img[start_y:start_y+h_c, start_x:start_x+w_c] = crop
        
        # 6. Resize 到模型输入大小 (如 320x320)
        final_img = cv2.resize(square_img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)

        # 7. Albumentations 增强
        if self.transforms:
            augmented = self.transforms(image=final_img)
            final_img = augmented['image']
        aspect_ratio = w_box / (h_box + 1e-6)
        area_ratio = (w_box * h_box) / (H * W)
        if self.is_test:
            return final_img, row['row_id'], torch.tensor([aspect_ratio, area_ratio], dtype=torch.float)
        else:
            return final_img, torch.tensor(row['class_id'], dtype=torch.long), torch.tensor([aspect_ratio, area_ratio])

# ==========================================
# 5. Augmentations
# ==========================================
# 训练集增强：翻转、旋转、亮度对比度
train_augs = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2), # 猪可能会平躺，垂直翻转有时候也合理，视情况而定
    A.Rotate(limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# 验证/测试集：仅标准化
val_augs = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==========================================
# 6. Model Definition
# ==========================================
class PigModel(nn.Module):
    def __init__(self, model_name, num_classes, weight_path=None):
        super().__init__()

        # 1. 不让 timm 去下载
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0
        )

        # 2. 手动加载本地权重
        if weight_path is not None:
            assert os.path.exists(weight_path), f"Weight not found: {weight_path}"
            state_dict = torch.load(weight_path, map_location="cpu")

            # timm 权重一般直接 load
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded pretrained weights from {weight_path}")

        in_features = self.backbone.num_features

        # 3. 分类头
        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_features + 2),  # +2 meta
            nn.Linear(in_features + 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, meta):
        feat = self.backbone(x)
        feat = torch.cat([feat, meta], dim=1)
        return self.fc(feat)
# ==========================================
# 7. Training Setup
# ==========================================
# DataLoader
train_dataset = PigDataset(train_data, TRAIN_IMG_DIR, transforms=train_augs, is_test=False)
valid_dataset = PigDataset(valid_data, TRAIN_IMG_DIR, transforms=val_augs, is_test=False)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                          num_workers=Config.num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                          num_workers=Config.num_workers, pin_memory=True)

# Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PigModel(
    model_name="efficientnet_b3",
    num_classes=5,
    weight_path="/root/efficientnet_b3.pth"
).to(device)
# class weight + focalloss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction='none', weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data['class_id']),
    y=train_data['class_id']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(Config.device)

criterion = FocalLoss(gamma=2.0, alpha=class_weights)

optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
scaler = GradScaler() # 混合精度训练

# ==========================================
# 8. Training Loop
# ==========================================
best_f1 = 0.0

for epoch in range(Config.EPOCHS):
    # --- Train ---
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
    for images, labels, meta in pbar:
        images = images.to(Config.device)
        labels = labels.to(Config.device)
        meta = meta.to(Config.device)
    
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images, meta)
            loss = criterion(outputs, labels)
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    train_f1 = f1_score(train_labels, train_preds, average='macro')
    
    # --- Validation ---
    model.eval()
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for images, labels, meta in valid_loader:
            images = images.to(Config.device)
            labels = labels.to(Config.device)
            meta = meta.to(Config.device)
        
            outputs = model(images, meta)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
    
    scheduler.step()
    
    # 保存最佳模型
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_pig_model.pth")
        print(f"Saved Best Model (F1: {best_f1:.4f})")

print("Training Complete!")

# ==========================================
# 9. Inference & Submission
# ==========================================
# 加载最佳模型权重
model.load_state_dict(torch.load("best_pig_model.pth"))
model.eval()

test_dataset = PigDataset(df_test, TEST_IMG_DIR, transforms=val_augs, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                         num_workers=Config.num_workers)

submission_rows = []

print("Starting Inference...")

def tta_predict(model, images, meta):
    preds = []

    # 原图
    preds.append(torch.softmax(model(images, meta), dim=1))

    # 水平翻转
    images_flipped = torch.flip(images, dims=[3])
    preds.append(torch.softmax(model(images_flipped, meta), dim=1))

    return torch.stack(preds).mean(0)

with torch.no_grad():
    for images, row_ids, meta in tqdm(test_loader, desc="Inference"):
        images = images.to(Config.device)
        meta = meta.to(Config.device)

        outputs = tta_predict(model, images, meta)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        for row_id, pred in zip(row_ids, preds):
            submission_rows.append({
                'row_id': row_id,
                'class_id': pred
            })

# 生成 CSV
sub_df = pd.DataFrame(submission_rows)
sub_df.to_csv("submission.csv", index=False)

print(f"Submission saved to submission.csv with {len(sub_df)} rows.")
print(sub_df.head())