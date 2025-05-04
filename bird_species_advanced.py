import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import warnings
import optuna
import json
import argparse
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import umap
from collections import Counter
import torch.nn.functional as F # 添加 F 的导入，FocalLoss 中需要

# Matplotlib 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

warnings.filterwarnings('ignore')

# 定义文件路径
training_path = r"C:\Users\19395\Downloads\Project_6_Bird_species_recognition\birds-species-recognition\birds-species-recognition\training.csv"
testing_path = r"C:\Users\19395\Downloads\Project_6_Bird_species_recognition\birds-species-recognition\birds-species-recognition\testing.csv"

# 解析CSV文件
def parse_csv(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',', 2)
            if len(parts) >= 3:
                image_path = parts[0]
                class_id = int(parts[1])

                # 提取类名
                if '\\' in image_path:
                    class_name = image_path.split('\\')[0]
                    if '.' in class_name:
                        class_name = class_name.split('.', 1)[1]
                else:
                    class_name = "Unknown"

                # 解析特征向量
                features = np.array([float(x) for x in parts[2].split(',')])

                data.append((image_path, class_name, class_id, features))
    return data

# 特征工程函数
def apply_feature_engineering(X_train, X_test, y_train, method='none', n_components=100):
    """应用特征工程技术"""
    print(f"应用特征工程: {method}, 组件数/特征数: {n_components}")

    if method == 'pca':
        # 应用PCA降维
        pca = PCA(n_components=n_components, random_state=42)
        X_train_transformed = pca.fit_transform(X_train)
        X_test_transformed = pca.transform(X_test)

        # 计算解释方差比
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"PCA解释方差比: {explained_variance:.4f}")

        # 绘制解释方差比
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('组件数')
        plt.ylabel('累计解释方差比')
        plt.title('PCA解释方差比')
        plt.grid(True)
        plt.savefig('pca_explained_variance_advanced.png')

        return X_train_transformed, X_test_transformed, pca

    elif method == 'select_k_best':
        # 应用特征选择
        selector = SelectKBest(f_classif, k=n_components)
        X_train_transformed = selector.fit_transform(X_train, y_train)
        X_test_transformed = selector.transform(X_test)

        # 获取选择的特征索引
        selected_indices = selector.get_support(indices=True)
        print(f"选择的特征索引 (前10): {selected_indices[:10]}...")

        # 绘制特征重要性
        plt.figure(figsize=(10, 6))
        # 限制显示的特征数量，避免过多
        num_features_to_plot = min(len(selector.scores_), 500)
        plt.bar(range(num_features_to_plot), selector.scores_[:num_features_to_plot])
        plt.xlabel('特征索引')
        plt.ylabel('F值')
        plt.title('特征重要性 (前 {} 个)'.format(num_features_to_plot))
        plt.savefig('feature_importance_advanced.png')

        return X_train_transformed, X_test_transformed, selector

    elif method == 'umap': # <-- 添加 UMAP 选项
        print(f"应用 UMAP 降维，目标维度: {n_components}")
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        X_train_transformed = reducer.fit_transform(X_train)
        X_test_transformed = reducer.transform(X_test)
        print("UMAP 降维完成")
        # 可以选择性地绘制 UMAP 结果
        # plt.figure(figsize=(10, 8))
        # plt.scatter(X_train_transformed[:, 0], X_train_transformed[:, 1], c=y_train, cmap='Spectral', s=5)
        # plt.title('UMAP projection of the training dataset')
        # plt.xlabel('UMAP Component 1')
        # plt.ylabel('UMAP Component 2')
        # plt.colorbar()
        # plt.savefig('umap_projection_advanced.png')
        return X_train_transformed, X_test_transformed, reducer

    elif method == 'none':
        print("未应用特征工程")
        return X_train, X_test, None

    else:
        print(f"未知的特征工程方法: {method}")
        return X_train, X_test, None

# 特征归一化
def l2_normalize_features(features):
    """对特征进行L2归一化"""
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / (norm + 1e-10)  # 添加小值避免除零

# 自定义数据集类
class BirdDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 增强的DNN模型 - 优化结构
class EnhancedBirdDNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers=[1024, 512, 256], dropout_rate=0.5, activation_fn=nn.ReLU):
        super(EnhancedBirdDNN, self).__init__()

        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.BatchNorm1d(hidden_layers[0])) # BatchNorm通常放在激活函数之前
        layers.append(activation_fn())
        layers.append(nn.Dropout(dropout_rate))

        # 隐藏层
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))

        # 输出层
        layers.append(nn.Linear(hidden_layers[-1], num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 定义FT-Transformer模型 - 优化结构
class FTTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=0, dim_feedforward_factor=4, dropout=0.1, activation_fn=nn.ReLU):
        super(FTTransformer, self).__init__()

        # 特征嵌入层 (保持不变，或根据需要调整)
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            activation_fn(),
            nn.Dropout(dropout)
        )

        # 类别嵌入 (CLS token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 位置编码 (保持不变，因为我们只有一个特征向量+CLS)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 2, d_model))

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * dim_feedforward_factor,
            dropout=dropout,
            activation=activation_fn(), # 使用可配置的激活函数
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 分类头 (保持不变，或根据需要调整)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), # 添加LayerNorm在分类头之前可能有助于稳定训练
            nn.Linear(d_model, d_model // 2),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 特征嵌入 [batch_size, d_model]
        features_embedded = self.feature_embedding(x)

        # 重塑为序列 [batch_size, 1, d_model]
        features_embedded = features_embedded.unsqueeze(1)

        # 添加CLS token [batch_size, 2, d_model]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat([cls_tokens, features_embedded], dim=1)

        # 添加位置编码
        transformer_input = transformer_input + self.pos_encoder

        # 通过Transformer编码器
        memory = self.transformer_encoder(transformer_input)

        # 使用CLS token进行分类 [batch_size, d_model]
        cls_output = memory[:, 0]

        # 分类
        output = self.classifier(cls_output)

        return output

# Focal Loss 实现
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none') # 计算原始交叉熵损失，不进行reduction
        pt = torch.exp(-ce_loss) # 计算预测正确类别的概率
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss # 计算Focal Loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

# 训练模型函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=15):
    model.to(device)

    # 早停机制
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | '
              f'Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}')

        # 早停
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# 评估模型函数
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    return test_loss, accuracy, all_preds, all_labels

# Optuna 目标函数
def objective(trial, model_type, input_dim, num_classes, train_loader, val_loader, device, num_epochs, patience):
    if model_type == 'dnn':
        # DNN 超参数建议
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_layers = []
        last_dim = input_dim
        for i in range(n_layers):
            # 建议每层的单元数，逐渐减小
            out_features = trial.suggest_int(f'n_units_l{i}', 64, 1024, log=True)
            hidden_layers.append(out_features)
            last_dim = out_features
        # 将列表转换为字符串以存储在 Optuna 中
        trial.set_user_attr('hidden_layers_list', hidden_layers)
        # 存储字符串表示形式以用于 JSON 保存
        trial.suggest_categorical('hidden_layers', [str(hidden_layers)])

        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop', 'SGD'])
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)

        model = EnhancedBirdDNN(input_dim, num_classes, hidden_layers=hidden_layers, dropout_rate=dropout_rate).to(device)

    elif model_type == 'ft_transformer': # <-- FT-Transformer 的超参数建议
        d_model = trial.suggest_categorical('d_model', [128, 256, 512])
        nhead = trial.suggest_categorical('nhead', [4, 8])
        if d_model % nhead != 0:
            raise optuna.exceptions.TrialPruned()
        num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 4) # Renamed from num_layers for clarity
        # num_decoder_layers = trial.suggest_int('num_decoder_layers', 0, 2) # Example if decoder is needed
        dim_feedforward_factor = trial.suggest_int('dim_feedforward_factor', 2, 8)
        activation_name = trial.suggest_categorical('activation_fn', ['ReLU', 'GELU'])
        activation_fn = getattr(nn, activation_name)

        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.4)
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

        model = FTTransformer(
            input_dim, num_classes, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, # num_decoder_layers=num_decoder_layers,
            dim_feedforward_factor=dim_feedforward_factor,
            dropout=dropout_rate,
            activation_fn=activation_fn
        ).to(device)

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    # 损失函数选择和超参数建议
    criterion_name = trial.suggest_categorical('criterion', ['CrossEntropyLoss', 'FocalLoss'])
    if criterion_name == 'FocalLoss':
        focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9)
        focal_gamma = trial.suggest_float('focal_gamma', 0.5, 5.0)
        # 注意：如果类别不平衡显著，alpha可能需要根据类别频率设置，而不是直接搜索
        # 简单的搜索可能对于alpha不是最优的，但可以作为起点
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean').to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # 优化器选择
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop', 'SGD'])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # 学习率调度器（可选，但通常有益）
    # scheduler_name = trial.suggest_categorical('scheduler', ['ReduceLROnPlateau', 'CosineAnnealingLR', 'None'])
    scheduler = None
    # if scheduler_name == 'ReduceLROnPlateau':
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    # elif scheduler_name == 'CosineAnnealingLR':
    #      scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练和验证循环
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 可选：梯度裁剪
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_train_acc = correct / total

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels) # Also calculate val loss if needed
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_acc = val_correct / val_total
        val_epoch_loss = val_loss / val_total

        # 更新学习率调度器 (如果使用)
        # if scheduler:
        #     if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        #          scheduler.step(val_epoch_acc)
        #     elif not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        #          scheduler.step()

        # print(f'Trial {trial.number} Epoch {epoch+1}/{num_epochs} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {val_epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}')

        # 检查是否是最佳准确率
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
            # 保存分数最高的模型的检查点 (可选)
            # torch.save(model.state_dict(), f'best_model_trial_{trial.number}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Trial {trial.number} early stopping at epoch {epoch+1}")
                break # 早停

        # Optuna 剪枝 - 基于验证准确率
        trial.report(val_epoch_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    # Placeholder comment indicating where the second block was removed

# ... existing code ...

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # 训练和验证循环 (与 train_model 类似，但只返回最终验证准确率)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_acc = val_correct / val_total

        # 检查是否是最佳准确率
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Trial {trial.number} early stopping at epoch {epoch+1}")
                break # 早停

        # Optuna 剪枝
        trial.report(val_epoch_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    # Placeholder comment indicating where the second block was removed

# ... existing code ...

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # 训练和验证循环 (与 train_model 类似，但只返回最终验证准确率)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_acc = val_correct / val_total

        # 检查是否是最佳准确率
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Trial {trial.number} early stopping at epoch {epoch+1}")
                break # 早停

        # Optuna 剪枝
        trial.report(val_epoch_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    # Placeholder comment indicating where the second block was removed

# ... existing code ...

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # 训练和验证循环 (与 train_model 类似，但只返回最终验证准确率)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_acc = val_correct / val_total

        # 检查是否是最佳准确率
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Trial {trial.number} early stopping at epoch {epoch+1}")
                break # 早停

        # Optuna 剪枝
        trial.report(val_epoch_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    # Placeholder comment indicating where the second block was removed

# ... existing code ...

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # 训练和验证循环 (与 train_model 类似，但只返回最终验证准确率)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_acc = val_correct / val_total

        # 检查是否是最佳准确率
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Trial {trial.number} early stopping at epoch {epoch+1}")
                break # 早停

        # Optuna 剪枝
        trial.report(val_epoch_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    # Placeholder comment indicating where the second block was removed

# ... existing code ...

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # 训练和验证循环 (与 train_model 类似，但只返回最终验证准确率)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_acc = val_correct / val_total

        # 检查是否是最佳准确率
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Trial {trial.number} early stopping at epoch {epoch+1}")
                break # 早停

        # Optuna 剪枝
        trial.report(val_epoch_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    # Placeholder comment indicating where the second block was removed

# ... existing code ...

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # 训练和验证循环 (与 train_model 类似，但只返回最终验证准确率)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_acc = val_correct / val_total

        # 检查是否是最佳准确率
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Trial {trial.number} early stopping at epoch {epoch+1}")
                break # 早停

        # Optuna 剪枝
        trial.report(val_epoch_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    # Placeholder comment indicating where the second block was removed

# ... existing code ...

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # 训练和验证循环 (与 train_model 类似，但只返回最终验证准确率)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_acc = val_correct / val_total

        # 检查是否是最佳准确率
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Trial {trial.number} early stopping at epoch {epoch+1}")
                break # 早停

        # Optuna 剪枝
        trial.report(val_epoch_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    # Placeholder comment indicating where the second block was removed

# ... existing code ...

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # 训练和验证循环 (与 train_model 类似，但只返回最终验证准确率)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_acc = val_correct / val_total

        # 检查是否是最佳准确率
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Trial {trial.number} early stopping at epoch {epoch+1}")
                break # 早停

        # Optuna 剪枝
        trial.report(val_epoch_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    # Placeholder comment indicating where the second block was removed

# ... existing code ...

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # 训练和验证循环 (与 train_model 类似，但只返回最终验证准确率)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
