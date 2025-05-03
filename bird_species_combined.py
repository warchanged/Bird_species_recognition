import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import json
import warnings
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

# 定义DNN模型
class BirdDNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers=[512, 256], dropout_rate=0.5):
        super(BirdDNN, self).__init__()
        
        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# 定义FT-Transformer模型
class FTTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super(FTTransformer, self).__init__()
        
        # 特征嵌入层
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 类别嵌入 (CLS token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, 2, d_model))  # 只有2个位置：CLS和特征
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
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
        x = torch.cat([cls_tokens, features_embedded], dim=1)
        
        # 添加位置编码
        x = x + self.pos_encoder
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 使用CLS token进行分类 [batch_size, d_model]
        x = x[:, 0]
        
        # 分类
        x = self.classifier(x)
        
        return x

# 训练模型函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=15):
    model.to(device)
    
    # 早停机制
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
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
        
        # 更新学习率
        scheduler.step(val_epoch_loss)
        
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

# 特征工程函数
def apply_feature_engineering(X_train, X_test, y_train, method='pca', n_components=100):
    """应用特征工程技术"""
    print(f"应用特征工程: {method}, 组件数: {n_components}")
    
    if method == 'pca':
        # 应用PCA降维
        pca = PCA(n_components=n_components)
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
        plt.savefig('pca_explained_variance.png')
        
        # 保存PCA模型
        import pickle
        with open('pca_model.pkl', 'wb') as f:
            pickle.dump(pca, f)
        
        return X_train_transformed, X_test_transformed
    
    elif method == 'select_k_best':
        # 应用特征选择
        selector = SelectKBest(f_classif, k=n_components)
        X_train_transformed = selector.fit_transform(X_train, y_train)
        X_test_transformed = selector.transform(X_test)
        
        # 获取选择的特征索引
        selected_indices = selector.get_support(indices=True)
        print(f"选择的特征索引: {selected_indices[:10]}...")
        
        # 绘制特征重要性
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(selector.scores_)), selector.scores_)
        plt.xlabel('特征索引')
        plt.ylabel('F值')
        plt.title('特征重要性')
        plt.savefig('feature_importance.png')
        
        # 保存特征选择模型
        import pickle
        with open('selector_model.pkl', 'wb') as f:
            pickle.dump(selector, f)
        
        return X_train_transformed, X_test_transformed
    
    else:
        print(f"未知的特征工程方法: {method}")
        return X_train, X_test

# 集成预测函数
def ensemble_predict(models, X_test, device):
    """使用多个模型进行集成预测"""
    all_probs = []
    
    for model in models:
        model.eval()
        X_tensor = torch.FloatTensor(X_test).to(device)
        
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    # 平均概率
    avg_probs = np.mean(all_probs, axis=0)
    predictions = np.argmax(avg_probs, axis=1)
    
    return predictions

# 主函数
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Bird Species Recognition with Feature Engineering and FT-Transformer')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'select_k_best', 'none'],
                        help='Feature engineering method: pca, select_k_best, or none')
    parser.add_argument('--n_components', type=int, default=100,
                        help='Number of components/features to keep')
    parser.add_argument('--model_type', type=str, default='ft_transformer', 
                        choices=['dnn', 'ft_transformer', 'ensemble'],
                        help='Model type: dnn, ft_transformer, or ensemble')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs for model training')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience for model training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 解析训练和测试数据
    print("解析训练数据...")
    training_data = parse_csv(training_path)
    print("解析测试数据...")
    testing_data = parse_csv(testing_path)
    
    # 提取特征和标签
    X_train = np.vstack([item[3] for item in training_data])
    y_train = np.array([item[2] - 1 for item in training_data])  # 减1使类别从0开始
    X_test = np.vstack([item[3] for item in testing_data])
    y_test = np.array([item[2] - 1 for item in testing_data])
    
    # 数据预处理
    print("数据预处理...")
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 应用特征工程
    if args.method != 'none':
        X_train_transformed, X_test_transformed = apply_feature_engineering(
            X_train_scaled, X_test_scaled, y_train, 
            method=args.method, n_components=args.n_components
        )
    else:
        X_train_transformed, X_test_transformed = X_train_scaled, X_test_scaled
    
    # 应用L2归一化
    X_train_normalized = l2_normalize_features(X_train_transformed)
    X_test_normalized = l2_normalize_features(X_test_transformed)
    
    # 划分训练集和验证集
    val_size = 0.1
    val_indices = np.random.choice(len(X_train_normalized), int(val_size * len(X_train_normalized)), replace=False)
    train_indices = np.array([i for i in range(len(X_train_normalized)) if i not in val_indices])
    
    X_val = X_train_normalized[val_indices]
    y_val = y_train[val_indices]
    X_train_final = X_train_normalized[train_indices]
    y_train_final = y_train[train_indices]
    
    print(f"原始特征维度: {X_train.shape[1]}")
    print(f"处理后特征维度: {X_train_transformed.shape[1]}")
    print(f"训练数据形状: {X_train_final.shape}")
    print(f"验证数据形状: {X_val.shape}")
    print(f"测试数据形状: {X_test_normalized.shape}")
    
    # 创建数据加载器
    train_dataset = BirdDataset(X_train_final, y_train_final)
    val_dataset = BirdDataset(X_val, y_val)
    test_dataset = BirdDataset(X_test_normalized, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 获取输入维度和类别数
    input_dim = X_train_transformed.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"特征维度: {input_dim}")
    print(f"类别数量: {num_classes}")
    
    # 创建模型
    models = []
    
    # 训练DNN模型
    if args.model_type in ['dnn', 'ensemble']:
        print("\n训练DNN模型...")
        dnn_model = BirdDNN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=[512, 256],
            dropout_rate=0.5
        )
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(dnn_model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 训练DNN模型
        dnn_model, dnn_train_losses, dnn_val_losses, dnn_train_accs, dnn_val_accs = train_model(
            dnn_model, train_loader, val_loader, criterion, optimizer, device, 
            num_epochs=args.epochs, patience=args.patience
        )
        
        # 评估DNN模型
        dnn_test_loss, dnn_test_acc, dnn_preds, dnn_labels = evaluate_model(
            dnn_model, test_loader, criterion, device
        )
        print(f"DNN模型测试准确率: {dnn_test_acc:.4f}")
        
        # 保存DNN模型
        torch.save(dnn_model.state_dict(), f'{args.method}_dnn_model.pth')
        
        # 添加到模型列表
        models.append(dnn_model)
        
        # 绘制DNN训练历史
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(dnn_train_accs)
        plt.plot(dnn_val_accs)
        plt.title('DNN - 准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend(['训练', '验证'])
        
        plt.subplot(1, 2, 2)
        plt.plot(dnn_train_losses)
        plt.plot(dnn_val_losses)
        plt.title('DNN - 损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend(['训练', '验证'])
        
        plt.tight_layout()
        plt.savefig(f'{args.method}_dnn_training_history.png')
    
    # 训练FT-Transformer模型
    if args.model_type in ['ft_transformer', 'ensemble']:
        print("\n训练FT-Transformer模型...")
        ft_transformer_model = FTTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=512,
            nhead=8,
            num_layers=3,
            dropout=0.1
        )
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(ft_transformer_model.parameters(), lr=0.0001, weight_decay=1e-4)
        
        # 训练FT-Transformer模型
        ft_transformer_model, ft_transformer_train_losses, ft_transformer_val_losses, ft_transformer_train_accs, ft_transformer_val_accs = train_model(
            ft_transformer_model, train_loader, val_loader, criterion, optimizer, device, 
            num_epochs=args.epochs, patience=args.patience
        )
        
        # 评估FT-Transformer模型
        ft_transformer_test_loss, ft_transformer_test_acc, ft_transformer_preds, ft_transformer_labels = evaluate_model(
            ft_transformer_model, test_loader, criterion, device
        )
        print(f"FT-Transformer模型测试准确率: {ft_transformer_test_acc:.4f}")
        
        # 保存FT-Transformer模型
        torch.save(ft_transformer_model.state_dict(), f'{args.method}_ft_transformer_model.pth')
        
        # 添加到模型列表
        models.append(ft_transformer_model)
        
        # 绘制FT-Transformer训练历史
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(ft_transformer_train_accs)
        plt.plot(ft_transformer_val_accs)
        plt.title('FT-Transformer - 准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend(['训练', '验证'])
        
        plt.subplot(1, 2, 2)
        plt.plot(ft_transformer_train_losses)
        plt.plot(ft_transformer_val_losses)
        plt.title('FT-Transformer - 损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend(['训练', '验证'])
        
        plt.tight_layout()
        plt.savefig(f'{args.method}_ft_transformer_training_history.png')
    
    # 训练SVM模型（如果使用集成）
    if args.model_type == 'ensemble':
        print("\n训练SVM模型...")
        svm_model = SVC(probability=True, C=10, gamma='scale', kernel='rbf')
        svm_model.fit(X_train_final, y_train_final)
        
        # 评估SVM模型
        svm_preds = svm_model.predict(X_test_normalized)
        svm_test_acc = accuracy_score(y_test, svm_preds)
        print(f"SVM模型测试准确率: {svm_test_acc:.4f}")
        
        # 保存SVM模型
        import pickle
        with open(f'{args.method}_svm_model.pkl', 'wb') as f:
            pickle.dump(svm_model, f)
    
    # 创建集成模型（如果使用集成）
    if args.model_type == 'ensemble' and len(models) > 0:
        print("\n创建集成模型...")
        
        # 使用集成预测
        ensemble_preds = ensemble_predict(models, X_test_normalized, device)
        
        # 评估集成模型
        ensemble_acc = accuracy_score(y_test, ensemble_preds)
        print(f"集成模型测试准确率: {ensemble_acc:.4f}")
        
        # 生成集成模型分类报告
        ensemble_report = classification_report(y_test, ensemble_preds, output_dict=True)
        print(f"集成模型精确率 (avg): {ensemble_report['macro avg']['precision']:.4f}")
        print(f"集成模型召回率 (avg): {ensemble_report['macro avg']['recall']:.4f}")
        print(f"集成模型F1分数 (avg): {ensemble_report['macro avg']['f1-score']:.4f}")
        
        # 比较各模型性能
        plt.figure(figsize=(10, 6))
        
        model_names = []
        accuracies = []
        
        if 'dnn_test_acc' in locals():
            model_names.append("DNN")
            accuracies.append(dnn_test_acc)
        
        if 'ft_transformer_test_acc' in locals():
            model_names.append("FT-Transformer")
            accuracies.append(ft_transformer_test_acc)
        
        if 'svm_test_acc' in locals():
            model_names.append("SVM")
            accuracies.append(svm_test_acc)
        
        model_names.append("集成")
        accuracies.append(ensemble_acc)
        
        plt.bar(model_names, accuracies)
        plt.title('模型准确率比较')
        plt.xlabel('模型')
        plt.ylabel('准确率')
        plt.ylim(0.5, 1.0)
        
        plt.tight_layout()
        plt.savefig(f'{args.method}_model_comparison.png')
    
    # 如果只使用单个模型，生成详细的分类报告
    if args.model_type == 'dnn':
        report = classification_report(dnn_labels, dnn_preds, output_dict=True)
        print(f"DNN模型精确率 (avg): {report['macro avg']['precision']:.4f}")
        print(f"DNN模型召回率 (avg): {report['macro avg']['recall']:.4f}")
        print(f"DNN模型F1分数 (avg): {report['macro avg']['f1-score']:.4f}")
    
    elif args.model_type == 'ft_transformer':
        report = classification_report(ft_transformer_labels, ft_transformer_preds, output_dict=True)
        print(f"FT-Transformer模型精确率 (avg): {report['macro avg']['precision']:.4f}")
        print(f"FT-Transformer模型召回率 (avg): {report['macro avg']['recall']:.4f}")
        print(f"FT-Transformer模型F1分数 (avg): {report['macro avg']['f1-score']:.4f}")
    
    print(f"\n分析完成。结果已保存为PNG文件。")
