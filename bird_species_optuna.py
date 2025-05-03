import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import optuna
import json
import pickle
import argparse
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
    def __init__(self, input_dim, num_classes, hidden_layers, dropout_rate=0.5, use_batch_norm=True):
        super(BirdDNN, self).__init__()

        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # 隐藏层
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # 输出层
        layers.append(nn.Linear(hidden_layers[-1], num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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

# 定义Transformer模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        # 特征映射到Transformer维度
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model*4,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # 将特征转换为序列形式 [batch_size, 1, input_dim]
        x = x.unsqueeze(1)

        # 投影到Transformer维度
        x = self.input_proj(x)

        # 添加位置编码
        x = x + self.pos_encoder

        # 通过Transformer编码器
        x = self.transformer_encoder(x)

        # 取序列的平均值作为特征表示
        x = x.mean(dim=1)

        # 分类
        x = self.classifier(x)

        return x

# 定义简化版FT-Transformer模型 (Feature Tokenizer Transformer)
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

# Optuna目标函数 - DNN模型
def objective_dnn(trial, X_train, y_train, X_val, y_val, input_dim, num_classes, device):
    # 定义超参数搜索空间
    n_layers = trial.suggest_int('n_layers', 2, 4)

    hidden_layers = []
    for i in range(n_layers):
        hidden_layers.append(trial.suggest_int(f'hidden_layer_{i}', 128, 2048, log=True))

    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])

    # 创建模型
    model = BirdDNN(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 创建数据加载器
    train_dataset = BirdDataset(X_train, y_train)
    val_dataset = BirdDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 训练模型
    model, _, _, _, _ = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=20, patience=5  # 减少训练时间
    )

    # 评估模型
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

    val_acc = val_correct / val_total

    return val_acc

# Optuna目标函数 - Transformer模型
def objective_transformer(trial, X_train, y_train, X_val, y_val, input_dim, num_classes, device):
    # 定义超参数搜索空间
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    nhead = trial.suggest_categorical('nhead', [4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # 创建模型
    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 创建数据加载器
    train_dataset = BirdDataset(X_train, y_train)
    val_dataset = BirdDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 训练模型
    model, _, _, _, _ = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=20, patience=5  # 减少训练时间
    )

    # 评估模型
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

    val_acc = val_correct / val_total

    return val_acc

# Optuna目标函数 - FT-Transformer模型
def objective_ft_transformer(trial, X_train, y_train, X_val, y_val, input_dim, num_classes, device):
    # 定义超参数搜索空间
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    nhead = trial.suggest_categorical('nhead', [4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # 创建模型
    model = FTTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 创建数据加载器
    train_dataset = BirdDataset(X_train, y_train)
    val_dataset = BirdDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 训练模型
    model, _, _, _, _ = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=20, patience=5  # 减少训练时间
    )

    # 评估模型
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

    val_acc = val_correct / val_total

    return val_acc

# 主函数
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Bird Species Recognition with Optuna Hyperparameter Optimization')
    parser.add_argument('--model_type', type=str, default='all',
                        choices=['dnn', 'transformer', 'ft_transformer', 'all'],
                        help='Model type to optimize: dnn, transformer, ft_transformer, or all')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of Optuna trials for each model type')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs for final model training')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience for final model training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Default batch size (will be overridden by Optuna if optimized)')
    parser.add_argument('--no_svm', action='store_true',
                        help='Skip SVM training and optimization')
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

    # 应用L2归一化
    X_train_normalized = l2_normalize_features(X_train_scaled)
    X_test_normalized = l2_normalize_features(X_test_scaled)

    # 划分训练集和验证集
    val_size = 0.1
    val_indices = np.random.choice(len(X_train_normalized), int(val_size * len(X_train_normalized)), replace=False)
    train_indices = np.array([i for i in range(len(X_train_normalized)) if i not in val_indices])

    X_val = X_train_normalized[val_indices]
    y_val = y_train[val_indices]
    X_train_final = X_train_normalized[train_indices]
    y_train_final = y_train[train_indices]

    print(f"训练数据形状: {X_train_final.shape}")
    print(f"验证数据形状: {X_val.shape}")
    print(f"测试数据形状: {X_test_normalized.shape}")

    # 获取输入维度和类别数
    input_dim = X_train_normalized.shape[1]
    num_classes = len(np.unique(y_train))

    print(f"特征维度: {input_dim}")
    print(f"类别数量: {num_classes}")

    # 创建集成预测函数
    def ensemble_predict(models, X, device):
        """使用多个模型进行集成预测"""
        all_probs = []

        for model in models:
            model.eval()
            X_tensor = torch.FloatTensor(X).to(device)

            with torch.no_grad():
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)

        # 平均概率
        avg_probs = np.mean(all_probs, axis=0)
        predictions = np.argmax(avg_probs, axis=1)

        return predictions

    # 使用命令行参数设置模型类型
    model_type = args.model_type

    # 使用Optuna进行超参数优化
    print(f"\n开始Optuna超参数优化 (模型类型: {model_type})...")

    # DNN模型优化
    if model_type in ["dnn", "all"]:
        print("\n优化DNN模型...")

        # 创建一个函数来包装DNN目标函数
        def objective_dnn_wrapper(trial):
            return objective_dnn(
                trial, X_train_final, y_train_final, X_val, y_val,
                input_dim, num_classes, device
            )

        # 创建Optuna研究
        dnn_study = optuna.create_study(direction='maximize', study_name="DNN优化")
        dnn_study.optimize(objective_dnn_wrapper, n_trials=args.n_trials)

        # 获取最佳超参数
        dnn_best_params = dnn_study.best_params
        dnn_best_value = dnn_study.best_value

        print(f"\nDNN最佳验证准确率: {dnn_best_value:.4f}")
        print("DNN最佳超参数:")
        for param, value in dnn_best_params.items():
            print(f"  {param}: {value}")

        # 保存DNN超参数
        with open('optuna_dnn_best_params.json', 'w') as f:
            json.dump(dnn_best_params, f, indent=4)

    # Transformer模型优化
    if model_type in ["transformer", "all"]:
        print("\n优化Transformer模型...")

        # 创建一个函数来包装Transformer目标函数
        def objective_transformer_wrapper(trial):
            return objective_transformer(
                trial, X_train_final, y_train_final, X_val, y_val,
                input_dim, num_classes, device
            )

        # 创建Optuna研究
        transformer_study = optuna.create_study(direction='maximize', study_name="Transformer优化")
        transformer_study.optimize(objective_transformer_wrapper, n_trials=args.n_trials)

        # 获取最佳超参数
        transformer_best_params = transformer_study.best_params
        transformer_best_value = transformer_study.best_value

        print(f"\nTransformer最佳验证准确率: {transformer_best_value:.4f}")
        print("Transformer最佳超参数:")
        for param, value in transformer_best_params.items():
            print(f"  {param}: {value}")

        # 保存Transformer超参数
        with open('optuna_transformer_best_params.json', 'w') as f:
            json.dump(transformer_best_params, f, indent=4)

    # FT-Transformer模型优化
    if model_type in ["ft_transformer", "all"]:
        print("\n优化FT-Transformer模型...")

        # 创建一个函数来包装FT-Transformer目标函数
        def objective_ft_transformer_wrapper(trial):
            return objective_ft_transformer(
                trial, X_train_final, y_train_final, X_val, y_val,
                input_dim, num_classes, device
            )

        # 创建Optuna研究
        ft_transformer_study = optuna.create_study(direction='maximize', study_name="FT-Transformer优化")
        ft_transformer_study.optimize(objective_ft_transformer_wrapper, n_trials=args.n_trials)

        # 获取最佳超参数
        ft_transformer_best_params = ft_transformer_study.best_params
        ft_transformer_best_value = ft_transformer_study.best_value

        print(f"\nFT-Transformer最佳验证准确率: {ft_transformer_best_value:.4f}")
        print("FT-Transformer最佳超参数:")
        for param, value in ft_transformer_best_params.items():
            print(f"  {param}: {value}")

        # 保存FT-Transformer超参数
        with open('optuna_ft_transformer_best_params.json', 'w') as f:
            json.dump(ft_transformer_best_params, f, indent=4)

    # 创建数据加载器
    train_dataset = BirdDataset(X_train_final, y_train_final)
    val_dataset = BirdDataset(X_val, y_val)
    test_dataset = BirdDataset(X_test_normalized, y_test)

    # 训练最终模型
    final_models = []

    # 训练最终DNN模型
    if model_type in ["dnn", "all"]:
        print("\n使用最佳超参数训练最终DNN模型...")

        # 构建隐藏层配置
        n_layers = dnn_best_params['n_layers']
        hidden_layers = [dnn_best_params[f'hidden_layer_{i}'] for i in range(n_layers)]

        # 创建最终DNN模型
        final_dnn_model = BirdDNN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dnn_best_params['dropout_rate'],
            use_batch_norm=dnn_best_params['use_batch_norm']
        )

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            final_dnn_model.parameters(),
            lr=dnn_best_params['learning_rate'],
            weight_decay=dnn_best_params['weight_decay']
        )

        # 创建数据加载器
        batch_size = dnn_best_params['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # 训练最终DNN模型
        final_dnn_model, dnn_train_losses, dnn_val_losses, dnn_train_accs, dnn_val_accs = train_model(
            final_dnn_model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=args.epochs, patience=args.patience
        )

        # 评估最终DNN模型
        final_dnn_model.eval()
        test_correct = 0
        test_total = 0
        dnn_preds = []
        dnn_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = final_dnn_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                dnn_preds.extend(predicted.cpu().numpy())
                dnn_labels.extend(labels.cpu().numpy())

        dnn_test_acc = test_correct / test_total
        print(f"\n最终DNN模型测试准确率: {dnn_test_acc:.4f}")

        # 保存最终DNN模型
        torch.save(final_dnn_model.state_dict(), 'optuna_dnn_best_model.pth')

        # 添加到最终模型列表
        final_models.append(final_dnn_model)

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
        plt.savefig('optuna_dnn_training_history.png')

        # 绘制DNN优化历史
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(dnn_study)
        plt.title('DNN优化历史')
        plt.tight_layout()
        plt.savefig('optuna_dnn_optimization_history.png')

        # 绘制DNN超参数重要性
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(dnn_study)
        plt.title('DNN超参数重要性')
        plt.tight_layout()
        plt.savefig('optuna_dnn_param_importances.png')

    # 训练最终Transformer模型
    if model_type in ["transformer", "all"]:
        print("\n使用最佳超参数训练最终Transformer模型...")

        # 创建最终Transformer模型
        final_transformer_model = TransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=transformer_best_params['d_model'],
            nhead=transformer_best_params['nhead'],
            num_layers=transformer_best_params['num_layers'],
            dropout=transformer_best_params['dropout']
        )

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            final_transformer_model.parameters(),
            lr=transformer_best_params['learning_rate'],
            weight_decay=transformer_best_params['weight_decay']
        )

        # 创建数据加载器
        batch_size = transformer_best_params['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # 训练最终Transformer模型
        final_transformer_model, transformer_train_losses, transformer_val_losses, transformer_train_accs, transformer_val_accs = train_model(
            final_transformer_model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=args.epochs, patience=args.patience
        )

        # 评估最终Transformer模型
        final_transformer_model.eval()
        test_correct = 0
        test_total = 0
        transformer_preds = []
        transformer_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = final_transformer_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                transformer_preds.extend(predicted.cpu().numpy())
                transformer_labels.extend(labels.cpu().numpy())

        transformer_test_acc = test_correct / test_total
        print(f"\n最终Transformer模型测试准确率: {transformer_test_acc:.4f}")

        # 保存最终Transformer模型
        torch.save(final_transformer_model.state_dict(), 'optuna_transformer_best_model.pth')

        # 添加到最终模型列表
        final_models.append(final_transformer_model)

        # 绘制Transformer训练历史
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(transformer_train_accs)
        plt.plot(transformer_val_accs)
        plt.title('Transformer - 准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend(['训练', '验证'])

        plt.subplot(1, 2, 2)
        plt.plot(transformer_train_losses)
        plt.plot(transformer_val_losses)
        plt.title('Transformer - 损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend(['训练', '验证'])

        plt.tight_layout()
        plt.savefig('optuna_transformer_training_history.png')

        # 绘制Transformer优化历史
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(transformer_study)
        plt.title('Transformer优化历史')
        plt.tight_layout()
        plt.savefig('optuna_transformer_optimization_history.png')

        # 绘制Transformer超参数重要性
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(transformer_study)
        plt.title('Transformer超参数重要性')
        plt.tight_layout()
        plt.savefig('optuna_transformer_param_importances.png')

    # 训练最终FT-Transformer模型
    if model_type in ["ft_transformer", "all"]:
        print("\n使用最佳超参数训练最终FT-Transformer模型...")

        # 创建最终FT-Transformer模型
        final_ft_transformer_model = FTTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=ft_transformer_best_params['d_model'],
            nhead=ft_transformer_best_params['nhead'],
            num_layers=ft_transformer_best_params['num_layers'],
            dropout=ft_transformer_best_params['dropout']
        )

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            final_ft_transformer_model.parameters(),
            lr=ft_transformer_best_params['learning_rate'],
            weight_decay=ft_transformer_best_params['weight_decay']
        )

        # 创建数据加载器
        batch_size = ft_transformer_best_params['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # 训练最终FT-Transformer模型
        final_ft_transformer_model, ft_transformer_train_losses, ft_transformer_val_losses, ft_transformer_train_accs, ft_transformer_val_accs = train_model(
            final_ft_transformer_model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=args.epochs, patience=args.patience
        )

        # 评估最终FT-Transformer模型
        final_ft_transformer_model.eval()
        test_correct = 0
        test_total = 0
        ft_transformer_preds = []
        ft_transformer_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = final_ft_transformer_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                ft_transformer_preds.extend(predicted.cpu().numpy())
                ft_transformer_labels.extend(labels.cpu().numpy())

        ft_transformer_test_acc = test_correct / test_total
        print(f"\n最终FT-Transformer模型测试准确率: {ft_transformer_test_acc:.4f}")

        # 保存最终FT-Transformer模型
        torch.save(final_ft_transformer_model.state_dict(), 'optuna_ft_transformer_best_model.pth')

        # 添加到最终模型列表
        final_models.append(final_ft_transformer_model)

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
        plt.savefig('optuna_ft_transformer_training_history.png')

        # 绘制FT-Transformer优化历史
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(ft_transformer_study)
        plt.title('FT-Transformer优化历史')
        plt.tight_layout()
        plt.savefig('optuna_ft_transformer_optimization_history.png')

        # 绘制FT-Transformer超参数重要性
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(ft_transformer_study)
        plt.title('FT-Transformer超参数重要性')
        plt.tight_layout()
        plt.savefig('optuna_ft_transformer_param_importances.png')

    # 训练SVM模型（如果未禁用）
    if not args.no_svm:
        print("\n训练SVM模型...")

        # 使用GridSearchCV优化SVM超参数
        # 定义参数网格
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }

        # 创建SVM模型
        svm_model = SVC(probability=True)

        # 创建GridSearchCV对象
        grid_search = GridSearchCV(
            svm_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )

        # 训练模型
        print("优化SVM超参数...")
        grid_search.fit(X_train_final, y_train_final)

        # 获取最佳参数
        svm_best_params = grid_search.best_params_
        print(f"SVM最佳参数: {svm_best_params}")

        # 使用最佳参数创建SVM模型
        best_svm = SVC(
            probability=True,
            C=svm_best_params['C'],
            gamma=svm_best_params['gamma'],
            kernel=svm_best_params['kernel']
        )

        # 训练最终SVM模型
        best_svm.fit(X_train_final, y_train_final)

        # 评估SVM模型
        svm_preds = best_svm.predict(X_test_normalized)
        svm_test_acc = accuracy_score(y_test, svm_preds)
        print(f"SVM模型测试准确率: {svm_test_acc:.4f}")

        # 保存SVM模型
        with open('optuna_svm_best_model.pkl', 'wb') as f:
            pickle.dump(best_svm, f)

        # 保存SVM超参数
        with open('optuna_svm_best_params.json', 'w') as f:
            json.dump(svm_best_params, f, indent=4)
    else:
        print("\n跳过SVM模型训练（已禁用）")
        svm_test_acc = 0.0
        best_svm = None

    # 创建集成学习
    print("\n创建集成模型...")

    # 准备所有模型的预测概率
    all_probs = []
    model_weights = []

    # 获取DNN预测概率
    if model_type in ["dnn", "all"]:
        final_dnn_model.eval()
        dnn_probs = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = final_dnn_model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                dnn_probs.extend(probs)
        all_probs.append(np.array(dnn_probs))
        model_weights.append(dnn_test_acc)

    # 获取Transformer预测概率
    if model_type in ["transformer", "all"]:
        final_transformer_model.eval()
        transformer_probs = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = final_transformer_model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                transformer_probs.extend(probs)
        all_probs.append(np.array(transformer_probs))
        model_weights.append(transformer_test_acc)

    # 获取FT-Transformer预测概率
    if model_type in ["ft_transformer", "all"]:
        final_ft_transformer_model.eval()
        ft_transformer_probs = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = final_ft_transformer_model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                ft_transformer_probs.extend(probs)
        all_probs.append(np.array(ft_transformer_probs))
        model_weights.append(ft_transformer_test_acc)

    # 获取SVM预测概率（如果SVM已训练）
    if best_svm is not None:
        svm_probs = best_svm.predict_proba(X_test_normalized)
        all_probs.append(svm_probs)
        model_weights.append(svm_test_acc)

    # 归一化权重
    model_weights = np.array(model_weights) / sum(model_weights)
    print(f"模型权重: {model_weights}")

    # 加权平均概率
    weighted_probs = np.zeros_like(all_probs[0])
    for i, probs in enumerate(all_probs):
        weighted_probs += model_weights[i] * probs

    # 获取最终预测
    ensemble_preds = np.argmax(weighted_probs, axis=1)

    # 评估集成模型
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    print(f"加权集成模型测试准确率: {ensemble_acc:.4f}")

    # 生成集成模型分类报告
    ensemble_report = classification_report(y_test, ensemble_preds, output_dict=True)
    print(f"集成模型精确率 (avg): {ensemble_report['macro avg']['precision']:.4f}")
    print(f"集成模型召回率 (avg): {ensemble_report['macro avg']['recall']:.4f}")
    print(f"集成模型F1分数 (avg): {ensemble_report['macro avg']['f1-score']:.4f}")

    # 比较各模型性能
    plt.figure(figsize=(10, 6))

    model_names = []
    accuracies = []

    if model_type in ["dnn", "all"]:
        model_names.append("DNN")
        accuracies.append(dnn_test_acc)

    if model_type in ["transformer", "all"]:
        model_names.append("Transformer")
        accuracies.append(transformer_test_acc)

    if model_type in ["ft_transformer", "all"]:
        model_names.append("FT-Transformer")
        accuracies.append(ft_transformer_test_acc)

    if best_svm is not None:
        model_names.append("SVM")
        accuracies.append(svm_test_acc)

    model_names.append("加权集成")
    accuracies.append(ensemble_acc)

    plt.bar(model_names, accuracies)
    plt.title('模型准确率比较')
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig('optuna_model_comparison.png')

    print("\n优化完成。结果已保存为PNG文件和JSON文件。")
