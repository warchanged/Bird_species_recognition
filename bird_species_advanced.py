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

# 增强的DNN模型 - 添加L2归一化和BatchNorm
class EnhancedBirdDNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers=[1024, 512, 256], dropout_rate=0.5):
        super(EnhancedBirdDNN, self).__init__()

        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.LayerNorm(hidden_layers[0]))  # L2归一化
        layers.append(nn.Dropout(dropout_rate))

        # 隐藏层
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            layers.append(nn.LayerNorm(hidden_layers[i+1]))  # L2归一化
            layers.append(nn.Dropout(dropout_rate))

        # 输出层
        layers.append(nn.Linear(hidden_layers[-1], num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Transformer模型
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

# 超参数调优函数 - 使用网格搜索
def grid_search_hyperparameters(X_train, y_train, X_val, y_val, input_dim, num_classes, device):
    # 定义超参数网格 (减少搜索空间以提高效率)
    param_grid = {
        'hidden_layers': [
            [512, 256],
            [1024, 512, 256]
        ],
        'dropout_rate': [0.3, 0.5],
        'learning_rate': [0.001, 0.0001],
        'weight_decay': [1e-4],
        'batch_size': [64]
    }

    # 创建结果存储
    results = []

    # 遍历所有超参数组合
    for hidden_layers in param_grid['hidden_layers']:
        for dropout_rate in param_grid['dropout_rate']:
            for learning_rate in param_grid['learning_rate']:
                for weight_decay in param_grid['weight_decay']:
                    for batch_size in param_grid['batch_size']:
                        print(f"\nTesting hyperparameters: hidden_layers={hidden_layers}, dropout_rate={dropout_rate}, "
                              f"learning_rate={learning_rate}, weight_decay={weight_decay}, batch_size={batch_size}")

                        # 创建模型
                        model = EnhancedBirdDNN(
                            input_dim=input_dim,
                            num_classes=num_classes,
                            hidden_layers=hidden_layers,
                            dropout_rate=dropout_rate
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
                        try:
                            trained_model, _, _, _, _ = train_model(
                                model, train_loader, val_loader, criterion, optimizer, device,
                                num_epochs=15, patience=3  # 进一步减少训练时间
                            )

                            # 评估模型
                            model.eval()
                            val_loss = 0.0
                            correct = 0
                            total = 0

                            with torch.no_grad():
                                for inputs, labels in val_loader:
                                    inputs, labels = inputs.to(device), labels.to(device)
                                    outputs = model(inputs)
                                    loss = criterion(outputs, labels)

                                    val_loss += loss.item() * inputs.size(0)
                                    _, predicted = torch.max(outputs.data, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()

                            val_acc = correct / total

                            # 存储结果
                            results.append({
                                'hidden_layers': hidden_layers,
                                'dropout_rate': dropout_rate,
                                'learning_rate': learning_rate,
                                'weight_decay': weight_decay,
                                'batch_size': batch_size,
                                'val_acc': val_acc,
                                'model': trained_model
                            })

                            print(f"Validation accuracy: {val_acc:.4f}")
                        except Exception as e:
                            print(f"Error during training: {e}")

    # 找到最佳超参数
    best_result = max(results, key=lambda x: x['val_acc'])
    print(f"\nBest hyperparameters: hidden_layers={best_result['hidden_layers']}, "
          f"dropout_rate={best_result['dropout_rate']}, learning_rate={best_result['learning_rate']}, "
          f"weight_decay={best_result['weight_decay']}, batch_size={best_result['batch_size']}")
    print(f"Best validation accuracy: {best_result['val_acc']:.4f}")

    return best_result

# 集成学习 - 结合DNN + Transformer + SVM
class PyTorchModelWrapper:
    """包装PyTorch模型以兼容scikit-learn API"""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def create_ensemble_predictions(dnn_model, transformer_model, svm_model, X_test, device):
    # 获取DNN预测
    dnn_wrapper = PyTorchModelWrapper(dnn_model, device)
    dnn_probs = dnn_wrapper.predict_proba(X_test)

    # 获取Transformer预测
    transformer_wrapper = PyTorchModelWrapper(transformer_model, device)
    transformer_probs = transformer_wrapper.predict_proba(X_test)

    # 获取SVM预测
    svm_probs = svm_model.predict_proba(X_test)

    # 组合预测 (平均概率)
    ensemble_probs = (dnn_probs + transformer_probs + svm_probs) / 3
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    return ensemble_preds, ensemble_probs

# 主函数
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

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

    # 创建数据加载器
    batch_size = 64
    train_dataset = BirdDataset(X_train_final, y_train_final)
    val_dataset = BirdDataset(X_val, y_val)
    test_dataset = BirdDataset(X_test_normalized, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 是否执行超参数调优 (耗时较长，可以设置为False跳过)
    do_hyperparameter_tuning = True

    if do_hyperparameter_tuning:
        print("\n开始超参数调优...")
        best_params = grid_search_hyperparameters(
            X_train_final, y_train_final, X_val, y_val, input_dim, num_classes, device
        )

        # 使用最佳超参数创建DNN模型
        dnn_model = EnhancedBirdDNN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=best_params['hidden_layers'],
            dropout_rate=best_params['dropout_rate']
        )

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            dnn_model.parameters(),
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
        )

        # 重新创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
    else:
        # 使用默认超参数创建DNN模型
        dnn_model = EnhancedBirdDNN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=[1024, 512, 256],
            dropout_rate=0.5
        )

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(dnn_model.parameters(), lr=0.001, weight_decay=1e-4)

    # 训练DNN模型
    print("\n训练增强型DNN模型...")
    dnn_model, dnn_train_losses, dnn_val_losses, dnn_train_accs, dnn_val_accs = train_model(
        dnn_model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, patience=10
    )

    # 评估DNN模型
    dnn_test_loss, dnn_test_acc, dnn_preds, dnn_true = evaluate_model(
        dnn_model, test_loader, criterion, device
    )
    print(f"DNN模型测试准确率: {dnn_test_acc:.4f}")

    # 训练Transformer模型
    print("\n训练Transformer模型...")
    transformer_model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=512,
        nhead=8,
        num_layers=3,
        dropout=0.1
    )

    # 定义损失函数和优化器
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.0001, weight_decay=1e-4)

    # 训练Transformer模型
    transformer_model, transformer_train_losses, transformer_val_losses, transformer_train_accs, transformer_val_accs = train_model(
        transformer_model, train_loader, val_loader, criterion, transformer_optimizer, device, num_epochs=30, patience=10
    )

    # 评估Transformer模型
    transformer_test_loss, transformer_test_acc, transformer_preds, transformer_true = evaluate_model(
        transformer_model, test_loader, criterion, device
    )
    print(f"Transformer模型测试准确率: {transformer_test_acc:.4f}")

    # 训练SVM模型
    print("\n训练SVM模型...")
    svm_model = SVC(probability=True, C=10, gamma='scale', kernel='rbf')
    svm_model.fit(X_train_final, y_train_final)

    # 评估SVM模型
    svm_preds = svm_model.predict(X_test_normalized)
    svm_test_acc = accuracy_score(y_test, svm_preds)
    print(f"SVM模型测试准确率: {svm_test_acc:.4f}")

    # 创建集成预测
    print("\n创建集成预测...")
    ensemble_preds, ensemble_probs = create_ensemble_predictions(
        dnn_model, transformer_model, svm_model, X_test_normalized, device
    )

    # 评估集成模型
    ensemble_test_acc = accuracy_score(y_test, ensemble_preds)
    print(f"集成模型测试准确率: {ensemble_test_acc:.4f}")

    # 生成分类报告
    print("\n生成分类报告...")

    # 获取类名
    class_id_to_name = {}
    for name, _, class_id, _ in training_data:
        if '\\' in name:
            class_name = name.split('\\')[0]
            if '.' in class_name:
                class_name = class_name.split('.', 1)[1]
            class_id_to_name[class_id - 1] = class_name  # 减1使类别从0开始

    # 计算总体指标
    dnn_report = classification_report(dnn_true, dnn_preds, output_dict=True)
    transformer_report = classification_report(transformer_true, transformer_preds, output_dict=True)
    svm_report = classification_report(y_test, svm_preds, output_dict=True)
    ensemble_report = classification_report(y_test, ensemble_preds, output_dict=True)

    print("\nDNN模型性能:")
    print(f"精确率 (avg): {dnn_report['macro avg']['precision']:.4f}")
    print(f"召回率 (avg): {dnn_report['macro avg']['recall']:.4f}")
    print(f"F1分数 (avg): {dnn_report['macro avg']['f1-score']:.4f}")

    print("\nTransformer模型性能:")
    print(f"精确率 (avg): {transformer_report['macro avg']['precision']:.4f}")
    print(f"召回率 (avg): {transformer_report['macro avg']['recall']:.4f}")
    print(f"F1分数 (avg): {transformer_report['macro avg']['f1-score']:.4f}")

    print("\nSVM模型性能:")
    print(f"精确率 (avg): {svm_report['macro avg']['precision']:.4f}")
    print(f"召回率 (avg): {svm_report['macro avg']['recall']:.4f}")
    print(f"F1分数 (avg): {svm_report['macro avg']['f1-score']:.4f}")

    print("\n集成模型性能:")
    print(f"精确率 (avg): {ensemble_report['macro avg']['precision']:.4f}")
    print(f"召回率 (avg): {ensemble_report['macro avg']['recall']:.4f}")
    print(f"F1分数 (avg): {ensemble_report['macro avg']['f1-score']:.4f}")

    # 绘制模型比较图表
    plt.figure(figsize=(12, 6))

    # 准确率比较
    plt.subplot(1, 2, 1)
    models = ['DNN', 'Transformer', 'SVM', 'Ensemble']
    accuracies = [dnn_test_acc, transformer_test_acc, svm_test_acc, ensemble_test_acc]
    plt.bar(models, accuracies)
    plt.title('模型准确率比较')
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.ylim(0.5, 1.0)

    # F1分数比较
    plt.subplot(1, 2, 2)
    f1_scores = [
        dnn_report['macro avg']['f1-score'],
        transformer_report['macro avg']['f1-score'],
        svm_report['macro avg']['f1-score'],
        ensemble_report['macro avg']['f1-score']
    ]
    plt.bar(models, f1_scores)
    plt.title('模型F1分数比较')
    plt.xlabel('模型')
    plt.ylabel('F1分数')
    plt.ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig('model_comparison_advanced.png')

    # 绘制DNN和Transformer的训练历史
    plt.figure(figsize=(12, 10))

    # DNN准确率
    plt.subplot(2, 2, 1)
    plt.plot(dnn_train_accs)
    plt.plot(dnn_val_accs)
    plt.title('DNN - 准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend(['训练', '验证'])

    # DNN损失
    plt.subplot(2, 2, 2)
    plt.plot(dnn_train_losses)
    plt.plot(dnn_val_losses)
    plt.title('DNN - 损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend(['训练', '验证'])

    # Transformer准确率
    plt.subplot(2, 2, 3)
    plt.plot(transformer_train_accs)
    plt.plot(transformer_val_accs)
    plt.title('Transformer - 准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend(['训练', '验证'])

    # Transformer损失
    plt.subplot(2, 2, 4)
    plt.plot(transformer_train_losses)
    plt.plot(transformer_val_losses)
    plt.title('Transformer - 损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend(['训练', '验证'])

    plt.tight_layout()
    plt.savefig('training_history_advanced.png')

    # 保存最佳模型
    torch.save(dnn_model.state_dict(), 'dnn_model.pth')
    torch.save(transformer_model.state_dict(), 'transformer_model.pth')

    print("\n分析完成。结果已保存为PNG文件。")
