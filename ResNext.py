import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnext50_32x4d
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # 设置随机种子保证可复现性
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载CIFAR-10数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)

    # CIFAR-10类别名称
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 修改num_workers为0以避免多进程问题
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # 加载预训练模型并修改最后一层
    model = resnext50_32x4d(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10) 
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 在 main() 函数开头添加
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 训练函数
    def train(model, loader, criterion, optimizer, epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 收集预测和标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        return running_loss / len(loader), 100. * correct / total, cm

    # 验证函数
    def validate(model, loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc="Validation")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 收集预测和标签
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1),
                    'acc': 100. * correct / total
                })
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        return running_loss / len(loader), 100. * correct / total, cm

    # 训练循环
    num_epochs = 20
    best_acc = 0.0
    best_train_cm = None
    best_val_cm = None

    # 创建保存模型的目录
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(num_epochs):
        train_loss, train_acc, train_cm = train(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, val_cm = validate(model, test_loader, criterion)
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_train_cm = train_cm
            best_val_cm = val_cm
            torch.save(model.state_dict(), 'checkpoints/best_resnext50_cifar10.pth')
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print("-" * 50)

    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training curves saved as 'training_curves.png'")
    
    def extract_features(model, loader):
        model.eval()
        features = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Extracting features"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                features.extend(outputs.cpu().numpy())
                labels_list.extend(labels.numpy())
                
        return np.array(features), np.array(labels_list)

    print("Extracting features for t-SNE visualization...")
    test_features, test_labels = extract_features(model, test_loader)
    
    # 随机采样部分数据点（t-SNE计算量大）
    n_samples = 1000  
    indices = np.random.choice(len(test_features), size=n_samples, replace=False)
    sampled_features = test_features[indices]
    sampled_labels = test_labels[indices]
    
    # 执行t-SNE降维
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(sampled_features)
    
    # 创建可视化DataFrame
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['label'] = sampled_labels
    tsne_df['class'] = tsne_df['label'].apply(lambda x: classes[x])
    
    # 绘制t-SNE图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue='class',
        palette=sns.color_palette("hls", 10),
        data=tsne_df,
        legend="full",
        alpha=0.7
    )
    plt.title('t-SNE Visualization of Test Set Features')
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("t-SNE visualization saved as 'tsne_visualization.png'")
    
    # 绘制最佳混淆矩阵
    print("\nTraining set confusion matrix:")
    plot_confusion_matrix(best_train_cm, classes, 
                     title='Training Set Confusion Matrix', 
                     save_path='Training_set_confusion_matrix.png')

    print("\nValidation set confusion matrix:")
    plot_confusion_matrix(best_val_cm, classes, 
                     title='Validation Set Confusion Matrix', 
                     save_path='Validation_set_confusion_matrix.png')

if __name__ == '__main__':
    main()