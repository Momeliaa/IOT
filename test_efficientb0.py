import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0
from torchvision import transforms
import os
from torchvision.transforms import InterpolationMode
import warnings
from torch.nn.utils import prune

warnings.filterwarnings("ignore")

# 1. Custom Dataset 클래스 정의
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['img']
        label = sample['label']
        if self.transform:
            image = self.transform(image)
        return image, label


# 2. 데이터 로드 및 전처리
# CIFAR-10 수정 데이터셋 로드
data = torch.load('data\\train_image\\partition0\\alpha_1_CIFAR10_partition_0.pt')  # 수정된 데이터셋
train_data, val_data = data[:int(0.8 * len(data))], data[int(0.8 * len(data)):]  # 80/20 split


# 데이터 증강 파이프라인 (학습용)
train_transform = transforms.Compose([
    # BILINEAR 보간법 적용
    transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),                               # 텐서로 변환(0~1 사이 정규화)
    transforms.Normalize((0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201))
])

# 검증용 데이터 전처리 (증강 미적용, 테스트용)
val_transform = transforms.Compose([
    # BILINEAR 보간법 적용
    transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201))
])

# 데이터 로더 생성
train_dataset = CustomCIFAR10Dataset(train_data, transform=train_transform)
val_dataset = CustomCIFAR10Dataset(val_data, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 배치 사이즈 16
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 3. 모델 정의(mobileNetV3_small)
model = efficientnet_b0(weights=None)  # Pretrained 대신 weights 사용 권장
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)  # CIFAR-10 수정 데이터에 맞게 MobileNet의 마지막 계층 조정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\nThe model will be running on", device, "device")
model = model.to(device)


# focalloss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha      # 클래스 불균형 조정()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # 예측 확률
        focal_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return focal_loss.mean()


# 손실 함수와 옵티마이저
# criterion = nn.CrossEntropyLoss()

# 데이터 불균형 문제를 해결하기 위해 focal loss 적용
criterion = FocalLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 학습률 0.001


# 4. 정확도 계산 함수
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 5. 학습 루프
num_epochs = 30  # 에포크 30
start_time = time.time()

for epoch in range(num_epochs):
    model.train()  # 학습 모드
    running_loss = 0.0
    epoch_start = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 에포크 종료 후 정확도 계산
    train_acc = calculate_accuracy(train_loader, model)
    val_acc = calculate_accuracy(val_loader, model)
    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%, Time: {epoch_time:.2f}s")


training_time = time.time() - start_time

# 6. 모델 크기 계산
model_save_path = 'efficientNetB0.pth'
torch.save(model.state_dict(), model_save_path)
model_size = os.path.getsize(model_save_path) / (1024 * 1024)  # 모델 크기(MB)

# 7. 단일 예측 소요 시간 계산
sample_image, _ = val_dataset[0]
sample_image = sample_image.unsqueeze(0).to(device)  # 배치 차원 추가
start_infer = time.time()
model.eval()
with torch.no_grad():
    _ = model(sample_image)
inference_time = (time.time() - start_infer) * 1000  # 단일 예측 시간(ms)

# 8. 최종 결과 출력
print(f"\nTraining Time: {training_time:.2f}s")
print(f"Model Size: {model_size:.2f} MB")
print(f"Single Inference Time: {inference_time:.2f} ms")
