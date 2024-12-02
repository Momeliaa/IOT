import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
import os
from torchvision.transforms import InterpolationMode
import warnings

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

# 가우시안 노이즈 추가 클래스 정의
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if isinstance(img, torch.Tensor):  # 이미지가 텐서일 경우
            noise = torch.randn(img.size()) * self.std + self.mean
            img = img + noise
            img = torch.clamp(img, 0.0, 1.0)  # 값 범위를 [0, 1]로 제한
        return img

# 데이터 증강 파이프라인 (학습용)
train_transform = transforms.Compose([
    # BICUBIC 보간법 적용
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2), interpolation=InterpolationMode.BICUBIC),

    # transforms.Resize((224, 224)),

    # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2), ),  # 확대 및 크롭
    transforms.RandomHorizontalFlip(p=0.5),              # 좌우 반전
    transforms.ColorJitter(contrast=0.5),                # 대비 조정
    AddGaussianNoise(mean=0.0, std=0.05),                # 가우시안 노이즈 추가

    transforms.ToTensor(),                               # 텐서로 변환(0~1 사이 정규화)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -1~1 사이 표준화
])

# 검증용 데이터 전처리 (증강 미적용)
val_transform = transforms.Compose([
    # BICUBIC 보간법 적용
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),

    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 데이터 로더 생성
train_dataset = CustomCIFAR10Dataset(train_data, transform=train_transform)
val_dataset = CustomCIFAR10Dataset(val_data, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 배치 사이즈 16
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 3. 모델 정의(mobileNetV3_small)
model = mobilenet_v3_small(weights=None)  # Pretrained 대신 weights 사용 권장
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 10)  # CIFAR-10 수정 데이터에 맞게 MobileNet의 마지막 계층 조정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\nThe model will be running on", device, "device")
model = model.to(device)

# 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
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
model_save_path = 'shufflenet_custom.pth'
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
