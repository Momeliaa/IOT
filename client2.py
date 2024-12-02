# client.py
import socket
import pickle
from tqdm import tqdm
import time
import torch
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import struct
from collections import OrderedDict
import warnings
import select

from torchvision.models import mobilenet_v3_small
from efficientnet_pytorch import EfficientNet

warnings.filterwarnings("ignore")

####################################################### 수정 가능 #######################################################
# host = ''  # laptop ip 기입 (실제 연합학습 시 아래 loopback ip는 주석 처리)
host = '127.0.0.1'  # loopback으로 연합학습 수행 시 사용될 ip
port = 8080  # 1024번 ~ 65535번 사이에서 사용
learning_rate = 0.001   # 사용자 편의에 맞게 조정
batch_size = 16   # 사용자 편의에 맞게 조정
epochs = 30   # 사용자 편의에 맞게 조정
partition_id = 1   # jetson 1번은 partition_id = 0, jetson 2번은 partition_id = 1로 수정 후 실행

# Network도 사용자 편의에 맞게 조정 (client와 server의 network와 동일)
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = mobilenet_v3_small(pretrained=False, num_classes=10)

    def forward(self, x):
        return self.model(x)

# efficientNet
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         # EfficientNet B0 모델 사용 (pretrained=False, num_classes=10)
#         self.model = EfficientNet.from_name('efficientnet-b0')  # pretrained=False로 변경 가능
#         self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=10)  # 클래스 수를 10으로 변경
#
#     def forward(self, x):
#         return self.model(x)


train_transform = transforms.Compose([  # 사용자 편의에 맞게 조정
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  # bicubic 방식으로 resize(확대)
    transforms.RandomHorizontalFlip(),                                  # 50% 확률로 좌/우 반전
    transforms.RandomCrop(32, padding=4),                          # 랜덤하게 자름
    transforms.ColorJitter(contrast=0.2),                               # 랜덤하게 constrast 밝기 변경
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),      # 가우시안 노이즈 랜덤하게 추가
    transforms.ToTensor()                                               # 픽셀값 정규화 [0, 1]
])

test_transform = transforms.Compose([   # resize, totensor만 작용(다른 augmentation은 사용x -> 원본 이미지 최대한 훼손x)
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])


class CustomDataset(Dataset):   # 사용자 편의에 맞게 조정
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image, label = sample["img"], sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def train(model, criterion, optimizer, train_loader, test_loader):   # pruning or quantization 적용시 필요한 경우 코드 추가 가능

    best_accuracy = 0.0

    model.to(device)

    for epoch in range(epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0

        for i, (images, labels) in enumerate(tqdm(train_loader, desc="Train"), 0):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_corrects.double() / total

        model.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Test"):

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                accuracy += (predicted==labels).sum().item()
        accuracy = (100 * accuracy / total)
        print(f"Epoch [{epoch + 1}/{epochs}] => Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy * 100:.2f}% | Test Accuracy: {accuracy:.2f}%")

    return model

##############################################################################################################################







####################################################### 수정 금지 ##############################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\nThe model will be running on", device, "device")

def load_partition_data(alpha, partition_id, path_prefix="CIFAR10", transform=None):
    print(f"Partition ID : {partition_id}")
    if partition_id == 0:
        file_path = f"./data/train_image/partition0/alpha_{alpha}_{path_prefix}_partition_{partition_id}.pt"
    else:
        file_path = f"./data/train_image/partition1/alpha_{alpha}_{path_prefix}_partition_{partition_id}.pt"

    data = torch.load(file_path)
    dataset = CustomDataset(data, transform=transform)

    return dataset



def main():
    train_dataset = load_partition_data(alpha=1, partition_id=partition_id, path_prefix="CIFAR10", transform=train_transform)

    test_dataset = torch.load('./data/test_image/test_dataset.pt')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ##############################################################################################################################
    # Quantization or Pruning과 같은 경량화 기법 코드 추가 가능
    model = Network()





    # Optimizer와 Criterion 수정 가능
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    ##############################################################################################################################



    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))

    while True:
        data_size = struct.unpack('>I', client.recv(4))[0]
        rec_payload = b""

        remaining_payload = data_size
        while remaining_payload != 0:
            rec_payload += client.recv(remaining_payload)
            remaining_payload = data_size - len(rec_payload)
        dict_weight = pickle.loads(rec_payload)
        weight = OrderedDict(dict_weight)
        print("\nReceived updated global model from server")

        model.load_state_dict(weight, strict=True)

        read_sockets, _, _ = select.select([client], [], [], 0)
        if read_sockets:
            print("Federated Learning finished")
            break

        model = train(model, criterion, optimizer, train_loader, test_loader)

        model_data = pickle.dumps(dict(model.state_dict().items()))
        client.sendall(struct.pack('>I', len(model_data)))
        client.sendall(model_data)

        print("Sent updated local model to server.")

if __name__ == "__main__":
    time.sleep(1)
    main()

##############################################################################################################################








