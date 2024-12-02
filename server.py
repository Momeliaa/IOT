# Laptop code
import threading
import socket
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision.transforms as transforms
import struct
from tqdm import tqdm
import copy
import warnings
import random

from torchvision.models import mobilenet_v3_small
from efficientnet_pytorch import EfficientNet

warnings.filterwarnings("ignore")

####################################################### 수정 가능 #######################################################
target_accuracy = 1.0  # 사용자 편의에 맞게 조정
global_round = 1   # 사용자 편의에 맞게 조정

batch_size = 16  # 사용자 편의에 맞게 조정
num_samples = 10000   # 사용자 편의에 맞게 조정
# host = '192.168.35.232'  # laptop ip 기입 (실제 연합학습 시 아래 loopback ip는 주석 처리)
host = '127.0.0.1' # loopback으로 연합학습 수행 시 사용될 ip
port = 8080 # 1024번 ~ 65535번

test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])
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




def measure_accuracy(global_model, test_loader):  # pruning or quantization 적용시 필요한 경우 코드 추가 가능
    model = Network()
    model.load_state_dict(global_model)
    model.to(device)
    model.eval()

    accuracy = 0.0
    total = 0.0

    inference_start = time.time()
    with torch.no_grad():
        print("\n")
        for inputs, labels in tqdm(test_loader, desc="Test"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

        accuracy = (100 * accuracy / total)
    inference_end = time.time()

    print(f"Inference time for {num_samples} images : {(inference_end - inference_start):.2f} seconds")

    return accuracy, model
##############################################################################################################################






####################################################### 수정 금지 ##############################################################
cnt = []
models = []  # 수신받은 model 저장할 리스트
semaphore = threading.Semaphore(0)

global_model = None
global_accuracy = 0.0
current_round = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def handle_client(conn, addr, model, test_loader):
    global models, global_model, global_accuracy, current_round, cnt
    print(f"Connected by {addr}")

    while True:
        if len(cnt) < 2:
            cnt.append(1)
            weight = pickle.dumps(dict(model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)

        data_size = struct.unpack('>I', conn.recv(4))[0]
        received_payload = b""
        remaining_payload_size = data_size
        while remaining_payload_size != 0:
            received_payload += conn.recv(remaining_payload_size)
            remaining_payload_size = data_size - len(received_payload)
        model = pickle.loads(received_payload)

        models.append(model)

        if len(models) == 2:
            current_round += 1
            global_model = average_models(models)
            global_accuracy, global_model = measure_accuracy(global_model, test_loader)
            print(f"Global round [{current_round} / {global_round}] Accuracy : {global_accuracy}%")
            get_model_size(global_model)
            models = []
            semaphore.release()
        else:
            semaphore.acquire()

        if (current_round == global_round) or (global_accuracy >= target_accuracy):
            weight = pickle.dumps(dict(global_model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)
            conn.close()
            break
        else:
            weight = pickle.dumps(dict(global_model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)

def get_model_size(global_model):
    model_size = len(pickle.dumps(dict(global_model.state_dict().items())))
    print(f"Model size : {model_size / (1024 ** 2):.4f} MB")

def get_random_subset(dataset, num_samples):
    if num_samples > len(dataset):
        raise ValueError(f"num_samples should not exceed {len(dataset)} (total number of samples in test dataset).")

    indices = random.sample(range(len(dataset)), num_samples)
    subset = Subset(dataset, indices)
    return subset

def average_models(models):
    weight_avg = copy.deepcopy(models[0])

    for key in weight_avg.keys():
        for i in range(1, len(models)):
            weight_avg[key] += models[i][key]
        weight_avg[key] = torch.div(weight_avg[key], len(models))

    return weight_avg


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen()
    connection = []
    address = []

    test_dataset = torch.load('./data/test_image/test_dataset.pt')

    test_dataset = get_random_subset(test_dataset, num_samples)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Server is listening on {host}:{port}")
    model = Network()

    while len(address) < 2 and len(connection) < 2:
        conn, addr = server.accept()
        connection.append(conn)
        address.append(addr)

    training_start = time.time()

    connection1 = threading.Thread(target=handle_client, args=(connection[0], address[0], model, test_loader))
    connection2 = threading.Thread(target=handle_client, args=(connection[1], address[1], model, test_loader))

    connection1.start();connection2.start()
    connection1.join();connection2.join()

    training_end = time.time()
    total_time = training_end - training_start
    print(f"\nTraining time: {int(total_time // 3600)} hours {int((total_time % 3600) // 60)} minutes {(total_time % 60):.2f} seconds")

    print("Federated Learning finished")


if __name__ == "__main__":
    main()
##############################################################################################################################

