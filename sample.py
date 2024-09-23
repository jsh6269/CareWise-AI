import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 재현성을 위해 난수 고정
np.random.seed(100)

# 1. y = 2x + 1 에 가까운 데이터 생성
data_x = [1.1, 2, 3, 4, 5, 6, 7, 8.5, 9, 10, 11, 12, 13.1, 14, 15]
data_y = [2, 5.2, 7.1, 9.2, 11.1, 13.2, 15, 17.1, 18, 21.2, 23.1, 25, 27.1, 29, 30.2]

X = torch.tensor(np.array([[item] for item in data_x]), dtype=torch.float32)
y = torch.tensor(np.array(data_y), dtype=torch.float32)


# 2. 간단한 신경망 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 입력 크기 1, 출력 크기 1의 단순 선형 레이어
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 3. 모델 초기화
model = SimpleNet()

# 4. 손실 함수와 optimizer 정의
criterion = nn.MSELoss()  # 평균 제곱 오차 손실 함수
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 확률적 경사하강법(SGD)

# 5. 학습 과정
num_epochs = 20
for epoch in range(num_epochs):
    # 순전파 (forward pass)
    outputs = model(X)
    loss = criterion(outputs, y.view(-1, 1))

    # 역전파 (backward pass) 및 최적화
    optimizer.zero_grad()  # 기존의 기울기를 0으로 초기화
    loss.backward()  # 기울기 계산
    optimizer.step()  # 가중치 업데이트

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. 최종 결과
print('예측된 기울기: ', model.linear.weight.tolist()[0])
print('예측된 상수항: ', float(model.linear.bias))

test_x = np.arange(1, 15, 0.1)
X = torch.tensor(np.array([[item] for item in test_x]), dtype=torch.float32)

with torch.no_grad():  # 기울기 계산을 하지 않도록 설정
    predicted_y = model(X)

# 원본 데이터와 예측된 데이터를 시각화
plt.figure(figsize=(10, 6))
plt.scatter(data_x, data_y, color='blue', label='Original Data')  # 실제 데이터
plt.plot(test_x, predicted_y.numpy(), color='red', label='Predicted Line')  # 예측된 선
plt.title('Model Prediction vs Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()
