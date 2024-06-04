import torch
import torch.nn as nn
import torch.optim as optim

# Definindo o Perceptron com uma camada escondida
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden = nn.Linear(2, 4)  # Aumentando o número de neurônios na camada escondida
        self.output = nn.Linear(4, 1)  # Conectando a camada escondida à camada de saída
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

# Definindo a função de custo
criterion = nn.MSELoss()

# Dados de treinamento para a porta XOR
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Inicializando o modelo, o otimizador e a função de custo
model = XORModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Reduzindo a taxa de aprendizado

# Treinamento do modelo
epochs = 10000  # Aumentando o número de épocas
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward pass e otimização
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Validação do modelo
model.eval()
with torch.no_grad():
    y_pred = model(X)
    predicted = (y_pred > 0.5).float()
    accuracy = (predicted == y).sum().item() / y.shape[0]
    print(f'Predicted:\n{predicted}\nAccuracy: {accuracy * 100:.2f}%')