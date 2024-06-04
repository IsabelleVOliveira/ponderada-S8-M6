import numpy as np

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Inicialização aleatória dos pesos e vieses
        self.weights_input = np.random.rand(self.input_size, self.hidden_size)
        self.weights_output = np.random.rand(self.hidden_size, self.output_size)

        self.bias = np.random.rand(self.hidden_size)
        self.output_bias = np.random.rand(self.output_size)

    def _sigmoid(self, x):
        """
        Implementa a função sigmoide
        Essa é uma função de ativação possível para os nós da rede neural.
        """
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        """
        Derivada da função sigmoide
        """
        return x * (1 - x)
    
    def forward_pass(self, inputs):
        # Calcula a saída da camada oculta
        self.hidden_layer_input = np.dot(inputs, self.weights_input) + self.bias
        self.hidden_layer_output = self._sigmoid(self.hidden_layer_input)

        # Calcula a saída da camada de saída
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_output) + self.output_bias
        self.output = self._sigmoid(self.output_layer_input)

        return self.output
    
    def train(self, training, answers, epochs=10000):
        for epoch in range(epochs):
            for inputs, answer in zip(training, answers):
                # Passagem direta para obter a saída
                final = self.forward_pass(inputs)
                # Calcula o erro
                error = answer - final
                
                # Gradiente para a camada de saída
                grad_final = error * self._sigmoid_derivative(final)

                # Calcula o erro da camada oculta
                final_hidden_layer = grad_final.dot(self.weights_output.T)
                grad_hidden_layer = final_hidden_layer * self._sigmoid_derivative(self.hidden_layer_output)

                # Atualiza os pesos e os vieses usando o gradiente descendente
                self.weights_output += self.hidden_layer_output.reshape(-1, 1) @ grad_final.reshape(1, -1) * self.learning_rate
                self.output_bias += grad_final * self.learning_rate

                self.weights_input += inputs.reshape(-1, 1) @ grad_hidden_layer.reshape(1, -1) * self.learning_rate
                self.bias += grad_hidden_layer * self.learning_rate

            if epoch % 1000 == 0:
                # Avalia o desempenho do modelo a cada 1000 épocas
                predictions = self.predict(training)
                mse = self._mse(predictions, answers)
                print(f'Epoch: {epoch}, MSE: {mse}')

    def predict(self, inputs):
        # Faz previsões para cada entrada
        return np.array([self.forward_pass(inp) for inp in inputs])
    
    def _mse(self, predictions, targets):
        # Calcula o erro médio quadrático
        return np.mean((predictions - targets) ** 2)

def main():
    # Dados de treinamento e respostas desejadas
    training = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    answers = np.array([
        [0],
        [1],
        [1],
        [0]
    ])
    
    # Criação e treinamento do perceptron
    perceptron = Perceptron(input_size=2, hidden_size=2, output_size=1)
    perceptron.train(training, answers)
    
    # Faz previsões e calcula o erro final
    predictions = perceptron.predict(training)
    mse = perceptron._mse(predictions, answers)
    print(f'Final MSE: {mse}')
    print('Predições:')
    print(predictions)
    print('Predição final:')
    print(np.round(predictions).astype(int))

if __name__ == "__main__":
    main()