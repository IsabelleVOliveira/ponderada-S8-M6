import numpy as np

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

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
        # Hidden layer
        self.hidden_layer_input = np.dot(inputs, self.weights_input) + self.bias
        self.hidden_layer_output = self._sigmoid(self.hidden_layer_input)

        # Output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_output) + self.output_bias
        self.output = self._sigmoid(self.output_layer_input)

        return self.output
    
    def train(self, training, answers, epochs=9000):
        for epoch in range(epochs):
            for inputs, answer in zip(training, answers):
                final = self.forward_pass(inputs)
                error = answer - final
                # Gradient for output
                grad_final = error * self._sigmoid_derivative(final)

                # Hidden layer error
                final_hidden_layer = grad_final.dot(self.weights_output.T)
                grad_hidden_layer = final_hidden_layer * self._sigmoid_derivative(self.hidden_layer_output)

                # Updating weights and biases
                self.weights_output += np.dot(self.hidden_layer_output[:, np.newaxis], grad_final[np.newaxis, :]) * self.learning_rate
                self.output_bias += grad_final * self.learning_rate

                self.weights_input += np.dot(np.array(inputs)[:, np.newaxis], grad_hidden_layer[np.newaxis, :]) * self.learning_rate
                self.bias += grad_hidden_layer * self.learning_rate

            if epoch % 1000 == 0:
                predictions, _ = self.predict(training)
                mse = self._mse(predictions, answers)
                print(f'Epoch: {epoch}, MSE: {mse}')

    def predict(self, inputs):
        predictions = []
        binary_predictions = []
        for inp in inputs:
            output = self.forward_pass(inp)
            predictions.append(output)
            binary_predictions.append(1 if output >= 0.5 else 0)
        return np.array(predictions).reshape(-1, 1), np.array(binary_predictions).reshape(-1, 1)
    
    def _mse(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)

def main():
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
    
    perceptron = Perceptron(input_size=2, hidden_size=2, output_size=1)
    perceptron.train(training, answers)
    
    predictions, binary_predictions = perceptron.predict(training)
    mse = perceptron._mse(predictions, answers)
    print(f'Final MSE: {mse}')
    print('Predições:')
    print(predictions)
    print('Predição final:')
    print(binary_predictions)

if __name__ == "__main__":
    main()