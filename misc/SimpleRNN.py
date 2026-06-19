import numpy as np
import torch

class SimpleRNNnumpy:
    def __init__(self, vocab_size, seed=None):
        self.vocab_size = vocab_size
        self.hidden_size = vocab_size # (no tiene capa de salida)

        rng = np.random.RandomState(seed) if seed is not None else np.random

        self.WI = rng.randn(self.hidden_size, self.vocab_size) * 0.01
        self.W = rng.randn(self.hidden_size, self.hidden_size) * 0.01
        self.b = np.zeros((self.hidden_size, 1))

    def forward(self, inputs, y_prev):
        xs, ys = {}, {}
        ys[-1] = np.copy(y_prev)

        for t in range(len(inputs)):
            xs[t] = inputs[t]
            ys[t] = np.tanh(np.dot(self.WI, xs[t]) + np.dot(self.W, ys[t-1]) + self.b)

        return xs, ys

    def backward(self, xs, ys, targets):
        dWI, dW = np.zeros_like(self.WI), np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        for t in range(len(xs)):
            yd = targets[t]

            dha = (ys[t] - yd) # derivada del error cuadrático medio
            for k in range(t, -1, -1): # acá es donde se puede truncar para el truncated BPTT
                dha = dha * (1 - ys[k]) * (1 + ys[k])  # (función de activación sigmoidea simétrica)

                db  += dha
                dWI += np.dot(dha, xs[k].T)
                dW += np.dot(dha, ys[k-1].T)

                dha = np.dot(self.W.T, dha) # propagación hacia el tiempo anterior

        return dWI, dW, db

    def update_params(self, dWI, dW, db, lr=0.1):
        self.WI -= lr * dWI
        self.W -= lr * dW
        self.b -= lr * db

    def train(self, inputs, targets, y_prev, learning_rate=0.1):
        xs, ys = self.forward(inputs, y_prev)
        dWI, dW, db = self.backward(xs, ys, targets)
        self.update_params(dWI, dW, db, learning_rate)

        loss = 0.0
        for t in range(len(inputs)):
            yd = targets[t]
            loss += 0.5 * np.sum((ys[t] - yd) ** 2)
        
        return loss


class SimpleRNNoptim:
    def __init__(self, vocab_size, seed=None):
        self.vocab_size = vocab_size
        self.hidden_size = vocab_size # (no tiene capa de salida)

        rng = np.random.RandomState(seed) if seed is not None else np.random

        self.WI = rng.randn(self.hidden_size, self.vocab_size) * 0.01
        self.W = rng.randn(self.hidden_size, self.hidden_size) * 0.01
        self.b = np.zeros((self.hidden_size, 1))

    def forward(self, inputs, y_prev):
        xs, ys = {}, {}
        ys[-1] = np.copy(y_prev)

        for t in range(len(inputs)):
            xs[t] = inputs[t]
            ys[t] = np.tanh(np.dot(self.WI, xs[t]) + np.dot(self.W, ys[t-1]) + self.b)

        return xs, ys

    def backward(self, xs, ys, targets):
        dWI, dW = np.zeros_like(self.WI), np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dhast = np.zeros_like(ys[0])

        for t in range(len(xs)-1, -1, -1):
            yd = targets[t]

            dhast = (ys[t] - yd) + dhast # sumando aporte que viene del futuro
            dhast = dhast * (1 - ys[t]) * (1 + ys[t])  # (función de activación sigmoidea simétrica)

            db  += dhast
            dWI += np.dot(dhast, xs[t].T)
            dW += np.dot(dhast, ys[t-1].T)

            dhast = np.dot(self.W.T, dhast) # propagación hacia el tiempo anterior

        return dWI, dW, db

    def update_params(self, dWI, dW, db, lr=0.1):
        self.WI -= lr * dWI
        self.W -= lr * dW
        self.b -= lr * db

    def train(self, inputs, targets, y_prev, learning_rate=0.1):
        xs, ys = self.forward(inputs, y_prev)
        dWI, dW, db = self.backward(xs, ys, targets)
        self.update_params(dWI, dW, db, learning_rate)

        loss = 0.0
        for t in range(len(inputs)):
            yd = targets[t]
            loss += 0.5 * np.sum((ys[t] - yd) ** 2)
        
        return loss


class SimpleRNNtorch:
    def __init__(self, vocab_size, device=None, dtype=torch.float32, seed=None):
        self.vocab_size = vocab_size
        self.hidden_size = vocab_size # (no tiene capa de salida)
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        rng = np.random.RandomState(seed) if seed is not None else np.random

        wi = rng.randn(self.hidden_size, self.vocab_size) * 0.01
        w = rng.randn(self.hidden_size, self.hidden_size) * 0.01
        self.WI = torch.tensor(wi, device=self.device, dtype=self.dtype)
        self.W = torch.tensor(w, device=self.device, dtype=self.dtype)
        self.b = torch.zeros((self.hidden_size, 1), device=self.device, dtype=self.dtype)

        self.WI.requires_grad_(True)
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)

    def forward(self, inputs, y_prev):
        xs, ys = {}, {}
        y_prev_t = torch.as_tensor(y_prev, device=self.device, dtype=self.dtype)
        if y_prev_t.ndim == 1:
            y_prev_t = y_prev_t.unsqueeze(1)
        ys[-1] = y_prev_t

        for t in range(len(inputs)):
            xs[t] = torch.as_tensor(inputs[t], device=self.device, dtype=self.dtype)
            if xs[t].ndim == 1:
                xs[t] = xs[t].unsqueeze(1)
            ys[t] = torch.tanh(self.WI @ xs[t] + self.W @ ys[t-1] + self.b)

        return xs, ys

    def update_params(self, loss, lr=0.1):
        loss.backward()
        with torch.no_grad():
            self.WI -= lr * self.WI.grad
            self.W -= lr * self.W.grad
            self.b -= lr * self.b.grad

            self.WI.grad.zero_()
            self.W.grad.zero_()
            self.b.grad.zero_()

    def train(self, inputs, targets, y_prev, learning_rate=0.1):
        xs, ys = self.forward(inputs, y_prev)

        loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for t in range(len(inputs)):
            yd = torch.as_tensor(targets[t], device=self.device, dtype=self.dtype)
            loss += 0.5 * torch.sum((ys[t] - yd) ** 2)
            # acá se podría hacer el grafo desenrollado por pytorch

        self.update_params(loss, learning_rate)
        
        return float(loss.detach().cpu().item())


class SimpleRNNtorchlayer:
    def __init__(self, vocab_size, device=None, dtype=torch.float32, seed=None):
        self.vocab_size = vocab_size
        self.hidden_size = vocab_size # (no tiene capa de salida)
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        rng = np.random.RandomState(seed) if seed is not None else np.random

        # Misma arquitectura: input_size == hidden_size == vocab_size, sin capa de salida.
        self.rnn = torch.nn.RNN(
            input_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            bias=True,
            batch_first=False,
            device=self.device,
            dtype=self.dtype,
        )

        wi = rng.randn(self.hidden_size, self.vocab_size) * 0.01
        w = rng.randn(self.hidden_size, self.hidden_size) * 0.01
        with torch.no_grad():
            self.rnn.weight_ih_l0.copy_(torch.tensor(wi, device=self.device, dtype=self.dtype))
            self.rnn.weight_hh_l0.copy_(torch.tensor(w, device=self.device, dtype=self.dtype))
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()

        # Para conservar el comportamiento de un único bias como en la implementación manual
        self.rnn.bias_hh_l0.requires_grad_(False)

        # Alias para mantener la misma interfaz que el resto de las clases
        self.WI = self.rnn.weight_ih_l0#.cpu().detach().numpy()
        self.W = self.rnn.weight_hh_l0#.cpu().detach().numpy()
        self.b = self.rnn.bias_ih_l0#.cpu().detach().numpy()

    def forward(self, inputs, y_prev):
        xs, ys = {}, {}
        y_prev_t = torch.as_tensor(y_prev, device=self.device, dtype=self.dtype)
        if y_prev_t.ndim == 1:
            y_prev_t = y_prev_t.unsqueeze(1)
        ys[-1] = y_prev_t

        h = y_prev_t.T.unsqueeze(0)  # (1, 1, hidden_size)

        for t in range(len(inputs)):
            xs[t] = torch.as_tensor(inputs[t], device=self.device, dtype=self.dtype)
            if xs[t].ndim == 1:
                xs[t] = xs[t].unsqueeze(1)

            x_t = xs[t].T.unsqueeze(0)  # (1, 1, input_size)
            out_t, h = self.rnn(x_t, h)
            ys[t] = out_t.squeeze(0).T  # (hidden_size, 1)

        return xs, ys

    def update_params(self, loss, lr=0.1):
        loss.backward()
        with torch.no_grad():
            for p in self.rnn.parameters():
                if p.grad is not None:
                    p -= lr * p.grad
                    p.grad.zero_()
    
    def train(self, inputs, targets, y_prev, learning_rate=0.1):
        xs, ys = self.forward(inputs, y_prev)

        loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for t in range(len(inputs)):
            yd = torch.as_tensor(targets[t], device=self.device, dtype=self.dtype)
            loss += 0.5 * torch.sum((ys[t] - yd) ** 2)

        self.update_params(loss, learning_rate)
        
        return float(loss.detach().cpu().item())
