import numpy as np
import torch

from SimpleRNN import SimpleRNNnumpy
from SimpleRNN import SimpleRNNoptim
from SimpleRNN import SimpleRNNtorch
from SimpleRNN import SimpleRNNtorchlayer


sentences = [
    "tenemos excelentes estudiantes",
    "la asignatura es buena pero mejorable",
    "el sueldo universitario es bajo"
]

all_words = " ".join(sentences).split()
vocab = sorted(list(set(all_words)))
vocab_size = len(vocab)

# Conversiones entre palabras, índices y vectores one-hot
word_to_ix = {}
ix_to_word = {}
for i, word in enumerate(vocab):
    word_to_ix[word] = i
    ix_to_word[i] = word

def word2onehot(word):
    vec = np.zeros((vocab_size, 1))
    vec[word_to_ix[word]] = 1
    return vec

def onehot2word(vec):
    if isinstance(vec, torch.Tensor):
        vec = vec.detach().cpu().numpy()
    idx = int(np.argmax(vec))
    return ix_to_word[idx]

def input2onehot(sentence):
    words = sentence.split()
    soh = []
    for w in words[:-1]:
        soh.append(word2onehot(w))
    return soh

def target2onehot(sentence):
    words = sentence.split()
    soh = []
    for w in words[1:]: # uno corrido (la salida deseada es la siguiente palabra)
        soh.append(word2onehot(w))
    return soh

def sentences2onehot(sentences):
    inputs = []
    targets = []
    for sentence in sentences:
        inputs.append(input2onehot(sentence))
        targets.append(target2onehot(sentence))
    return inputs, targets

# Funciones de entrenamiento y generación de texto
def train_model(rnn, inputs, targets, learning_rate=0.05, epochs=2000):
    loss_history = []
    y_prev = np.zeros((rnn.hidden_size, 1))

    print("Época -- Error")
    for i in range(epochs):
        epoch_loss = 0.0
        for x, y_target in zip(inputs, targets):

            loss = rnn.train(x, y_target, y_prev, learning_rate)

            epoch_loss += loss

        loss_history.append(epoch_loss)
        if i % 500 == 0:
            print(f"{i:5d} -- {epoch_loss:.4f}")

    return loss_history


def generate_with(rnn, seed_word, length=8):
    y = np.zeros((rnn.hidden_size, 1))
    x = word2onehot(seed_word)
    
    result = [seed_word]
    for iw in range(length - 1):
        xs, ys = rnn.forward([x], y)
        y = ys[0]
        yw = onehot2word(y)
        x = word2onehot(yw) # en la próxima entra la salida anterior en onbe-hot
        #x = y                # en la próxima entra la misma salida como entrada también

        result.append(yw)

    return " ".join(result)

# Función de entrenamiento y generación para un modelo
def run_traintest(model, seed=None):
    
    rnn = model(vocab_size, seed=seed)
    
    print(f"\n===== {model.__name__}: entrenando  =====")
    inputs, targets = sentences2onehot(sentences)
    loss_history = train_model(rnn, inputs, targets)
    print(f"final:   {loss_history[-1]:.4f}")

    print(f"\n Pesos finales:")
    print(f"WI: {rnn.WI[0:5,0]}...")
    print(f"W:  {rnn.W[0:5,0]}...")
    print(f"b:  {rnn.b[0:5].squeeze()}...")

    print(f"\n===== {model.__name__}: generando frases =====")
    print("Inicio con 'tenemos' y longitud exacta:")
    print(">> ", generate_with(rnn, "tenemos", 3))

    print("Inicio con 'la' y longitud exacta:")
    print(">> ", generate_with(rnn, "la", 6))

    print("Inicio con 'el' y longitud exacta:")
    print(">> ", generate_with(rnn, "el", 5))

    print("Inicio con 'el' y más longitud:")
    print(">> ", generate_with(rnn, "el", 7))


if __name__ == "__main__":
    common_seed = 33
    run_traintest(SimpleRNNnumpy, seed=common_seed)
    run_traintest(SimpleRNNoptim, seed=common_seed)
    run_traintest(SimpleRNNtorch, seed=common_seed)
    run_traintest(SimpleRNNtorchlayer, seed=common_seed)
