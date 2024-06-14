import numpy as np
#from scipy.sparse import lil_matrix
import torch
from torch_geometric.data import Data
from tqdm import tqdm

##############################################################################
# FUNCIONES GENERALES
########################



#===============================================================
def build_pattern(name, sequence, real_base_pairs, Dmin=4, window=9):

    assert isinstance(real_base_pairs, list), '"real_base_pairs" debe ser una lista de listas/tuplas.'

    bps = np.array(real_base_pairs) - 1

    nts = ('A','U','C','G')

    #------------------------
    # SECUENCIA A EMBEDDING
    # Cada nt es un one-hot
    #------------------------
    embedding = seq2embedding(sequence, nts=nts)

    #-----------------------------
    # GENERAR POSITIONAL EMCODING
    #-----------------------------
    #positional_embedding = getPositionEncoding(seq_len=len(sequence), d=len(nts), n=100)  # Mismo tamaño que las features??

    #----------------------------------------
    # SUMAR POSITIONAL ENCODING AL EMBEDDING
    #----------------------------------------



    #-----------------------------
    # IDENTIFICAR POSIBLES ENLACES (GC, AU, GU)
    #-----------------------------
    backbone = get_backbone(sequence)

    connections, strength = get_connections(sequence, pairings=[('G','C'),('A','U'), ('G','U')], Dmin=4)

    target = backbone[1][:]
    target.extend(connections[1])
    source = backbone[0][:]
    source.extend(connections[0])
    edge_index = np.array([target,source])  # INCLUYE LA ESTRUCTURA DE UNION ENTRE nts

    # edge_index = np.array(connections)  # SOLO CANONICAS


    #-----------------------------
    # GENERAR LISTADO DE ENLACES VALIDOS
    #-----------------------------
    connections = np.array(connections)

    y = np.zeros(connections.shape[1])

    for bp in bps:

        t = np.where(connections[1,:]==bp[0])
        s = np.where(connections[0,:]==bp[1])
        Q = np.intersect1d(s[0],t[0])

        if len(Q) > 0:
            i = Q[0]
            y[i] = 1

        t = np.where(connections[1,:]==bp[1])
        s = np.where(connections[0,:]==bp[0])
        Q = np.intersect1d(s[0],t[0])

        if len(Q) > 0:
            i = Q[0]
            y[i] = 1
        # else:
        #     print(f'{name}, {sequence} -- BP: {sequence[int(bp[0])]} ({bp[0]}) - {sequence[int(bp[1])]} ({bp[1]}), {name}')

    y *= np.array(strength)


    #-----------------------------
    # CONSTRUIR PATRON
    #-----------------------------
    pattern = Data(name=name,                                  # CODIGO IDENTIFICADOR DE LA SECUENCIA
                   seq=sequence,                               # SECUENCIA DE nts
                   family=name.split('_')[0],                  # FAMILIA DE LA SECUENCIA
                   x=torch.Tensor(embedding),                  # REPRESENTACION ONE-HOT DE LOS nts
                   backbone=torch.Tensor(backbone),            # COMO SE CONECTAN LOS nts ENTRE ELLOS (SECUENCIA)
                   connections=connections,                    # POSIBLES CONEXIONES CANONICAS
                   edge_index=torch.from_numpy(edge_index),    # CONEXIONES ENTRE LOS NODOS,  # CONEXIONES ENTRE LOS NODOS
                   y = torch.tensor(y.T, dtype=torch.long))    # ENLACES VALIDOS ENTRE nts
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    return pattern
#===============================================================



#===============================================================
def seq2embedding(sequence, nts=('A','U','C','G')):
    '''
    idx:
    sequence:
    window: tamaño de la ventana alrededor del nucleotido
    '''

    embedding = []

    for s in sequence:

        if s in nts:
            e = [0] * 4
            idx = nts.index(s)
            e[idx] = 1

        else:
            N = len(nts)
            e = [1/N] * N

        embedding.append(e)

    return embedding
#===============================================================



#===============================================================
def getPositionEncoding(seq_len, d, n=10000):
    '''
    seq_len: Longitud de la secuencia
    d: tamaño del embedding
    '''

    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P
#===============================================================



#===============================================================
def get_backbone(sequence):
    '''
    Esta función devuelve la secuencia de conexiones entre nucleótidos.
    '''

    b = [i for i in range(len(sequence))]
    backbone = np.array([b[:-1],b[1:]])
    backbone = np.concatenate((backbone, np.flip(backbone, axis=0)), axis=1)
    return backbone.tolist()
#===============================================================



#===============================================================
def get_connections(sequence, pairings=[('G','C'),('A','U'), ('G','U')], Dmin=4):

    # nts=('A','U','C','G')
    nts = {'A':1, 'U':3, 'C':5, 'G':7}

    seq = np.array([nts[s] for s in sequence]).reshape(-1,1)  # ---> Transformo la secuencia de nts en secuencia de números primos
    pairs = [int(nts[pair[0]]*nts[pair[1]]) for pair in pairings]  # ---> Transformo los pares en productos de números primos

    M = seq @ seq.T  # Matriz con todas las posibles conexiones

    connections = [[],[]]
    strength = []
    # Filtro secuencias de interés
    for pair in pairs:

        # links = np.triu(M) == pair  # Sólo considero la matriz triangular inferior
        links = M == pair  # Sólo considero la matriz triangular inferior

        source,target = np.where(links)

        idx = np.abs(target-source) >= Dmin  # Selecciono enlaces que están separados al menos "Dmin"

        connections[0].extend(target[idx].tolist())
        connections[1].extend(source[idx].tolist())

        if pair == 35: # GC
            strength.extend([3] * len(target[idx]))
        elif pair == 21: # GU
            strength.extend([2] * len(target[idx]))
        elif pair == 3: # AU
            strength.extend([1] * len(target[idx]))
        else:
            strength.extend([0] * len(target[idx]))

    return connections, strength
#===============================================================

#####################################################################################

#===============================================================
def pair_strength(pair):
    if "A" in pair and "U" in pair:
        return 2.
    if "G" in pair and "C" in pair:
        return 3.
    if "G" in pair and "U" in pair:
        return 0.8
    else:
        return 0
#===============================================================


#===============================================================
def ufold_prob_mat(seq):
    """Receive sequence and compute local conection probabilities (Ufold paper)"""

    Kadd = 30  # Ventana que se analiza para el score
    window = 3
    N = len(seq)

    mat = np.zeros((N, N), dtype=np.float32)  # En pytorch el default es float32

    L = np.arange(N)
    pairs = np.array(np.meshgrid(L, L)).T.reshape(-1,2)
    pairs = pairs[np.abs(pairs[:,0] - pairs[:,1]) > window,:]  # Pares posibles (a más de 3 nts)


    for i,j in pairs:

        coefficient = 0

        ########################################################################
        for add in range(Kadd):

            ##-------------------------------------------
            if (i - add >= 0) and (j + add < N):
                score = pair_strength((seq[i - add], seq[j + add]))

                if score == 0:
                    break
                else:
                    coefficient += score * np.exp(-0.5*(add**2))
            else:
                break
        ########################################################################


        ########################################################################
        if coefficient > 0:

            for add in range(1, Kadd):

                if (i + add < N) and (j - add >= 0):
                    score = pair_strength((seq[i + add], seq[j - add]))

                    if score == 0:
                        break
                    else:
                        coefficient += score * np.exp(-0.5*(add**2))
                else:
                    break
        ########################################################################

        mat[i,j] = coefficient

    return mat
#===============================================================
