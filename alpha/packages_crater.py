import numpy as np

def import_params(filename):
    parameters = {}
    with open(filename, 'r') as file:
        for line in file:
            if "=" in line:
                key, value = line.split("=")
                parameters[key.strip()] = float(value.split("#")[0].strip())
    return parameters

def from_ij_to_n(i,j,Mx): #Comment numéroter les cellules à partir des i,j
    return i+j*Mx

def from_n_to_ij(n,Mx): #from a single index n, get i,j
    i = n % Mx        # Indice de la colonne (x)
    j = n // Mx         # Indice de la ligne (y)

    return i, j

def from_pos_get_cell(pos,d_cell,Mx): # a partir de x0, y0, obtenir le numéro de cellule
    # a=(pos)//d_cell
    i,j=int(pos[0]//d_cell),int(pos[1]//d_cell)

    # i,j=a.astype(np.int16)
    return from_ij_to_n(i,j,Mx)

Mx=1012
i,j=from_n_to_ij(102,Mx)
print(from_ij_to_n(i,j,Mx))


def F(d,voisins,Data,dt,k):
    An=np.zeros(2) #shape = 2? 1er terme pour composante normale
    As=np.zeros(2) # 1er terme pour composante tangentielle
    Bn=np.zeros(2) #shape = 2? 2e terme pour composante normale
    Bs=np.zeros(2) # 2e terme pour composante tangentielle
    for id_voisin in voisins:
        d_voisin=Data[id_voisin]
        # Relative velocity of the neighbour
        v_rel = d_voisin[2:4]- d[2:4]
        # Vecteur normal
        n = d[0:2] - d_voisin[0:2] # normale entrante
        # Calcul du vecteur unitaire radial
        n_hat = n / np.linalg.norm(n) 

            ## Radial component ##

        # Composante radiale de la vitesse relative
        v_radial = np.dot(v_rel, n_hat)*n_hat # produit scalaire : vitesse entrante

            ## Tangential component ##

        # Vitesse tangentielle (vecteur)
        v_tang = v_rel - v_radial

        delta_n = v_radial * dt # positif si le voisin "entre"
        An-=k*delta_n

    return np.zeros(2)
    return As+An+Bs+Bn