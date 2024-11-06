import numpy as np


def import_params(filename):
    parameters = {}
    with open(filename, "r") as file:
        for line in file:
            if "=" in line:
                key, value = line.split("=")
                parameters[key.strip()] = float(value.split("#")[0].strip())
    return parameters


def from_ij_to_n(i, j, Mx):  # Comment numéroter les cellules à partir des i,j
    return i + j * Mx


def from_n_to_ij(n, Mx):  # from a single index n, get i,j
    i = n % Mx  # Indice de la colonne (x)
    j = n // Mx  # Indice de la ligne (y)

    return i, j


def K(E, nu, r, alpha):
    return (4 / 3) * (E / 2 * (1 - nu**2)) * (r / 2) ** (2 - alpha)


def C(k, m):
    return 2 * np.sqrt(k * m)
