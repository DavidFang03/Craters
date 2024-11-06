from packages_crater import (
    import_params,
    from_ij_to_n,
    from_n_to_ij,
    K,
    C,
)
from show_stable import show
import time
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor

################################################# PARAMETERS #############################################
params = import_params("./params.txt")
rho = params["rho"]
E = params["E"]
nu = params["nu"]
g = params["g"]
r = params["r"]
dt = params["dt"]
alpha = params["alpha"]
Nx = int(params["Nx"])
Ny = int(params["Ny"])
frames = int(params["frames"])
skip_frames = int(params["skip_frames"])

# Particles charasteristics
d = 2 * r
m = rho * (4 / 3) * np.pi * (r**3)  # 1e-5 kg
k = K(E, nu, r, alpha)
c = C(k, m)
Poids = np.array([0, -m * g])

# Boundaries
wall_l = 0
wall_r = (2 * Nx + 3) * r

# Simulation parameters
N = Nx * Ny  # Total number of particles
dt_crit = 2 * np.sqrt(m / k)

# Cell parameters
d_cell = r
Mx, My = 4 * Nx, 4 * Ny  # Number of cells in x and y

# Initialisation of the global variables
Particles = np.zeros((N, 2), dtype=np.int32)  # [[id, cell_nb],...]
Velocities = np.zeros((N, 2), dtype=np.float64)  # [[v_x,v_y],...]
Positions = np.zeros((N, 2), dtype=np.float64)  # [[x,y],...]
Cells = [
    [set() for j in range(My)] for i in range(Mx)
]  # {id1,id2,...} each cell contains a set of particles

Data = []  # Data to save


################################################# FUNCTIONS #############################################
def F(id, Positions, Velocities, Cells):
    """
    Force (Kelvin-Voigt)
    id: id de la particule système
    Positions:toutes les positions
    Velocities : toutes les vitesses
    Cells : toutes les celluls
    """
    An = np.zeros(2)  # shape = 2? 1er terme pour composante normale
    At = np.zeros(2)  # 1er terme pour composante tangentielle
    Bn = np.zeros(2)  # shape = 2? 2e terme pour composante normale
    Bt = np.zeros(2)  # 2e terme pour composante tangentielle
    voisins = Voisins(Particles[id], Cells)

    for id_voisin in voisins:
        u_n = Positions[id_voisin] - Positions[id]  # vecteur normal sortant

        if np.sum(u_n**2) < (2 * r) ** 2:
            n_hat = u_n / np.linalg.norm(u_n)

            v_rel = Velocities[id][0:2] - Velocities[id_voisin][0:2]
            v_radial = (
                np.dot(v_rel, n_hat) * n_hat
            )  # produit scalaire : vitesse du système

            delta_n = 2 * r * n_hat - u_n
            An -= k * delta_n
            Bn -= c * v_radial

    return An + Bn + At + Bt + Poids


def Voisins(p, Cells):
    """
    A une particule p, renvoie la liste des ids des particules voisines.
    """
    n_cell = p[1]
    i, j = from_n_to_ij(n_cell, Mx)
    voisins = []
    for di, dj in [
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
        (-2, 1),
        (2, 1),
        (1, -2),
        (1, 2),
        (-2, -1),
        (2, -1),
        (-1, -2),
        (-1, 2),
        (-2, -2),
        (2, -2),
        (-2, 2),
        (2, 2),
    ]:
        if 0 <= i + di < Mx and 0 <= j + dj < My:

            voisins += Cells[i + di][j + dj]

    return voisins


def get_cell(pos):
    """
    From position, get the cell number
    """
    i, j = math.floor(pos[0] // d_cell), math.floor(pos[1] // d_cell)
    return from_ij_to_n(i, j, Mx)


def update_cell(n):
    """
    Given a particle n, update the cell
    """

    new_cell = get_cell(Positions[n])
    old_cell = Particles[n][1]

    if new_cell != old_cell:
        # print(new_cell)
        i_cell, j_cell = from_n_to_ij(old_cell, Mx)  # Get the old cell
        new_i_cell, new_j_cell = from_n_to_ij(new_cell, Mx)  # Get the new cell
        Cells[i_cell][j_cell].remove(n)  # Remove the particle from the old cell
        Particles[n][1] = new_cell  # Update the cell of the particle
        try:
            Cells[new_i_cell][new_j_cell].add(n)  # Add the particle to the new cell
        except IndexError as e:
            # np.savetxt("data.txt", np.reshape(Data, (len(Data), -1)))
            show()
            print("err but saved", e)
            exit(1)


################################################# INITIALIZATION #############################################


def generate_positions():
    """
    Generate N random positions for the particles
    """
    positions_i = []
    max_rd = 0
    max_rd_for_j = 0
    min_y = r

    y = 0
    for j in range(Ny):

        x = 0
        max_rd_for_j += max_rd
        max_rd = 0
        for i in range(Nx):
            pertubation_y = np.random.uniform(0, r / 2)
            pertubation_x = np.random.uniform(0, r / 2)
            y = (2 * j + 1) * r + max_rd_for_j + pertubation_y
            x += 2 * r + pertubation_x
            if y < min_y:
                min_y = y
            if pertubation_y > max_rd:
                max_rd = pertubation_y

            positions_i.append([x, y])

    positions_i = np.array(positions_i)
    # np.savetxt("positions.txt", positions_i)
    positions_i[:, 1] -= min_y - r
    # np.savetxt("positions_aftersub.txt", positions_i)

    return positions_i


def init():
    positions_i = generate_positions()

    print(f"k={k:.1e}N.m-1 c={c:.1e}N.m-1.s m={m:.1e}kg")  # 3e7
    print(f"dt_crit:{dt_crit:1e}s, dt={dt:.1e}s")

    for n in range(N):
        i, j = from_n_to_ij(n, Nx)
        Positions[n, 0:2] = positions_i[n]
        Velocities[n, 0:2] = 0, -1e-2
        Particles[n] = [n, get_cell(Positions[n])]
        i, j = from_n_to_ij(Particles[n][1], Mx)
        Cells[i][j].add(n)


################################################# MAIN LOOP #############################################
def check_boundaries(n):
    if Positions[n, 1] <= r:
        # print(Positions[n, 0])
        Positions[n] = (Positions[n, 0], r)
        Velocities[n, 1] = 0
    if Positions[n, 0] <= wall_l + r:
        Positions[n] = (wall_l + r, Positions[n, 1])
        Velocities[n, 0] = 0
    elif Positions[n, 0] >= wall_r - r:
        Positions[n] = (wall_r - r, Positions[n, 1])
        Velocities[n, 0] = 0


def update_position(n):
    Positions[n] += Velocities[n] * dt


def update_velocity(n):
    f = F(n, Positions, Velocities, Cells)
    Velocities[n] += (1 / m) * f * dt


def update_particle(n):
    update_position(n)
    update_velocity(n)
    check_boundaries(n)
    update_cell(n)


# def update(frame):
#     for n in range(N):
#         update_particle(n)


################################################# MAIN #############################################
def main():
    with ProcessPoolExecutor() as executor:
        for frame in range(frames):
            executor.map(update_particle, range(N))
            # update(frame)
            if frame % skip_frames == 0:
                print(f"{frame}/{frames}")
                # Data.append(Positions.copy())


if __name__ == "__main__":
    init()
    # exit(1)
    print("initialisation terminée")
    # np.savetxt("data_init.txt", Positions)
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    elapsed_time = end - start
    print(
        f"Durée écoulée : {elapsed_time:.1e} secondes soit {elapsed_time/(frames):.1e} par itération pour N={Nx*Ny}"
    )
    # np.savetxt("data_fin.txt", Positions)
    # np.savetxt("data.txt", np.reshape(Data, (len(Data), -1)))
    # show()
