from packages_crater import (
    import_params,
    K,
    C,
)
from show_stable import show
import time
import numpy as np
import math
import os

# from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numba import cuda

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
max_particles_per_cell = 2

# Initialisation of the global variables
Particles = None  # [[id, cell_nb],...]
Velocities = None  # [[v_x,v_y],...]
Positions = None  # [[x,y],...]
Cells = None  # {id1,id2,...} each cell contains a set of particles

Data = None  # Data to save


################################################# FUNCTIONS #############################################
@cuda.jit(device=True)
def from_ij_to_n(i, j, Mx):  # Comment numéroter les cellules à partir des i,j
    return i + j * Mx


@cuda.jit(device=True)
def from_n_to_ij(n, Mx):  # from a single index n, get i,j
    i = n % Mx  # Indice de la colonne (x)
    j = n // Mx  # Indice de la ligne (y)

    return i, j


@cuda.jit(device=True)
def F(n, particles, positions, velocities, cells, r, k, c, Poids, Mx, My):
    """
    Force (Kelvin-Voigt)
    id: id de la particule système
    Positions:toutes les positions
    Velocities : toutes les vitesses
    Cells : toutes les celluls
    """
    An = cuda.local.array(2, dtype=np.float64)
    At = cuda.local.array(2, dtype=np.float64)
    Bn = cuda.local.array(2, dtype=np.float64)
    Bt = cuda.local.array(2, dtype=np.float64)
    voisins = Voisins(particles[n], cells, Mx, My)

    for id_voisin in voisins:
        if id_voisin < 0:
            continue
        u_n = positions[id_voisin] - positions[n]  # vecteur normal sortant
        sum_u_n_squared = u_n[0] ** 2 + u_n[1] ** 2
        if sum_u_n_squared < (2 * r) ** 2:
            norm_u_n = math.sqrt(sum_u_n_squared)
            n_hat = u_n / norm_u_n

            v_rel = velocities[n, 0:2] - velocities[id_voisin, 0:2]
            v_radial = (
                np.dot(v_rel, n_hat) * n_hat
            )  # produit scalaire : vitesse du système

            delta_n = 2 * r * n_hat - u_n
            An -= k * delta_n
            Bn -= c * v_radial

    return An + Bn + At + Bt + Poids


@cuda.jit(device=True)
def Voisins(p, cells, Mx, My):
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

            voisins += cells[i + di][j + dj]

    return voisins


@cuda.jit(device=True)
def get_cell(pos, Mx, d_cell):
    """
    From position, get the cell number
    """
    i, j = math.floor(pos[0] // d_cell), math.floor(pos[1] // d_cell)
    return from_ij_to_n(i, j, Mx)


################################################# INITIALIZATION #############################################


def generate_positions(new_config):
    """
    Generate N random positions for the particles
    """
    if new_config == False:
        return np.genfromtxt("positions_i.txt")
    positions_i = []
    max_rd = 0
    max_rd_for_j = 0
    min_y = r

    y = 0
    j = 0
    n = 0
    while n < N:
        max_rd_for_j += max_rd
        max_rd = 0
        x = 0
        while x < wall_r - 3 * r:
            pertubation_y = np.random.uniform(0, r / 2)
            pertubation_x = np.random.uniform(0, r / 2)
            y = (2 * j + 1) * r + max_rd_for_j + pertubation_y
            x += 2 * r + pertubation_x
            if y < min_y:
                min_y = y
            if pertubation_y > max_rd:
                max_rd = pertubation_y

            positions_i.append([x, y])
            n += 1

        j += 1

    positions_i = np.array(positions_i)

    positions_i[:, 1] -= min_y - r
    np.savetxt("positions_i.txt", positions_i)
    print("init save done")
    # np.savetxt("positions_aftersub.txt", positions_i)

    return positions_i


def init(new_config):
    global Positions, Velocities, Particles, Cells, Data
    Particles = np.zeros((N, 2), dtype=np.int32)  # [[id, cell_nb],...]
    Velocities = np.zeros((N, 2), dtype=np.float64)  # [[v_x,v_y],...]
    Positions = np.zeros((N, 2), dtype=np.float64)  # [[x,y],...]
    Cells = [
        [[] for j in range(My)] for i in range(Mx)
    ]  # {id1,id2,...} each cell contains a set of particles

    Data = []  # Data to save

    positions_i = generate_positions(new_config)

    print(f"k={k:.1e}N.m-1 c={c:.1e}N.m-1.s m={m:.1e}kg")  # 3e7
    print(f"dt_crit:{dt_crit:1e}s, dt={dt:.1e}s")

    for n in range(N):
        Positions[n, 0:2] = positions_i[n]
        Velocities[n, 0:2] = 0, -1e-2
        Particles[n] = [n, -1]


################################################# MAIN LOOP #############################################
@cuda.jit(device=True)
def update_cell(n, particles, positions, cells, Mx, d_cell):
    """
    Given a particle n, update the cell
    """

    new_cell = get_cell(positions[n], Mx, d_cell)
    old_cell = particles[n][1]

    if old_cell == -1:
        new_i_cell, new_j_cell = from_n_to_ij(new_cell, Mx)
        cells[new_i_cell][new_j_cell].append(n)
        particles[n][1] = new_cell
    elif new_cell != old_cell:
        i_cell, j_cell = from_n_to_ij(old_cell, Mx)  # Get the old cell
        new_i_cell, new_j_cell = from_n_to_ij(new_cell, Mx)  # Get the new cell
        cells[i_cell][j_cell].remove(n)  # Remove the particle from the old cell
        particles[n][1] = new_cell  # Update the cell of the particle
        cells[new_i_cell][new_j_cell].append(n)


@cuda.jit(device=True)
def check_boundaries(n, positions, velocities, r, wall_l, wall_r):
    if positions[n, 1] <= r:
        positions[n] = positions[n, 0], r
        velocities[n, 1] = 0
    if positions[n, 0] <= wall_l + r:
        positions[n] = wall_l + r, positions[n, 1]
        velocities[n, 0] = 0
    elif positions[n, 0] >= wall_r - r:
        positions[n] = wall_r - r, positions[n, 1]
        velocities[n, 0] = 0


@cuda.jit(device=True)
def update_position(n, positions, velocities, dt):
    positions[n] += velocities[n] * dt


@cuda.jit(device=True)
def update_velocity(
    n, particles, positions, velocities, cells, m, dt, r, k, c, Poids, Mx, My
):
    f = F(n, particles, positions, velocities, cells, r, k, c, Poids, Mx, My)
    velocities[n] += (1 / m) * f * dt


@cuda.jit
def update_particles(
    particles,
    positions,
    velocities,
    cells,
    dt,
    r,
    wall_l,
    wall_r,
    m,
    k,
    c,
    Poids,
    Mx,
    My,
):
    idx = cuda.grid(1)
    if idx < positions.shape[0]:
        # Update positions based on velocities
        update_cell(idx, particles, positions, cells, Mx, d_cell)
        update_velocity(
            idx, particles, positions, velocities, cells, m, dt, r, k, c, Poids, Mx, My
        )
        update_position(idx, positions, velocities, dt)
        check_boundaries(idx, positions, velocities, r, wall_l, wall_r)


################################################# MAIN #############################################
def main():
    # Copy data to GPU
    d_positions = cuda.to_device(Positions)
    d_particles = cuda.to_device(Particles)
    d_velocities = cuda.to_device(Velocities)
    d_cells = cuda.to_device(Cells)

    threads_per_block = 128
    blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

    for frame in range(frames):
        update_particles[blocks_per_grid, threads_per_block](
            d_particles,
            d_positions,
            d_velocities,
            d_cells,
            dt,
            r,
            wall_l,
            wall_r,
            m,
            k,
            c,
            Poids,
            Mx,
            My,
        )
        cuda.synchronize()  # Ensure all threads have completed

    # Copy data back to CPU
    positions_updated = d_positions.copy_to_host()

    return positions_updated


if __name__ == "__main__":

    init(False)
    start = time.perf_counter()
    final_data = main()
    end = time.perf_counter()
    elapsed_time = end - start
    print(
        f"Durée écoulée : {elapsed_time:.1e} secondes soit {elapsed_time/(frames):.1e} par itération pour N={Nx*Ny}"
    )
    np.savetxt(f"positions_cuda.txt", final_data)
