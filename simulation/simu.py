'''
Simulation of a granular material using a Kelvin-Voigt model
Generates two files: data.txt and data_force.txt which
contains the positions and the forces on the particles at each frame
'''

from packages_crater import (
    import_params,
    K,
    C,
)
import time
import numpy as np
import math

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
max_particles_per_cell = 2  # Maximum number of particles per cell : this
# is crucial for the performance of the simulation because it allows to use numpy arrays

# Initialisation of the global variables
Particles = None  # [[id, cell_nb],...]
Velocities = None  # [[v_x,v_y],...]
Positions = None  # [[x,y],...]
Forces=None
Cells = None  # {id1,id2,...} each cell contains a set of particles

Data = None  # Positions to save
Data_force = None # Force to save
# Data_velocities = None # Velocities to save


################################################# FUNCTIONS #############################################
def from_ij_to_n(i, j, Mx): 
    '''
    from i,j get the single index n
    '''
    return i + j * Mx


def from_n_to_ij(n, Mx):
    '''
    from a single index n get the i,j
    '''
    i = n % Mx  # Indice de la colonne (x)
    j = n // Mx  # Indice de la ligne (y)

    return i, j


def F(id, Positions, Velocities, Cells):
    """
    Compute the force acting on the particle id (using Kelvin-Voigt model)
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
    Get the neighbours of a particle p using the cells divisions method.
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
    ]: # They are 24 neighbours cells to consider
        if 0 <= i + di < Mx and 0 <= j + dj < My:
            for k in range(len(Cells[i + di, j + dj])):
                if Cells[i + di, j + dj, k] not in [-1, id]:
                    voisins.append(Cells[i + di, j + dj][k])
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

    new_cell = get_cell(Positions[n]) # Get the old cell
    old_cell = Particles[n][1] # Get the new cell

    if new_cell != old_cell: # If the cell has changed
        i_cell, j_cell = from_n_to_ij(old_cell, Mx)  
        new_i_cell, new_j_cell = from_n_to_ij(new_cell, Mx)  
        remove_from_cell(n, i_cell, j_cell, Cells)
        Particles[n][1] = new_cell  
        try: # Update the cell of the particle
            add_to_cell(n, new_i_cell, new_j_cell, Cells)
        except IndexError as e:
            np.savetxt("positions_error.txt", Positions) # Save the positions if there is an error
            print(e)
            exit(1)


################################################# INITIALIZATION #############################################


def generate_positions(new_config):
    """
    Generate N random positions for the particles 
    and ensure that they are not overlapping
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
    return positions_i


def remove_from_cell(id, i, j, cells):
    '''
    remove the particle id from the cell i,j
    '''
    cells[i, j][cells[i, j] == id] = -1

def add_to_cell(id, i, j, cells):
    '''
    Add the particle id to the cell i,j
    '''
    cells[i, j][np.where(cells[i, j] == -1)[0][0]] = id


def init(new_config):
    '''
    Initialize the initial conditions of the simulation
    with random positions.
    Small vertical velocity is given to the particles so
    that the first frames of the simulation are not useless.
    '''
    global Positions, Velocities, Particles, Cells, Data, Forces, Data_force
    Particles = np.zeros((N, 2), dtype=np.int32)  # [[id, cell_nb],...]
    Velocities = np.zeros((N, 2), dtype=np.float64)  # [[v_x,v_y],...]
    Positions = np.zeros((N, 2), dtype=np.float64)  # [[x,y],...]
    Forces = np.zeros(N, dtype=np.float64) # Norm of the force acting on each particle (wiwthout weight)
    Cells = (
        np.zeros((Mx, My, max_particles_per_cell), dtype=np.int32) - 1
    )  # Each cell contains the id of the particles in it

    Data = []  # Positions data to save
    Data_force = [] # Force data to save

    positions_i = generate_positions(new_config)

    print(f"k={k:.1e}N.m-1 c={c:.1e}N.m-1.s m={m:.1e}kg")  # 3e7
    print(f"dt_crit:{dt_crit:1e}s, dt={dt:.1e}s")

    for n in range(N):
        i, j = from_n_to_ij(n, Nx)
        Positions[n, 0:2] = positions_i[n]
        Velocities[n, 0:2] = 0, -1e-2
        Particles[n] = [n, get_cell(Positions[n])]
        i, j = from_n_to_ij(Particles[n][1], Mx)
        add_to_cell(n, i, j, Cells)


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
    Forces[n] = np.linalg.norm(f-Poids)
    Velocities[n] += (1 / m) * f * dt


def update_particle(n):
    '''
    Update the particle n
    '''
    update_position(n)
    update_velocity(n)
    check_boundaries(n)
    update_cell(n)


def update(frame):
    '''
    Update all the particles
    '''
    for n in range(N):
        update_particle(n)


################################################# MAIN #############################################
def main():
    '''
    Launch the simulation
    '''
    for frame in range(frames):
        update(frame)
        if frame % skip_frames == 0:
            print(f"{frame}/{frames}")
            Data.append(Positions.copy())
            Data_force.append(Forces.copy())


def benchmark(new_config=True):
    '''
    Initialisation and launch of the simulation
    '''
    init(new_config)
    print("init done")

    start = time.perf_counter()
    main()
    end = time.perf_counter()
    elapsed_time = end - start
    return elapsed_time


if __name__ == "__main__":
    new_config = True # True si initialiser des nouvelles conditions initiales
    t = benchmark(new_config)


    print(
        f"Without multiprocessing: {t:.1e} or {t/(frames):.1e} per frame for N={Nx*Ny}"
    )
    np.savetxt("./data.txt", np.reshape(Data, (len(Data), -1)))
    np.savetxt("./data_force.txt", np.reshape(Data_force, (-1,len(Data_force))))

    exit(1)
