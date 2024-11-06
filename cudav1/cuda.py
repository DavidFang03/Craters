import numpy as np
from numba import cuda

# Parameters
N = 10000  # Number of particles
F = 1000  # Number of frames
dt = 0.01  # Time step
r = 0.1  # Radius of particles

# Initialize positions and velocities
positions = np.random.rand(N, 2).astype(np.float32)
velocities = np.random.rand(N, 2).astype(np.float32)


@cuda.jit
def update_positions(positions, velocities, dt, r, wall_l, wall_r):
    idx = cuda.grid(1)
    if idx < positions.shape[0]:
        # Update positions based on velocities
        positions[idx, 0] += velocities[idx, 0] * dt
        positions[idx, 1] += velocities[idx, 1] * dt

        # Handle boundary conditions
        if positions[idx, 1] <= r:
            positions[idx, 1] = r
            velocities[idx, 1] = 0
        if positions[idx, 0] <= wall_l + r:
            positions[idx, 0] = wall_l + r
            velocities[idx, 0] = 0
        elif positions[idx, 0] >= wall_r - r:
            positions[idx, 0] = wall_r - r
            velocities[idx, 0] = 0


def main():
    # Copy data to GPU
    d_positions = cuda.to_device(positions)
    d_velocities = cuda.to_device(velocities)

    threads_per_block = 256
    blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

    wall_l = 0.0
    wall_r = 1.0

    for frame in range(F):
        update_positions[blocks_per_grid, threads_per_block](
            d_positions, d_velocities, dt, r, wall_l, wall_r
        )
        cuda.synchronize()  # Ensure all threads have completed

    # Copy data back to CPU
    positions_updated = d_positions.copy_to_host()

    print("Final positions:", positions_updated)


if __name__ == "__main__":
    main()
