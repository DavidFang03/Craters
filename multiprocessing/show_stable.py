from packages_crater import import_params
import numpy as np
import matplotlib.pyplot as plt
from IntPlots import IntPlot
from matplotlib.animation import FuncAnimation


def show():

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
    Mx, My = Nx * 4, Ny * 4

    N = Nx * Ny

    # Data_init = np.genfromtxt("data_init.txt")
    # Data_fin = np.genfromtxt("data_fin.txt")
    Data = np.genfromtxt("data.txt")
    # Data_init = Data[0]

    num_particles = Nx * Ny
    num_coordinates = 2

    # Restructurer les données en 3D : [frame][particle][coordinate]
    Data = Data.reshape(-1, num_particles, num_coordinates)

    print(np.shape(Data))

    # P = IntPlot(1, 2, titles=["Etat initial", "Etat final"])
    P2 = IntPlot()

    # fig,ax=plt.subplots()
    # for n in range(len(Data_init)):
    #     pos_i = Data_init[n]
    #     # pos_f = Data_fin[n]

    #     circle_i = plt.Circle(pos_i, r, ec="blue", fill=False)
    #     P.axs[0, 0].add_patch(circle_i)
    # circle_f = plt.Circle(pos_f, r, ec="blue", fill=False)
    # P.axs[0, 1].add_patch(circle_f)

    def style(ax):
        ax.autoscale(False)
        ax.set_xticks(np.arange(0, 2 * My * r, r))  # Grille horizontale tous les d_cell
        ax.set_yticks(np.arange(0, 2 * Mx * r, r))  # Grille verticale tous les d_cell
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(0, Mx * r)
        ax.set_ylim(0, My * r)
        P2.ax.set_axis_off()

    # style(P.axs[0, 0])
    # style(P.axs[0, 1])
    style(P2.ax)

    circles = []
    for n in range(len(Data[0])):
        circle = plt.Circle(Data[0, n], r, ec="blue", fill=False)
        circles.append(circle)
        P2.ax.add_patch(circle)

    def anim(frame):
        for n in range(N):
            circles[n].center = Data[frame, n]
        return circles

    showed_frames = np.shape(Data)[0]
    # plt.show()
    ani = FuncAnimation(P2.fig, anim, frames=showed_frames, blit=True)
    # plt.show()
    ani.save("./media/animation.mp4", writer="ffmpeg", fps=10)
    print("Animation saved in media/animation.mp4")


if __name__ == "__main__":
    show()
