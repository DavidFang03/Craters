'''
This script is used to show the results of the simulation. 
It can be used to show a snapshot of the simulation at a given instant or to show an animation of the simulation.
'''

from packages_crater import import_params
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.cm as cm
import matplotlib.colors as mcolors


def show(datafilename,forcename,snap=True,instant="i"):

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


    num_particles = Nx * Ny
    num_coordinates = 2

    Data = np.genfromtxt(datafilename)
    Data = Data.reshape(-1, num_particles, num_coordinates)
    Force = np.genfromtxt(forcename)
    Force = Data.reshape(-1, num_particles)
    print(np.shape(Force))
    print(len(Force[0]))
    fig,ax = plt.subplots()

    def style(ax):
        ax.autoscale(False)
        ax.set_xticks(np.arange(0, 2 * My * r, r))
        ax.set_yticks(np.arange(0, 2 * Mx * r, r)) 
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(0, Mx * r)
        ax.set_ylim(-r, My * r)
        ax.set_axis_off()

        wall_top =  3*Ny*r
        wall_right = (2*Nx+3)*r
        wall_left = 0
        ax.annotate(
        '', xy=(2*r+r, wall_top), xytext=(0+r, wall_top),
        arrowprops=dict(arrowstyle=f'|-|, widthA={0.3*r},widthB={0.3*r}', color='black')
    )

        ax.text(2*r, wall_top+0.5*r, f'{r*1000}mm', fontsize=6, ha='center', va='bottom')
        ax.plot([wall_left, wall_right], [0,0], color='black', linewidth=1)
        ax.plot([wall_right, wall_right], [wall_top, 0], color='black', linewidth=0.5)
        ax.plot([wall_left, wall_left], [wall_top, 0], color='black', linewidth=0.5)        

    style(ax)

    norm = mcolors.Normalize(vmin=np.min(Force), vmax=np.max(Force))
    cmap = cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Force (N)')

    if snap:
        if instant == 'i':
            Data = Data[0]
            Force = Force[0]
        elif instant =='f':
            Data = Data[-1]
            Force= Force[-1]
        else:
            Data=Data[instant]
            Force=Force[instant]


        circles = []
        for n in range(np.shape(Data)[0]):
            circle = plt.Circle(Data[n], r, ec='black',fill=cmap(norm(Force[n])))
            circles.append(circle)
            ax.add_patch(circle)
        fig.savefig(f"media/frame_{instant}.jpg")

    else:
        def anim(frame):
            print(Force[frame,0])
            for n in range(N):
                circles[n].center = Data[frame, n]
                circles[n].set_color(cmap(norm(Force[frame,n])))
            return circles
        circles = []

        for n in range(N):
            circle = plt.Circle(Data[0,n], r, ec='black', fill=cmap(norm(Force[0,n])))
            circles.append(circle)
            ax.add_patch(circle)

        showed_frames = np.shape(Data)[0]
        ani = FuncAnimation(fig, anim, frames=showed_frames, blit=True)

        ani.save("./media/animation.mp4", writer="ffmpeg", fps=30)
        print("Animation saved in media/frame.mp4")



if __name__ == "__main__":
    instant='f' # 'i' for initial, 'f' for final, or an integer for a specific instant
    snap=True # True for a snapshot, False for an animation
    show("data.txt","data_force.txt", snap=snap, instant=instant)
