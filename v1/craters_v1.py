# 2 discs
# force accumulée 
# voisins : chaque cell de taille r
# > Comment détecter la perte de voisinage ?
from packages_crater import import_params, from_ij_to_n, from_n_to_ij, from_pos_get_cell

params = import_params('params.txt')
rho = params['rho']
E = params['E']
nu = params['nu']
g = params['g']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow

fig,ax=plt.subplots()

r=1e-3
d=2*r
d_cell=d
dt=1e-3
m=rho*(4/3)*np.pi*(r**3) # 1e-5 kg  

alpha=1
k=(4/3)*(E/2*(1-nu**2))*(r/2)**(2-alpha)

def F(d,voisins,Data):
    '''
        Force à modéliser
    '''
    return np.zeros(2)

def Voisins(p,Cells):
    '''
        A une particule p, renvoie la liste des ids des particules voisines.
    '''
    n_cell=p[1]
    i,j=from_n_to_ij(n_cell,Mx)
    voisins=[]
    for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
        if 0 <= i+di < Mx and 0 <= j+dj < My:
            
            voisins+=Cells[i+di][j+dj]

    return voisins

################################################# INITIALISATION #############################################
                                #Crée une liste de particules et de cellules
duration=1e3*dt
t=0
eq=False
Nx,Ny=1,2 #Nombre de particules selon x,y
N=Nx*Ny #Nombre de particules total
Mx,My=2*Nx,2*Ny #Nombre de cellules selon x,y

ax.autoscale(False)

ax.set_xticks(np.arange(0, 2*My*d_cell, d_cell))  # Grille horizontale tous les d_cell
ax.set_yticks(np.arange(0, 2*Mx*d_cell, d_cell))  # Grille verticale tous les d_cell

ax.set_xlim(0, Mx*d_cell)
ax.set_ylim(0, My*d_cell)
ax.set_aspect('equal')
ax.grid(True)
ar_width,hd_width,hd_length=0.05*r,0.1*r,0.05*r

circles = []
arrows=[]

Particles = np.zeros((N, 2),dtype=np.int32) #[[id, [cell_nb_x, cell_nb_y]],...] id de la particule et coords de la cellule à laquelle elle appartient.
Data = np.zeros((N, 6), dtype=np.float64) #[[x,y,v_x,v_y,Px,Py],...]
Cells=[[[] for j in range(My)] for i in range(Mx)] # [id1,id2,...] id des particules présentes dans une cellule
Texts=[[ax.annotate(f"{from_ij_to_n(i,j,Mx)} : {Cells[i][j]}", xy=(i*d_cell,j*d_cell)) for j in range(My)] for i in range(Mx)]
# attention, le premier indice correspond à la colonne, le 2 a la ligne

for i in range(Nx):
    for j in range(Ny):
        
        n=i+Nx*j #Attention
        x0,y0 = 1*d_cell*(i+0.5)+d_cell*0.1*i,1.1*d_cell*(j+0.5+0.1) # On place un disque légèrement au dessus d'un autre
            #x,y,vx,vy,P
        Data[n,0] = x0
        Data[n,1] = y0
        circles.append(plt.Circle((x0,y0), radius=r, ec='blue',fill=False))
        Data[n,2] = 0 #vx
        Data[n,3] = 0 #vy
        Data[n,4] = 0 #Force accumulée "P"
            #id, cell_nb
        Particles[n,0] = n
        Particles[n,1] = from_pos_get_cell(np.array([x0,y0]),d_cell,Mx) #numéro de la cellule (un array np ne peut avoir comme élement une liste)
            #MàJ de la cell concernée
        i_cell,j_cell=from_n_to_ij(Particles[n,1],Mx)
        Cells[i_cell][j_cell].append(n)

        ax.add_patch(circles[-1])
        arrow = FancyArrow(
        circles[-1].center[0], circles[-1].center[1],  # Position de la flèche (même centre que le cercle)
        0,  # Direction x aléatoire
        0,  # Direction y aléatoire
        width=ar_width, head_width=hd_width, head_length=hd_length, fc='red', ec='red'
    )
        ax.add_patch(arrow)
        arrows.append(arrow)

def update(frame):
    for _ in range(skip_frames):
        for n in range(N):
            p=Particles[n]
            d=Data[n]
            id=p[0]

            voisins=Voisins(p,Cells)

            f=F(d,voisins,Data)
            Data[n][4:6]+=f

            f_tot=f+np.array([0,-m*g])

            # Apply force

            Data[n][2:4] += f_tot*dt/m
            circles[n].center += Data[n][2:4] * dt
            if circles[n].center[1] <= r:
                circles[n].center = (circles[n].center[0], r)
                Data[n][3]=0
                
            # Mettre à jour les positions

            new_cell=from_pos_get_cell(circles[n].center,d_cell,Mx)

            cell=p[1]
            if new_cell != cell:
                i_cell,j_cell=from_n_to_ij(cell,Mx)
                new_i_cell,new_j_cell=from_n_to_ij(new_cell,Mx)
                Cells[i_cell][j_cell].remove(id) # a optimiser
                Particles[n][1]=new_cell
                Cells[new_i_cell][new_j_cell].append(id)

            dx,dy=r*f/np.linalg.norm(m*g)
            
            arrow = arrows[n]
            arrow.remove()  # Supprimer l'ancienne flèche

            arrows[n] = FancyArrow(
                circles[n].center[0], circles[n].center[1], dx, dy, width=ar_width, head_width=hd_width, head_length=hd_length, fc='red', ec='red'
            )
            ax.add_patch(arrows[n])  # Ajouter la nouvelle flèche
    for i in range(Mx):
        for j in range(My):
            Texts[i][j].set_text(f"{from_ij_to_n(i,j,Mx)} : {Cells[i][j]}")
    return circles + arrows + [Texts[i][j] for i in range(Mx) for j in range(My)]

skip_frames=1

ani = FuncAnimation(fig, update, frames=30,blit=True,interval=1000)

ani.save('animation.mp4', writer='ffmpeg', fps=10)