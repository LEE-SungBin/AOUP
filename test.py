import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Inital Conditions
y0 = [-1,  # x1
      0,  # y1
      1,  # x2
      0,  # y2
      0,  # x3
      0,  # y3
      0.306893,  # vx1
      0.125507,  # vy1
      0.306893,  # vx2
      0.125507,  # vy2
      -2*0.306893,  # vx3
      -2*0.125507]  # vy3

# Time steps and the total number of steps
N = 10000
T = 0.001

# Definition of the Function


def ThreeBody(t, y):
    f = np.zeros(12)

   # The velocities of the three bodies
    f[0] = y[6]
    f[1] = y[7]
    f[2] = y[8]
    f[3] = y[9]
    f[4] = y[10]
    f[5] = y[11]

   # The x and y positions of each object respectively
    f[6] = -(y[0]-y[2])/(((y[0]-y[2])**2+(y[1]-y[3])**2)**(3/2)) \
        - (y[0]-y[4])/(((y[0]-y[4])**2+(y[1]-y[5])**2)**(3/2))

    f[7] = -(y[1]-y[3])/(((y[0]-y[2])**2+(y[1]-y[3])**2)**(3/2)) \
        - (y[1]-y[5])/(((y[0]-y[4])**2+(y[1]-y[5])**2)**(3/2))

    f[8] = -(y[2]-y[0])/(((y[2]-y[0])**2+(y[3]-y[1])**2)**(3/2)) \
        - (y[2]-y[4])/(((y[2]-y[4])**2+(y[3]-y[5])**2)**(3/2))

    f[9] = -(y[3]-y[1])/(((y[2]-y[0])**2+(y[3]-y[1])**2)**(3/2)) \
        - (y[3]-y[5])/(((y[2]-y[4])**2+(y[3]-y[5])**2)**(3/2))

    f[10] = -(y[4]-y[0])/(((y[4]-y[0])**2+(y[5]-y[1])**2)**(3/2)) \
        - (y[4]-y[2])/(((y[4]-y[2])**2+(y[5]-y[3])**2)**(3/2))

    f[11] = -(y[5]-y[1])/(((y[4]-y[0])**2+(y[5]-y[1])**2)**(3/2)) \
        - (y[5]-y[3])/(((y[4]-y[2])**2+(y[5]-y[3])**2)**(3/2))

    return f


# Solving for the positions of all bodies
t = np.linspace(0, N*T, N)
solution = solve_ivp(ThreeBody, [0, 800], y0, t_eval=t, rtol=1e-12)

fig = plt.figure()
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-0.5, 0.5))
lines = [ax.plot([], [], lw=2) for _ in range(3)]


def animate(n):
    lines[0][0].set_data(solution.y[0][:n], solution.y[1][:n])
    lines[1][0].set_data(solution.y[2][:n], solution.y[3][:n])
    lines[2][0].set_data(solution.y[4][:n], solution.y[5][:n])
    return lines[0][0], lines[1][0], lines[2][0],


anim = FuncAnimation(fig, animate, frames=1000, interval=10, blit=True)
plt.show()
