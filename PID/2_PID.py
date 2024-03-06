import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
    
    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class CarritoAvanzado:
    def __init__(self, position, angle, mass=1.0):
        self.position = np.array(position, dtype='float64')
        self.angle = angle
        self.velocity = np.array([0.0, 0.0])
        self.mass = mass
    
    def apply_force(self, force, delta_angle, dt):
        acceleration = force / self.mass
        self.angle += delta_angle * dt
        self.velocity += np.array([np.cos(self.angle), np.sin(self.angle)]) * acceleration * dt
        self.position += self.velocity * dt
    
    def update_friction(self, friction_coefficient):
        friction_force = -self.velocity * friction_coefficient
        self.velocity += friction_force

def ha_llegado(carrito, target_position, threshold=0.5):
    distancia = np.linalg.norm(carrito.position - target_position)
    return distancia < threshold

# Definición de variables globales para la animación
xdata, ydata = [], []
target_path = [np.array([7, 3])]  # Definir una trayectoria o punto objetivo
steps = 100  # Número de pasos de la simulación

carrito = CarritoAvanzado([0, 0], 0)
pid = PIDController(0.3, 0.06, 0.5)
dt = 0.1  # Intervalo de tiempo entre pasos

fig, ax = plt.subplots()
ln, = plt.plot([], [], 'r-', animated=True)
target_point = plt.scatter([], [], s=100, color='blue', marker='x')  # Punto de llegada

def init():
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    target_position = target_path[-1]  # Asume el último punto como objetivo
    target_point.set_offsets(target_position)  # Dibuja el punto objetivo
    return ln, target_point

def update(frame):
    global xdata, ydata
    if frame < len(target_path):
        target_position = target_path[frame]
    else:
        target_position = target_path[-1]

    if ha_llegado(carrito, target_position):
        ani.event_source.stop()
        print(f"El carrito ha llegado al objetivo en el frame {frame}.")
        return ln, target_point

    error = np.linalg.norm(target_position - carrito.position)
    delta_angle = pid.control(error, dt)
    carrito.apply_force(force=10, delta_angle=delta_angle, dt=dt)
    carrito.update_friction(friction_coefficient=0.1)

    xdata.append(carrito.position[0])
    ydata.append(carrito.position[1])
    ln.set_data(xdata, ydata)
    return ln, target_point

ani = FuncAnimation(fig, update, frames=range(steps),
                    init_func=init, blit=True)
plt.show()
