import numpy as np
import matplotlib.pyplot as plt

class Carrito:
    def __init__(self, position, angle):
        self.position = np.array(position, dtype='float64')  # Posición [x, y]
        self.angle = angle  # Orientación en radianes
        
    def update(self, velocity, delta_angle, dt):
        """
        Actualiza la posición y orientación del carrito.
        - velocity: Velocidad lineal
        - delta_angle: Cambio en la dirección (radianes)
        - dt: Paso de tiempo
        """
        self.angle += delta_angle * dt
        self.position += np.array([np.cos(self.angle), np.sin(self.angle)]) * velocity * dt


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


# Parámetros iniciales
initial_position = [0, 0]
initial_angle = 0
target_position = [15, 15]  # Punto objetivo

carrito = Carrito(initial_position, initial_angle)
pid = PIDController(kp=0.004, ki=0.0, kd=0.07)

dt = 0.1  # Paso de tiempo
steps = 200  # Número de pasos de simulación

# Para la visualización
x_traj, y_traj = [], []

for _ in range(steps):
    # Calculamos el error como la distancia al punto objetivo
    error = np.linalg.norm(target_position - carrito.position)
    delta_angle = pid.control(error, dt)
    
    carrito.update(velocity=1, delta_angle=delta_angle, dt=dt)  # Asignamos una velocidad constante
    
    x_traj.append(carrito.position[0])
    y_traj.append(carrito.position[1])

# Dibujamos la trayectoria del carrito
plt.plot(x_traj, y_traj, label='Trayectoria del carrito')
plt.scatter(*target_position, color='red', label='Punto objetivo')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.axis('equal')
plt.show()

