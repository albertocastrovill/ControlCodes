import numpy as np

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x0):
        self.A = A  # Matriz de transición de estado
        self.B = B  # Matriz de control de entrada
        self.H = H  # Matriz de medición
        self.Q = Q  # Covarianza del ruido del proceso
        self.R = R  # Covarianza del ruido de medición
        self.P = P  # Estimación de la covarianza del error
        self.x = x0  # Estado inicial
        
    def predict(self, u=0):
        # Predicción del estado siguiente
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x
    
    def update(self, z):
        # Actualización basada en la medición
        K = np.dot(self.P, self.H.T) / (np.dot(np.dot(self.H, self.P), self.H.T) + self.R)
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

# Definición de parámetros del modelo
dt = 0.1  # Intervalo de tiempo
A = np.array([[1, dt], [0, 1]])  # Matriz de transición de estado
B = np.array([[0.5 * dt**2], [dt]])  # Matriz de control de entrada
H = np.array([[1, 0]])  # Matriz de medición
Q = np.array([[1, 0], [0, 1]])  # Covarianza del ruido del proceso
R = np.array([[1]])  # Covarianza del ruido de medición
P = np.array([[1, 0], [0, 1]])  # Estimación inicial de la covarianza del error
x0 = np.array([[0], [0]])  # Estado inicial

kf = KalmanFilter(A, B, H, Q, R, P, x0)

# Simulación
n_steps = 50
true_velocity = 1  # Velocidad constante
true_positions = [x0[0, 0]]
measured_positions = [x0[0, 0] + np.random.normal(0, np.sqrt(R[0,0]))]
estimated_positions = [x0[0, 0]]

for _ in range(n_steps):
    # Actualización del estado real
    true_position = true_positions[-1] + true_velocity * dt
    true_positions.append(true_position)
    
    # Generación de la medición ruidosa
    measured_position = true_position + np.random.normal(0, np.sqrt(R[0,0]))
    measured_positions.append(measured_position)
    
    # Predicción y actualización del Filtro de Kalman
    kf.predict(u=np.array([[0], [true_velocity]]))  # asumimos aceleración = 0
    kf.update(np.array([[measured_position]]))
    
    estimated_positions.append(kf.x[0, 0])

# Visualización
import matplotlib.pyplot as plt

plt.plot(true_positions, label='Posición Real')
plt.plot(measured_positions, 'x', label='Mediciones')
plt.plot(estimated_positions, label='Estimación Kalman')
plt.legend()
plt.xlabel('Paso')
plt.ylabel('Posición')
plt.show()

# Ejemplo de aplicación del filtro de Kalman para estimar la posición de un objeto en movimiento.
