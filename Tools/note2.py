# -*- coding: utf-8 -*-
"""
Proyecto: Simulación de Eventos Discretos - Problema de Reparación
Autor: [Tu Nombre]
Fecha: [Fecha]

Descripción:
Este notebook implementa la simulación del problema de reparación descrito en la Sección 7.7 
del libro "Simulation, Fifth Edition" de Sheldon M. Ross. El objetivo es simular un sistema 
de máquinas con repuestos y analizar el tiempo hasta que el sistema falla (crash time).
"""

# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon
from tqdm import tqdm

# Configuración inicial
np.random.seed(42)  # Semilla para reproducibilidad

# Parámetros iniciales del sistema
n = 4  # Máquinas en uso
s = 3  # Máquinas de repuesto
lambda_rate = 1  # Tasa de fallo de las máquinas (distribución exponencial)
mu_rate = 2      # Tasa de reparación (distribución exponencial)

# Funciones auxiliares
def generate_failure_times(n, lambda_rate):
    """Genera tiempos de fallo para n máquinas."""
    return np.sort(expon.rvs(scale=1/lambda_rate, size=n))

def simulate_repair_system(n, s, lambda_rate, mu_rate):
    """
    Simula el sistema de reparación hasta que el sistema falla.
    
    Parámetros:
    - n: Número de máquinas en uso
    - s: Número de máquinas de repuesto
    - lambda_rate: Tasa de fallo de las máquinas
    - mu_rate: Tasa de reparación
    
    Retorna:
    - crash_time: Tiempo en el que el sistema falla
    """
    # Inicialización
    t = 0  # Tiempo actual
    r = 0  # Máquinas dañadas
    t_star = np.inf  # Tiempo de finalización de la próxima reparación
    failure_times = generate_failure_times(n, lambda_rate)  # Tiempos de fallo iniciales
    repair_queue = []  # Cola de máquinas esperando reparación
    
    while True:
        # Determinar el próximo evento
        next_failure = failure_times[0] if len(failure_times) > 0 else np.inf
        next_event_time = min(next_failure, t_star)
        
        # Actualizar el tiempo
        t = next_event_time
        
        if next_event_time == next_failure:
            # Evento: Una máquina falla
            r += 1
            failure_times = failure_times[1:]  # Remover el tiempo de fallo procesado
            
            if r > s:
                # El sistema falla (crash)
                return t
            
            # Agregar la máquina a la cola de reparación
            repair_queue.append(t)
            
            # Generar un nuevo tiempo de fallo si hay máquinas funcionando
            if len(failure_times) < n + s - r:
                new_failure_time = t + expon.rvs(scale=1/lambda_rate)
                failure_times = np.append(failure_times, new_failure_time)
                failure_times.sort()
        
        elif next_event_time == t_star:
            # Evento: Una máquina se repara
            r -= 1
            repair_queue.pop(0)  # Remover la máquina reparada
            
            if len(repair_queue) > 0:
                # Programar la próxima reparación
                repair_time = expon.rvs(scale=1/mu_rate)
                t_star = t + repair_time
            else:
                t_star = np.inf

# Simulación múltiple
num_simulations = 1000
crash_times = []

for _ in tqdm(range(num_simulations), desc="Simulando"):
    crash_time = simulate_repair_system(n, s, lambda_rate, mu_rate)
    crash_times.append(crash_time)

# Convertir los resultados a un DataFrame
results = pd.DataFrame(crash_times, columns=["Crash Time"])

# Análisis estadístico
mean_crash_time = results["Crash Time"].mean()
std_crash_time = results["Crash Time"].std()

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.hist(results["Crash Time"], bins=30, color="skyblue", edgecolor="black", alpha=0.7)
plt.axvline(mean_crash_time, color="red", linestyle="--", label=f"Media: {mean_crash_time:.2f}")
plt.title("Distribución del Tiempo de Falla del Sistema")
plt.xlabel("Tiempo de Falla")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.show()

# Resultados finales
print(f"Tiempo promedio hasta el fallo del sistema: {mean_crash_time:.2f}")
print(f"Desviación estándar del tiempo de fallo: {std_crash_time:.2f}")
