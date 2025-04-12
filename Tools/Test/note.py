# %% [markdown]
"""
# Simulación de Sistema de Reparación de Máquinas

Este notebook implementa una simulación de eventos discretos para el problema 7.7: 
"Un problema de reparación" donde tenemos n máquinas idénticas trabajando en paralelo 
y una única instalación de reparación.
"""

# %% [markdown]
"""
## 1. Introducción

El sistema consiste en:
- n máquinas idénticas funcionando en paralelo
- 1 instalación de reparación
- Las máquinas fallan según distribución exponencial con tasa λ
- El tiempo de reparación sigue una distribución G
- Objetivo: Estimar el tiempo hasta que r máquinas estén simultáneamente inactivas
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon, uniform, norm
import pandas as pd
from tqdm import tqdm

# Configuración estética
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]

# %% [markdown]
"""
## 2. Configuración Inicial y Parámetros

Definimos los parámetros iniciales del sistema y las distribuciones de probabilidad.
"""

# %%
class SimulationParameters:
    def __init__(self, n_machines=10, failure_rate=0.1, r_threshold=3, 
                 repair_dist='uniform', repair_params=(1, 3), random_seed=None):
        """
        Parámetros de la simulación
        
        Args:
            n_machines: Número total de máquinas en el sistema
            failure_rate: Tasa λ de fallo (exponencial)
            r_threshold: Número r de máquinas inactivas para detener simulación
            repair_dist: Distribución de tiempos de reparación ('uniform', 'expon', 'normal')
            repair_params: Parámetros para la distribución de reparación
            random_seed: Semilla para reproducibilidad
        """
        self.n_machines = n_machines
        self.failure_rate = failure_rate
        self.r_threshold = r_threshold
        self.repair_dist = repair_dist
        self.repair_params = repair_params
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def generate_repair_time(self):
        """Genera un tiempo de reparación según la distribución especificada"""
        if self.repair_dist == 'uniform':
            a, b = self.repair_params
            return uniform.rvs(loc=a, scale=b-a)
        elif self.repair_dist == 'expon':
            scale = self.repair_params[0]
            return expon.rvs(scale=scale)
        elif self.repair_dist == 'normal':
            mean, std = self.repair_params
            return norm.rvs(loc=mean, scale=std)
        else:
            raise ValueError("Distribución de reparación no reconocida")

# %% [markdown]
"""
## 3. Implementación de la Simulación

Clase principal que implementa la lógica de simulación de eventos discretos.
"""

# %%
class MachineRepairSimulation:
    def __init__(self, params):
        self.params = params
        self.reset_simulation()
        
    def reset_simulation(self):
        """Reinicia el estado de la simulación"""
        # Estado inicial: todas las máquinas funcionando
        self.active_machines = self.params.n_machines
        self.inactive_machines = 0
        self.repair_queue = []
        self.under_repair = None
        self.current_time = 0
        self.time_to_failures = self._generate_initial_failures()
        self.next_repair_completion = float('inf')
        self.event_log = []
        self.max_inactive_reached = 0
        
    def _generate_initial_failures(self):
        """Genera tiempos iniciales de fallo para todas las máquinas activas"""
        return sorted(expon.rvs(scale=1/self.params.failure_rate, 
                               size=self.active_machines))
    
    def _log_event(self, event_type, description):
        """Registra un evento en el log"""
        self.event_log.append({
            'time': self.current_time,
            'event_type': event_type,
            'active_machines': self.active_machines,
            'inactive_machines': self.inactive_machines,
            'queue_length': len(self.repair_queue),
            'description': description
        })
    
    def run_simulation(self):
        """Ejecuta la simulación hasta alcanzar el umbral de máquinas inactivas"""
        self.reset_simulation()
        self._log_event('INIT', 'Simulación iniciada')
        
        while True:
            # Determinar el próximo evento
            next_failure = self.time_to_failures[0] if self.time_to_failures else float('inf')
            next_event_time = min(next_failure, self.next_repair_completion)
            
            # Verificar condición de parada
            if self.inactive_machines >= self.params.r_threshold:
                self._log_event('STOP', f'Alcanzado umbral de {self.params.r_threshold} máquinas inactivas')
                return self.current_time
            
            if next_event_time == float('inf'):
                self._log_event('ERROR', 'No hay eventos pendientes')
                return float('inf')
            
            # Avanzar el tiempo
            self.current_time = next_event_time
            
            # Procesar evento
            if next_event_time == next_failure:
                self._process_failure_event()
            else:
                self._process_repair_completion_event()
            
            # Actualizar máximo de máquinas inactivas
            if self.inactive_machines > self.max_inactive_reached:
                self.max_inactive_reached = self.inactive_machines
    
    def _process_failure_event(self):
        """Procesa el evento de fallo de una máquina"""
        # La máquina falla
        self.time_to_failures.pop(0)
        self.active_machines -= 1
        self.inactive_machines += 1
        
        # Añadir a cola de reparación o comenzar reparación inmediatamente
        if self.under_repair is None:
            self._start_repair()
        else:
            self.repair_queue.append(self.current_time)
        
        self._log_event('FAILURE', 'Máquina falló')
        
        # Programar próximo fallo si quedan máquinas activas
        if self.active_machines > 0:
            next_failure_time = self.current_time + expon.rvs(scale=1/self.params.failure_rate)
            # Insertar manteniendo orden
            inserted = False
            for i, t in enumerate(self.time_to_failures):
                if next_failure_time < t:
                    self.time_to_failures.insert(i, next_failure_time)
                    inserted = True
                    break
            if not inserted:
                self.time_to_failures.append(next_failure_time)
    
    def _start_repair(self):
        """Comienza a reparar una máquina"""
        repair_time = self.params.generate_repair_time()
        self.next_repair_completion = self.current_time + repair_time
        self.under_repair = self.current_time  # Tiempo en que entró a reparación
        self._log_event('REPAIR_START', f'Reparación iniciada (duración: {repair_time:.2f})')
    
    def _process_repair_completion_event(self):
        """Procesa el evento de completación de reparación"""
        # Máquina reparada vuelve a funcionar
        self.active_machines += 1
        self.inactive_machines -= 1
        self.under_repair = None
        self.next_repair_completion = float('inf')
        
        self._log_event('REPAIR_END', 'Máquina reparada y reactivada')
        
        # Si hay máquinas en cola, comenzar siguiente reparación
        if self.repair_queue:
            self.repair_queue.pop(0)
            self._start_repair()

# %% [markdown]
"""
## 4. Ejecución de la Simulación y Visualización de Resultados

Ejecutamos múltiples réplicas de la simulación y analizamos los resultados.
"""

# %%
def run_multiple_simulations(params, n_replications=1000):
    """Ejecuta múltiples réplicas de la simulación y recopila resultados"""
    sim = MachineRepairSimulation(params)
    results = []
    event_logs = []
    
    for _ in tqdm(range(n_replications), desc="Ejecutando simulaciones"):
        time_to_threshold = sim.run_simulation()
        results.append(time_to_threshold)
        event_logs.append(sim.event_log)
    
    return results, event_logs

# Configuración de parámetros
sim_params = SimulationParameters(
    n_machines=10,
    failure_rate=0.1,
    r_threshold=3,
    repair_dist='uniform',
    repair_params=(1, 3),
    random_seed=42
)

# Ejecutar simulaciones
results, event_logs = run_multiple_simulations(sim_params, n_replications=1000)

# Convertir resultados a DataFrame para análisis
results_df = pd.DataFrame({'time_to_threshold': results})

# %% [markdown]
"""
### Análisis Estadístico Básico
"""

# %%
# Estadísticas descriptivas
stats = results_df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
print(stats)

# Visualización
plt.figure(figsize=(12, 6))
sns.histplot(results_df['time_to_threshold'], kde=True, bins=30)
plt.title('Distribución del tiempo hasta alcanzar r máquinas inactivas')
plt.xlabel('Tiempo')
plt.ylabel('Frecuencia')
plt.show()

# %% [markdown]
"""
### Visualización de una Trayectoria de Simulación
"""

# %%
# Seleccionar una simulación para visualización detallada
sample_log = event_logs[0]
sample_log_df = pd.DataFrame(sample_log)

# Gráfico de máquinas activas/inactivas a lo largo del tiempo
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.step(sample_log_df['time'], sample_log_df['active_machines'], where='post', label='Máquinas activas')
plt.step(sample_log_df['time'], sample_log_df['inactive_machines'], where='post', label='Máquinas inactivas')
plt.axhline(y=sim_params.r_threshold, color='r', linestyle='--', label='Umbral r')
plt.title('Evolución del estado del sistema')
plt.xlabel('Tiempo')
plt.ylabel('Número de máquinas')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.step(sample_log_df['time'], sample_log_df['queue_length'], where='post', color='g')
plt.title('Longitud de la cola de reparación')
plt.xlabel('Tiempo')
plt.ylabel('Máquinas en cola')
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## 5. Análisis de Sensibilidad

Investiguemos cómo varían los resultados al cambiar los parámetros del sistema.
"""

# %%
def sensitivity_analysis(base_params, param_to_vary, values, n_replications=500):
    """Realiza análisis de sensibilidad para un parámetro"""
    results = []
    
    for value in tqdm(values, desc=f"Variando {param_to_vary}"):
        params_dict = base_params.__dict__.copy()
        params_dict[param_to_vary] = value
        new_params = SimulationParameters(**params_dict)
        
        sim_results, _ = run_multiple_simulations(new_params, n_replications)
        mean_time = np.mean(sim_results)
        results.append(mean_time)
    
    return results

# Variar tasa de fallo
failure_rates = np.linspace(0.05, 0.3, 10)
failure_rate_results = sensitivity_analysis(sim_params, 'failure_rate', failure_rates)

# Variar número de máquinas
n_machines_values = range(5, 20, 2)
n_machines_results = sensitivity_analysis(sim_params, 'n_machines', n_machines_values)

# Visualización
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(failure_rates, failure_rate_results, 'o-')
plt.title('Impacto de la tasa de fallo (λ)')
plt.xlabel('Tasa de fallo (λ)')
plt.ylabel('Tiempo medio hasta umbral')

plt.subplot(1, 2, 2)
plt.plot(n_machines_values, n_machines_results, 'o-')
plt.title('Impacto del número de máquinas (n)')
plt.xlabel('Número de máquinas (n)')
plt.ylabel('Tiempo medio hasta umbral')

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## 6. Conclusiones

Los resultados muestran que:
1. El tiempo hasta alcanzar r máquinas inactivas sigue una distribución [describir forma].
2. Al aumentar la tasa de fallo λ, el tiempo hasta alcanzar el umbral disminuye como era de esperar.
3. Con más máquinas en el sistema, se tarda más en alcanzar el umbral relativo de inactividad.
4. [Agregar más conclusiones específicas basadas en tus resultados]

Este modelo puede extenderse para:
- Considerar múltiples técnicos de reparación
- Incorporar prioridades en la cola de reparación
- Modelar diferentes distribuciones de fallo para diferentes máquinas
"""