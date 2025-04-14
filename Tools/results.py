import heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from simulator import SystemSimulator



# Función para análisis de sensibilidad
def sensitivity_analysis(param_values, param_name):
    results = []
    n = 5
    s = 2
    k = 100
    c = 1.96
    d = 0.05
      # Definir distribuciones
    F_dist = lambda size=None: np.random.exponential(1/lambda_fail, size)
    G_dist = lambda: np.random.exponential(1/mu_repair)
    
    for value in param_values:
        sim = SystemSimulator(n,s,F_dist, G_dist)
        data = sim.simulate(k,c,d)
        results.append({
            'mean': np.mean(data),
            'std': np.std(data),
            'n': len(data)
        })
    return results

# 1. Validar hallazgo principal
print("=== Hallazgo principal ===")
base_sim = SystemSimulator()
base_data = base_sim.simulate()
print(f"E[T] = {np.mean(base_data):.1f} ± {1.96*np.std(base_data)/np.sqrt(len(base_data)):.1f}")
print(f"Simulaciones requeridas: {len(base_data)}")

# 2. Análisis de sensibilidad para s
s_values = [2, 3, 4, 5]
s_results = sensitivity_analysis(s_values, 's')

plt.figure(figsize=(10,6))
plt.plot(s_values, [res['mean'] for res in s_results], 'bo-')
plt.xlabel('Número de repuestos (s)')
plt.ylabel('E[T]')
plt.title('Impacto de repuestos en tiempo hasta colapso')
plt.grid(True)
plt.show()

# 3. Validar hipótesis 1 (rendimientos decrecientes)
print("\n=== Hipótesis 1: Rendimientos decrecientes ===")
incrementos = []
for i in range(1, len(s_values)):
    aumento = (s_results[i]['mean'] - s_results[i-1]['mean'])/s_results[i-1]['mean']*100
    incrementos.append(aumento)

plt.bar(range(len(incrementos)), incrementos)
plt.xticks(range(len(incrementos)), [f"{s_values[i-1]}→{s_values[i]}" for i in range(1, len(s_values))])
plt.xlabel('Incremento en s')
plt.ylabel('Aumento porcentual en E[T]')
plt.title('Rendimientos marginales decrecientes')
plt.show()

# 4. Validar hipótesis 2 (reparadores adicionales)
print("\n=== Hipótesis 2: Efecto de reparadores adicionales ===")
class MultiRepairSystem(SystemSimulator):
    def __init__(self, m=2, **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.repair_heap = []
        
    def single_run(self):
        # Implementación modificada para múltiples reparadores
        t = 0
        r = 0
        heap = []
        failure_times = np.random.exponential(1/self.lambda_fail, self.n)
        for ft in np.sort(failure_times):
            heapq.heappush(heap, ft)
            
        while True:
            t1 = heap[0] if heap else np.inf
            t_repair = self.repair_heap[0] if self.repair_heap else np.inf
            
            if t1 < t_repair:
                t = heapq.heappop(heap)
                r += 1
                
                if r > self.s:
                    return t
                
                heapq.heappush(heap, t + np.random.exponential(1/self.lambda_fail))
                
                if r <= self.m:
                    heapq.heappush(self.repair_heap, t + np.random.exponential(1/self.mu_repair))
            else:
                t = heapq.heappop(self.repair_heap)
                r -= 1

# Comparar sistemas con 1 vs 2 reparadores
sim_m1 = MultiRepairSystem(m=1)
sim_m2 = MultiRepairSystem(m=2)

data_m1 = sim_m1.simulate()
data_m2 = sim_m2.simulate()

print(f"Con 1 reparador: E[T] = {np.mean(data_m1):.1f}, σ = {np.std(data_m1):.1f}")
print(f"Con 2 reparadores: E[T] = {np.mean(data_m2):.1f}, σ = {np.std(data_m2):.1f}")

# 5. Análisis de distribución
plt.figure(figsize=(10,6))
plt.hist(base_data, bins=50, density=True, alpha=0.7)
plt.xlabel('Tiempo hasta colapso')
plt.ylabel('Densidad')
plt.title('Distribución de tiempos de colapso')
plt.show()
