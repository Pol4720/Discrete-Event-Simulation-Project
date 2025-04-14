import heapq
import numpy as np
import matplotlib.pyplot as plt


class SystemSimulator:
    def __init__(self, n, s, F_dist, G_dist):
        """
        Inicializa el simulador del sistema
        n: Máquinas operativas requeridas
        s: Máquinas de repuesto
        F_dist: Función que genera tiempos de falla
        G_dist: Función que genera tiempos de reparación
        """
        self.n = n
        self.s = s
        self.F = F_dist
        self.G = G_dist

    def single_run(self):
        """Ejecuta una sola simulación hasta el colapso del sistema"""
        t = 0
        r = 0
        t_star = np.inf
        heap = []
        
        # Generar tiempos iniciales de falla
        failure_times = self.F(size=self.n)
        for ft in np.sort(failure_times):
            heapq.heappush(heap, ft)
        
        while True:
            t1 = heap[0] if heap else np.inf
            t_star = t_star if t_star is not None else np.inf
            
            # Determinar próximo evento
            if t1 < t_star:
                # Caso 1: Fallo de máquina
                t = heapq.heappop(heap)
                r += 1
                
                if r == self.s + 1:
                    return t
                
                # Generar nuevo tiempo de falla para el repuesto
                new_X = self.F()
                new_failure = t + new_X
                heapq.heappush(heap, new_failure)
                
                if r == 1:
                    # Iniciar reparación
                    repair_time = self.G()
                    t_star = t + repair_time
                    
            else:
                # Caso 2: Reparación completada
                t = t_star
                r -= 1
                
                if r > 0:
                    # Generar nuevo tiempo de reparación
                    repair_time = self.G()
                    t_star = t + repair_time
                else:
                    t_star = np.inf

    def simulate(self, k_runs, c, d):
        """Ejecuta múltiples simulaciones y calcula el tiempo promedio"""
        crash_times = []
        for _ in range(k_runs):
            crash_time = self.single_run()
            crash_times.append(crash_time)
        
        sample_std_dev = np.std(crash_times)
        
        while ((c * (sample_std_dev / np.sqrt(len(crash_times)))) >= d):
            crash_time = self.single_run()
            crash_times.append(crash_time)
            sample_std_dev = np.std(crash_times)
            
        
        return np.mean(crash_times), crash_times
    
    

# Ejemplo de uso con distribuciones exponenciales
if __name__ == "__main__":
    # Parámetros
    n = 5
    s = 5
    lambda_fail = 0.1
    mu_repair = 0.5
    k = 1000

    # Definir distribuciones
    F_dist = lambda size=None: np.random.exponential(1/lambda_fail, size)
    G_dist = lambda: np.random.exponential(1/mu_repair)

    # Crear simulador
    simulator = SystemSimulator(n, s, F_dist, G_dist)
    c = 1.5
    d = 0.5
    # Ejecutar simulación
    mean_time = simulator.simulate(k, c, d)
    print(f"Tiempo promedio hasta el colapso: {mean_time[0]:.2f}")
    print(len(mean_time[1]))