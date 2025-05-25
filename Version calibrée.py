import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

class LOBSim:
    def __init__(self, iprice=100.0, tick_size=0.01):
        self.iprice = iprice
        self.tick_size = tick_size
        self.types = [
            'LO_ask_p1',   
            'CO_ask_p1',   
            'LO_ask0',    
            'CO_ask0',    
            'MO_ask0',        
            'LO_ask_m1', 
            'LO_bid_p1', 
            'MO_bid0',        
            'CO_bid0',    
            'LO_bid0',    
            'CO_bid_m1',   
            'LO_bid_m1'    
        ]
        
        self.calibrated = self.params()
        
        self.r_numbers = [1, 10, 50, 100, 200, 500]
        self.r_weights = np.array([0.35, 0.20, 0.15, 0.15, 0.10, 0.05])
        self.r_weights = self.r_weights / self.r_weights.sum()
        self.geometric_p = 0.05
        
        self.memory_window = 300.0  
        self.k_cut = 0.001
        self.max_intensity = 2.0    
        
    def params(self):
        mu = np.array([0.05, 0.03, 0.15, 0.10, 0.20, 0.005, 0.005, 0.20, 0.10, 0.15, 0.03, 0.05])
        def time_intensity(t):
            fraction = (t % 3600) / 3600
            u = 1.0 + 0.3 * np.sin(2 * np.pi * fraction)
            return max(0.7, min(1.3, u))  
        
        k_params = {
            'a': np.zeros((12, 12)),
            'B': np.ones((12, 12)) * 1.5,   
            'G': np.ones((12, 12)) * 0.1
        }
        
        auto_ex = [0.15, 0.10, 0.20, 0.15, 0.25, 0.05, 0.05, 0.25, 0.15, 0.20, 0.10, 0.15]
        for i in range(12):
            k_params['a'][i, i] = auto_ex[i]
            
        cross_ex = [
            (4, 2, 0.10),   
            (4, 9, 0.08),   
            (7, 9, 0.10),   
            (7, 2, 0.08),   
            # LO excitent MO
            (2, 4, 0.08),   
            (9, 7, 0.08),               
            #CO excitent LO
            (3, 2, 0.12),   
            (8, 9, 0.12),              
            #IS
            (5, 6, 0.05),   
            (6, 5, 0.05),               

            (0, 1, 0.08),   
            (11, 10, 0.08), 
            #MO excitent CO
            (4, 3, 0.06),   
            (7, 8, 0.06),   
        ]
        
        for i, j, a in cross_ex:
            k_params['a'][i, j] = a
        matrice = np.zeros((12, 12))
        for i in range(12):
            for j in range(12):
                if k_params['a'][i, j] > 0:
                    B = k_params['B'][i, j]
                    if B > 1:
                        matrice[i, j] = k_params['a'][i, j] / (B - 1)
            
        return {
            'mu': mu,
            'time_factor': time_intensity,
            'ks': k_params,
            'B_spread': 1.5,
            'matrice': matrice
        }
    
    def initialize_lob(self):
        return {
            'ask_p1': [400, self.iprice + 5*self.tick_size, 12],
            'ask0': [200, self.iprice + self.tick_size, 8],
            'ask_m1': [0, self.iprice, 0],
            'bid_p1': [0, self.iprice, 0],
            'bid0': [200, self.iprice - self.tick_size, 8],
            'bid_m1': [400, self.iprice - 5*self.tick_size, 12]
        }
    
    def power_law_k(self, dt, a, B, G):
        if dt <= 0:
            return 0
        result = a * (1 + G * dt) ** (-B)
        return result if result > self.k_cut else 0
    
    def order_size(self, type):
        if np.random.random() < 0.65:
            return np.random.choice(self.r_numbers, p=self.r_weights)
        else:
            return max(1, min(1000, np.random.geometric(self.geometric_p)))
    
    def spread(self, lob):
        ask_price = lob['ask0'][1] if lob['ask0'][0] > 0 else lob['ask_p1'][1]
        bid_price = lob['bid0'][1] if lob['bid0'][0] > 0 else lob['bid_m1'][1]
        return max(1, (ask_price - bid_price) / self.tick_size)
    
    def clean_history(self, history, current_time):
        cutoff_time = current_time - self.memory_window
        for i in range(12):
            if isinstance(history[i], deque):
                history[i] = list(history[i])
            history[i] = [t for t in history[i] if t > cutoff_time]
            history[i] = deque(history[i], maxlen=500)  
    
    def compute_intensity_optimized(self, event_idx, t, history, lob):
        mu = self.calibrated['mu'][event_idx]
        time_factor = self.calibrated['time_factor'](t)
        
        excitation = 0.0
        cutoff_time = t - self.memory_window
        
        for j in range(12):
            a = self.calibrated['ks']['a'][j, event_idx]
            if a > 0 and len(history[j]) > 0:
                B = self.calibrated['ks']['B'][j, event_idx]
                G = self.calibrated['ks']['G'][j, event_idx]
                
                recent_events = [t_j for t_j in history[j] if t_j > cutoff_time and t_j < t]
                recent_events = recent_events[-50:]  
                
                for t_j in recent_events:
                    dt = t - t_j
                    k_val = self.power_law_k(dt, a, B, G)
                    if k_val > 0:
                        excitation += k_val
        
        excitation = min(excitation, 1.0)  
        
        base_intensity = time_factor * (mu + excitation)
        
        if event_idx in [5, 6]: 
            spread = self.spread(lob)
            if spread <= 1:
                return 0.0
            else:
                spread_factor = (spread - 1) ** self.calibrated['B_spread']
                result = base_intensity * spread_factor
                return max(0, min(result, self.max_intensity))
        
        return max(0, min(base_intensity, self.max_intensity))
    
    def mid_price(self, price, type, lob, order_size=0):
        change = 0
        
        if type == 'LO_ask_m1':
            change = -self.tick_size / 2
        elif type == 'LO_bid_p1':
            change = +self.tick_size / 2
        elif type == 'CO_ask0' and lob['ask0'][2] <= 1:
            change = +self.tick_size / 2
        elif type == 'CO_bid0' and lob['bid0'][2] <= 1:
            change = -self.tick_size / 2
        elif type == 'MO_ask0':
            if order_size >= lob['ask0'][0] * 0.5:
                change = +self.tick_size / 4
        elif type == 'MO_bid0':
            if order_size >= lob['bid0'][0] * 0.5:
                change = -self.tick_size / 4
                
        return price + change
    
    def update_lob(self, lob, type, order_size):
        maj = {}
        for key, value in lob.items():
            maj[key] = value.copy() if isinstance(value, list) else value
        
        if type == 'LO_ask_p1':
            maj['ask_p1'][0] += order_size
            maj['ask_p1'][2] += 1
        elif type == 'CO_ask_p1':
            conso = min(order_size, maj['ask_p1'][0])
            maj['ask_p1'][0] -= conso
            maj['ask_p1'][2] = max(0, maj['ask_p1'][2] - 1)
        elif type == 'LO_ask0':
            maj['ask0'][0] += order_size
            maj['ask0'][2] += 1
        elif type == 'CO_ask0':
            conso = min(order_size, maj['ask0'][0])
            maj['ask0'][0] -= conso
            maj['ask0'][2] = max(0, maj['ask0'][2] - 1)
        elif type == 'MO_ask0':
            conso_top = min(order_size, maj['ask0'][0])
            maj['ask0'][0] -= conso_top
            reste = order_size - conso_top
            if reste > 0:
                conso_deep = min(reste, maj['ask_p1'][0])
                maj['ask_p1'][0] -= conso_deep
        elif type == 'LO_ask_m1':
            spread = self.spread(maj)
            if spread > 1:
                maj['ask_m1'][0] = order_size
                maj['ask_m1'][1] = maj['bid0'][1] + self.tick_size/2
                maj['ask_m1'][2] = 1
        elif type == 'LO_bid_m1':
            maj['bid_m1'][0] += order_size
            maj['bid_m1'][2] += 1
        elif type == 'CO_bid_m1':
            conso = min(order_size, maj['bid_m1'][0])
            maj['bid_m1'][0] -= conso
            maj['bid_m1'][2] = max(0, maj['bid_m1'][2] - 1)
        elif type == 'LO_bid0':
            maj['bid0'][0] += order_size
            maj['bid0'][2] += 1
        elif type == 'CO_bid0':
            conso = min(order_size, maj['bid0'][0])
            maj['bid0'][0] -= conso
            maj['bid0'][2] = max(0, maj['bid0'][2] - 1)
        elif type == 'MO_bid0':
            conso_top = min(order_size, maj['bid0'][0])
            maj['bid0'][0] -= conso_top
            reste = order_size - conso_top
            if reste > 0:
                conso_deep = min(reste, maj['bid_m1'][0])
                maj['bid_m1'][0] -= conso_deep
        elif type == 'LO_bid_p1':
            spread = self.spread(maj)
            if spread > 1:
                maj['bid_p1'][0] = order_size
                maj['bid_p1'][1] = maj['ask0'][1] - self.tick_size/2
                maj['bid_p1'][2] = 1
            
        return maj
    
    def simulate(self, T=3600, max_events=50000):
        t = 0
        lob = self.initialize_lob()
        price = self.iprice
        
        history = [deque(maxlen=500) for _ in range(12)]
        
        times = [0]
        prices = [price]
        spreads = [self.spread(lob)]
        event_log = []
        event_counts = np.zeros(12)
        
        intensity_history = []
        event_rate_history = []
        
        event_count = 0
        cleanup = 0
        rate_check = 0

        while t < T and event_count < max_events:
            if t - cleanup > 60:
                self.clean_history(history, t)
                cleanup = t
            
            if t - rate_check > 30:  
                if t > 0:
                    current_rate = event_count / t
                    event_rate_history.append((t, current_rate))
                rate_check = t
            
            intensities = np.array([
                self.compute_intensity_optimized(i, t, history, lob) 
                for i in range(12)
            ])
            
            total_intensity = np.sum(intensities)
            intensity_history.append((t, total_intensity))
            
            if total_intensity > 20:
                intensities = intensities * (10 / total_intensity)
                total_intensity = np.sum(intensities)
            
            dt = np.random.exponential(1.0 / total_intensity)
            t += dt
            
            probs = intensities / total_intensity
            event_idx = np.random.choice(12, p=probs)
            type = self.types[event_idx]
            
            order_size = self.order_size(type)
            maj = self.update_lob(lob, type, order_size)
            new_price = self.mid_price(price, type, lob, order_size)
            
            history[event_idx].append(t)
            times.append(t)
            prices.append(new_price)
            spreads.append(self.spread(maj))
            event_counts[event_idx] += 1
            
            if len(event_log) < 1000:
                event_log.append({
                    'time': t,
                    'event': type,
                    'size': order_size,
                    'price': new_price,
                    'spread': spreads[-1]
                })
            
            lob = maj
            price = new_price
            event_count += 1
            
            if event_count % 5000 == 0:
                current_rate = event_count / t if t > 0 else 0
                print(f"Événement {event_count}: t={t:.1f}s, Prix=${price:.3f}, ")
        
        return {
            'times': np.array(times),
            'prices': np.array(prices),
            'spreads': np.array(spreads),
            'events': event_log,
            'event_counts': event_counts,
            'final_lob': lob,
            'total_time': t,
            'intensity_history': intensity_history,
            'event_rate_history': event_rate_history
        }
    
    def plot(self, results):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        times = results['times']
        prices = results['prices']
        spreads = results['spreads']
        
        axes[0,0].plot(times, prices, 'b-', alpha = 0.8, linewidth=0.8)
        axes[0,0].set_title('Évolution du Mid-Price', fontsize=12)
        axes[0,0].set_xlabel('Temps')
        axes[0,0].set_ylabel('Prix')
        axes[0,0].grid(True, alpha = 0.3)
        
        axes[0,1].plot(times, spreads, 'r-', alpha = 0.7, linewidth=0.8)
        axes[0,1].set_title('Évolution du Spread', fontsize=12)
        axes[0,1].set_xlabel('Temps')
        axes[0,1].set_ylabel('Spread (ticks)')
        axes[0,1].grid(True, alpha = 0.3)
        
        event_counts = results['event_counts']
        event_labels = [name.replace('_', '\n') for name in self.types]
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        bars = axes[1,0].bar(range(12), event_counts, color=colors, alpha = 0.8)
        axes[1,0].set_title('Répartition des Événements', fontsize=12)
        axes[1,0].set_xlabel('Type d\'événement')
        axes[1,0].set_ylabel('Nombre d\'occurrences')
        axes[1,0].set_xticks(range(12))
        axes[1,0].set_xticklabels(event_labels, fontsize=8)
        axes[1,0].grid(True, alpha = 0.3)
        
        spread_counts = pd.Series(spreads).value_counts().sort_index()
        axes[1,1].bar(spread_counts.index, spread_counts.values, alpha = 0.7, color='purple')
        axes[1,1].set_title('Distribution des Spreads', fontsize=12)
        axes[1,1].set_xlabel('Spread (ticks)')
        axes[1,1].set_ylabel('Fréquence')
        axes[0,1].grid(True, alpha = 0.3)
        
        plt.tight_layout()
        plt.show()
        

if __name__ == "__main__":
    simulator = LOBSim(iprice=100.0, tick_size=0.01)
    results = simulator.simulate(T=1800, max_events=3000)
    analysis = simulator.plot(results)