import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class LOBSim:
    def __init__(self, iprice, tick_size, seed=None):
        self.iprice = iprice
        self.tick_size = tick_size
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self.types = [
            'LO_ask_p1', 'CO_ask_p1', 'LO_ask0', 'CO_ask0', 'MO_ask0', 'LO_ask_m1',
            'LO_bid_p1', 'MO_bid0', 'CO_bid0', 'LO_bid0', 'CO_bid_m1', 'LO_bid_m1'
        ]

        self.calibrated = self.params()
        self.r_numbers = [1, 10, 50, 100, 200, 500]
        self.r_weights = np.array([0.35, 0.20, 0.15, 0.15, 0.10, 0.05])
        self.geometric_p = 0.05
        self.memory_window = 300.0
        self.k_cut = 0.001
        self.max_intensity = 2.0

    def params(self):
        mu = np.array([0.05, 0.03, 0.15, 0.10, 0.20, 0.005, 0.005, 0.20, 0.10, 0.15, 0.03, 0.05])
        
        def time_intensity(t):
            return max(0.7, min(1.3, 1.0 + 0.3 * np.sin(2 * np.pi * (t % 3600) / 3600)))

        k_params = {'a': np.zeros((12, 12)), 'B': np.ones((12, 12)) * 1.5, 'G': np.ones((12, 12)) * 0.1}
        
        auto_ex = [0.15, 0.10, 0.20, 0.15, 0.25, 0.05, 0.05, 0.25, 0.15, 0.20, 0.10, 0.15]
        for i in range(12):
            k_params['a'][i, i] = auto_ex[i]

        cross_ex = [(4, 2, 0.10), (4, 9, 0.08), (7, 9, 0.10), (7, 2, 0.08), (2, 4, 0.08), 
                   (9, 7, 0.08), (3, 2, 0.12), (8, 9, 0.12), (5, 6, 0.05), (6, 5, 0.05),
                   (0, 1, 0.08), (11, 10, 0.08), (4, 3, 0.06), (7, 8, 0.06)]

        for i, j, a in cross_ex:
            k_params['a'][i, j] = a

        return {'mu': mu, 'time_factor': time_intensity, 'k_params': k_params, 'B_spread': 1.5}
    
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
        return max(1, min(1000, np.random.geometric(self.geometric_p)))

    def spread(self, lob):
        ask_price = lob['ask0'][1] if lob['ask0'][0] > 0 else lob['ask_p1'][1]
        bid_price = lob['bid0'][1] if lob['bid0'][0] > 0 else lob['bid_m1'][1]
        return max(1, (ask_price - bid_price) / self.tick_size)
    
    def clean_history(self, history, current_time):
        cutoff_time = current_time - self.memory_window
        for i in range(12):
            history[i] = deque([t for t in history[i] if t > cutoff_time], maxlen=500)

    def intensity(self, event_idx, t, history, lob):
        mu = self.calibrated['mu'][event_idx]
        time_factor = self.calibrated['time_factor'](t)
        excitation = 0.0
        cutoff_time = t - self.memory_window

        for j in range(12):
            a = self.calibrated['k_params']['a'][j, event_idx]
            if a > 0:
                B = self.calibrated['k_params']['B'][j, event_idx]
                G = self.calibrated['k_params']['G'][j, event_idx]
                recent_events = [t_j for t_j in history[j] if cutoff_time < t_j < t][-50:]
                
                for t_j in recent_events:
                    k_val = self.power_law_k(t - t_j, a, B, G)
                    if k_val > 0:
                        excitation += k_val

        base_intensity = time_factor * (mu + min(excitation, 1.0))

        if event_idx in [5, 6]:  # IS events
            spread = self.spread(lob)
            if spread <= 1:
                return 0.0
            base_intensity *= (spread - 1) ** self.calibrated['B_spread']

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
        elif type == 'MO_ask0' and order_size >= lob['ask0'][0] * 0.5:
            change = +self.tick_size / 4
        elif type == 'MO_bid0' and order_size >= lob['bid0'][0] * 0.5:
            change = -self.tick_size / 4
        return price + change

    def update_lob(self, lob, type, order_size):
        maj = {key: value.copy() for key, value in lob.items()}
        
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
                maj['ask_p1'][0] -= min(reste, maj['ask_p1'][0])
        elif type == 'LO_ask_m1':
            if self.spread(maj) > 1:
                maj['ask_m1'] = [order_size, maj['bid0'][1] + self.tick_size/2, 1]
        elif type in ['LO_bid_m1', 'CO_bid_m1', 'LO_bid0', 'CO_bid0', 'MO_bid0']:
            if type == 'LO_bid_m1':
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
                    maj['bid_m1'][0] -= min(reste, maj['bid_m1'][0])
        elif type == 'LO_bid_p1':
            if self.spread(maj) > 1:
                maj['bid_p1'] = [order_size, maj['ask0'][1] - self.tick_size/2, 1]

        return maj
    
    def execute_MO(self, lob, side, order_size):
        maj = lob.copy()
        reste = order_size
        cost = 0.0
        exe_volume = 0
        exe_levels = []

        levels = ['ask0', 'ask_p1'] if side == 'buy' else ['bid0', 'bid_m1']
        
        for level in levels:
            if reste <= 0:
                break
            dispo = maj[level][0]
            if dispo > 0:
                conso = min(reste, dispo)
                price = maj[level][1]
                cost += conso * price
                exe_volume += conso
                reste -= conso
                maj[level][0] -= conso
                maj[level][2] = 0 if maj[level][0] == 0 else max(1, maj[level][2] - 1)
                exe_levels.append((level, conso, price))

        if reste > 0:
            exe_levels.append(('CANCELLED', reste, 0))

        execution_price = cost / exe_volume if exe_volume > 0 else 0
        
        return maj, execution_price, {}, exe_volume, reste, exe_levels
    
    def create_meta_order_schedule(self, remaining_size, duration, start_time, cycle_num=1):
        n_orders = max(5, min(50, int(remaining_size / 100)))
        
        child_size = remaining_size // n_orders
        remainder = remaining_size % n_orders
        
        interval = duration / n_orders
        
        schedule = []
        for i in range(n_orders):
            execution_time = start_time + i * interval
            size = child_size + (1 if i < remainder else 0)
            schedule.append((execution_time, size))
        
        return schedule
    
    def calculate_market_impact(self, pre, exe_price, post, side):
        if side == 'buy':
            immediate_impact = exe_price - pre
            permanent_impact = post[-1] - pre if post else 0
        else:
            immediate_impact = pre - exe_price
            permanent_impact = pre - post[-1] if post else 0
            
        return {
            'immediate_impact': immediate_impact,
            'permanent_impact': permanent_impact,
            'implementation_shortfall': immediate_impact,
            'temporary_impact': immediate_impact - permanent_impact
        }
    
    def simulate_with_meta_order(self, T=3600, max_events=50000, MO_time=1800, MO_size=1000, MO_side='buy', MO_duration=300, observation_window=300, max_cycles=10):
        t = 0
        lob = self.initialize_lob()
        price = self.iprice
        history = [deque(maxlen=500) for _ in range(12)]
        
        times = [0]
        prices = [price]
        spreads = [self.spread(lob)]
        event_log = []
        event_counts = np.zeros(12)
        
        original_order_size = MO_size
        remaining_size = MO_size
        current_cycle = 1
        meta_order_schedule = []
        meta_order_executions = []
        next_meta_order_idx = 0
        cycle_start_time = MO_time
        
        pre_order_window_prices = []
        post_order_window_prices = []
        
        meta_order_started = False
        meta_order_completed = False
        total_executed_volume = 0
        total_cost = 0.0
        pre_meta_price = None
        all_cycles_info = []
        
        event_count = 0
        cleanup = 0

        while t < T and event_count < max_events:
            if t - cleanup > 60:
                self.clean_history(history, t)
                cleanup = t

            if MO_time - observation_window <= t < MO_time:
                pre_order_window_prices.append(price)

            if (remaining_size > 0 and current_cycle <= max_cycles and 
                len(meta_order_schedule) == 0 and t >= cycle_start_time):
                
                meta_order_schedule = self.create_meta_order_schedule(
                    remaining_size, MO_duration, cycle_start_time, current_cycle
                )
                next_meta_order_idx = 0
                
                print(f"Cycle {current_cycle}: {remaining_size} actions à exécuter, "
                      f"{len(meta_order_schedule)} child orders")

            if (len(meta_order_schedule) > 0 and 
                next_meta_order_idx < len(meta_order_schedule) and 
                t >= meta_order_schedule[next_meta_order_idx][0]):
                
                if not meta_order_started:
                    meta_order_started = True
                    pre_meta_price = price
                
                execution_time, child_size = meta_order_schedule[next_meta_order_idx]
                
                new_lob, exe_price, conso_volumes, exe_volume, reste_volume, exe_levels = self.execute_MO(lob, MO_side, child_size)
                lob = new_lob
                
                execution_info = {
                    'cycle': current_cycle,
                    'execution_time': execution_time,
                    'child_size': child_size,
                    'exe_price': exe_price,
                    'exe_volume': exe_volume,
                    'reste_volume': reste_volume,
                    'exe_levels': exe_levels
                }
                meta_order_executions.append(execution_info)
                
                total_executed_volume += exe_volume
                if exe_volume > 0:
                    total_cost += exe_volume * exe_price
                
                if lob['ask0'][0] > 0 and lob['bid0'][0] > 0:
                    price = (lob['ask0'][1] + lob['bid0'][1]) / 2
                else:
                    ask_price = lob['ask0'][1] if lob['ask0'][0] > 0 else lob['ask_p1'][1]
                    bid_price = lob['bid0'][1] if lob['bid0'][0] > 0 else lob['bid_m1'][1]
                    price = (ask_price + bid_price) / 2
                
                times.append(t)
                prices.append(price)
                spreads.append(self.spread(lob))
                
                next_meta_order_idx += 1
                
                if next_meta_order_idx >= len(meta_order_schedule):
                    cycle_executed_volume = sum(ex['exe_volume'] for ex in meta_order_executions 
                                              if ex['cycle'] == current_cycle)
                    remaining_size -= cycle_executed_volume
                    
                    cycle_info = {
                        'cycle': current_cycle,
                        'start_time': cycle_start_time,
                        'end_time': t,
                        'planned_volume': sum(size for _, size in meta_order_schedule),
                        'executed_volume': cycle_executed_volume,
                        'remaining_after_cycle': remaining_size
                    }
                    all_cycles_info.append(cycle_info)
                    
                    print(f"Cycle {current_cycle} terminé: {cycle_executed_volume} exécutés, "
                          f"{remaining_size} restants")
                    
                    meta_order_schedule = []
                    
                    if remaining_size > 0 and current_cycle < max_cycles:
                        current_cycle += 1
                        cycle_start_time = t + 30 
                    else:
                        meta_order_completed = True
                        print(f"Meta-order terminé après {current_cycle} cycles")

            if (meta_order_completed and 
                t <= cycle_start_time + observation_window):
                post_order_window_prices.append(price)

            intensities = np.array([self.intensity(i, t, history, lob) for i in range(12)])
            total_intensity = np.sum(intensities)
            
            if total_intensity > 20:
                intensities = intensities * (10 / total_intensity)
                total_intensity = np.sum(intensities)

            dt = np.random.exponential(1.0 / total_intensity)
            t += dt

            event_idx = np.random.choice(12, p=intensities / total_intensity)
            type = self.types[event_idx]
            order_size = self.order_size(type)
            
            maj = self.update_lob(lob, type, order_size)
            new_price = self.mid_price(price, type, lob, order_size)

            history[event_idx].append(t)
            times.append(t)
            prices.append(new_price)
            spreads.append(self.spread(maj))
            event_counts[event_idx] += 1

            if len(event_log) < 2000:
                event_log.append({'time': t, 'event': type, 'size': order_size, 'price': new_price, 'spread': spreads[-1]})

            lob = maj
            price = new_price
            event_count += 1

        impact_analysis = None
        MO_results = None
        
        if meta_order_started and total_executed_volume > 0:
            avg_execution_price = total_cost / total_executed_volume
            
            if all_cycles_info:
                total_duration = all_cycles_info[-1]['end_time'] - MO_time
                actual_end_time = all_cycles_info[-1]['end_time']
            else:
                total_duration = MO_duration
                actual_end_time = MO_time + MO_duration
            
            MO_results = {
                'start_time': MO_time,
                'end_time': actual_end_time,
                'planned_duration': MO_duration,
                'actual_duration': total_duration,
                'total_order_size': original_order_size,
                'exe_volume': total_executed_volume,
                'reste_volume': original_order_size - total_executed_volume,
                'fill_rate': total_executed_volume / original_order_size,
                'avg_exe_price': avg_execution_price,
                'n_cycles': current_cycle if meta_order_completed else current_cycle - 1,
                'n_child_orders': len(meta_order_executions),
                'child_executions': meta_order_executions,
                'cycles_info': all_cycles_info,
                'fully_filled': (original_order_size - total_executed_volume) == 0
            }
            
            impact_analysis = self.calculate_market_impact(pre_meta_price, avg_execution_price, post_order_window_prices, MO_side)
            impact_analysis.update({
                'pre_order_avg_price': np.mean(pre_order_window_prices) if pre_order_window_prices else pre_meta_price,
                'post_order_avg_price': np.mean(post_order_window_prices) if post_order_window_prices else avg_execution_price,
                'exe_price': avg_execution_price,
                'exe_volume': total_executed_volume,
                'reste_volume': original_order_size - total_executed_volume,
                'fill_rate': total_executed_volume / original_order_size,
                'meta_order_duration': total_duration,
                'n_child_orders': len(meta_order_executions),
                'n_cycles': current_cycle if meta_order_completed else current_cycle - 1,
                'fully_filled': (original_order_size - total_executed_volume) == 0
            })

        return {
            'times': np.array(times), 'prices': np.array(prices), 'spreads': np.array(spreads),
            'events': event_log, 'event_counts': event_counts, 'final_lob': lob,
            'total_time': t, 'MO_results': MO_results, 'impact_analysis': impact_analysis,
            'MO_time': MO_time if meta_order_started else None,
            'meta_order_schedule': meta_order_schedule,
            'meta_order_executions': meta_order_executions,
            'cycles_info': all_cycles_info
        }
    
    def plot_with_impact(self, results):
        fig, axes = plt.subplots(1, 2, figsize=(20, 16))

        #Price
        for i, result in enumerate(results):
            times, prices = result['times'], result['prices']
            if result['MO_time'] and result['MO_results']:
                start = result['MO_results']['start_time']
                end = result['MO_results']['end_time']
                axes[1].axvline(start, linestyle='--', alpha=0.5, color=f'C{i}', label=f'Début MO {i+1}')
                axes[1].axvline(end, linestyle=':', alpha=0.5, color=f'C{i}', label=f'Fin MO {i+1}')
            axes[1].plot(times, prices, alpha=0.8, linewidth=1)

        axes[1].set_title('Évolution du Prix Mid avec Meta-Orders Linéaires')
        axes[1].set_xlabel('Temps (s)')
        axes[1].set_ylabel('Prix')

        impacts = {'immediate': [], 'permanent': [], 'temporary': []}
        order_sizes = []
        durations = []
        
        for result in results:
            if result['impact_analysis']:
                impact = result['impact_analysis']
                impacts['immediate'].append(impact['immediate_impact'])
                impacts['permanent'].append(impact['permanent_impact'])
                impacts['temporary'].append(impact['temporary_impact'])
                order_sizes.append(result['MO_results']['total_order_size'])
                durations.append(result['MO_results']['actual_duration'])

        if order_sizes:
            x_pos = np.arange(len(order_sizes))
            width = 0.25
            axes[2].bar(x_pos - width, impacts['immediate'], width, label='Impact Immédiat', alpha=0.8)
            axes[2].bar(x_pos, impacts['permanent'], width, label='Impact Permanent', alpha=0.8)
            axes[2].bar(x_pos + width, impacts['temporary'], width, label='Impact Temporaire', alpha=0.8)
            axes[2].set_xtick_params(x_pos)
            axes[2].set_xticklabels([f'{int(size)}' for size in order_sizes])

        axes[2].set_title('Décomposition de l\'Impact de Marché')
        axes[2].set_xlabel('Taille du Meta-Order')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sizes = [0, 1000]
    durations = [100, 100] 
    results = []

    for i, (size, duration) in enumerate(zip(sizes, durations)):
        sim = LOBSim(iprice=100.0, tick_size=0.01, seed=834)
        result = sim.simulate_with_meta_order(
            T=3600, 
            max_events=25000, 
            MO_time=1700, 
            MO_size=size, 
            MO_side='buy', 
            MO_duration=duration,
            observation_window=500
        )
        results.append(result)
    sim.plot_with_impact(results)