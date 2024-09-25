import random
import collections

class CacheHitrateBenchmark:
    def __init__(self, data_classes: dict[str, list[int]]):
        self.__data_classes = data_classes
        self.__prefetch_caches = {}


    def add_cache(self, descriptor: str, generate_priority_queue):
        self.__prefetch_caches[descriptor] = generate_priority_queue


    def benchmark_random(self, hit_gen_func, n_random=100, n_reruns=50):
        avg_hitrates = {}

        for i in range(n_random):
            factors = self._generate_random_state()
            hit_generator = hit_gen_func(factors)
            hit_rates = collections.defaultdict(list)
            
            for j in range(n_reruns):
                # get 50% of most requested tiles (tiles, that had at least 1 hit)
                # ordered by request amount
                requested = next(hit_generator)
                prefetched_tiles = collections.defaultdict(list)
                local_hits = {}

                for alg_name, alg_func in self.__prefetch_caches.items():
                    # get len(requested) most likely tiles
                    prefetched_tiles[alg_name] = alg_func(factors, len(requested))
                    local_hits[alg_name] = 0

                for tile in requested:
                    for alg_name in self.__prefetch_caches.keys():
                        if not tile in prefetched_tiles[alg_name]:
                            continue

                        local_hits[alg_name] += 1

                for alg_name, amt_hits in local_hits.items():
                    hit_rates[alg_name].append(amt_hits / len(requested))
            
            for alg_name, l_hit_rates in hit_rates.items():
                avg_hitrates[alg_name] = sum(l_hit_rates) / len(l_hit_rates)
        
        return avg_hitrates
            

    def _generate_random_state(self):
        random_state = {}
        
        for data_class, options in self.__data_classes.items():
            random_state[data_class] = random.choice(options)
        
        return random_state