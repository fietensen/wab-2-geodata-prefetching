import random

class RandomPrefetcher:
    def __init__(self, tiles):
        self.__all_tiles = tiles


    def choose_tiles(self, ext_factors, num_caches):
        all_tiles = self.__all_tiles.copy()
        prefetched = []

        for i in range(num_caches):
            random_idx = random.randint(0, len(all_tiles)-1)
            prefetched.append(all_tiles.pop(random_idx))
        
        return prefetched