from benchmark_classes.cache_hitrate_benchmark import CacheHitrateBenchmark
from benchmark_classes.random_prefetcher import RandomPrefetcher
from benchmark_classes.bayes_prefetcher import BayesPrefetcher
from benchmark_classes.hitrate_generator import HitrateGenerator
from benchmark_classes.model import MultinomialNBClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

features = {
    'temp': [0, 1, 2, 3, 4],
    'snow': [0, 1, 2, 3, 4],
    'wspd': [0, 1, 2, 3, 4],
    'coco': [0, 1, 2],
    'vacation': [0, 1],
    'holiday': [0, 1],
    'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'weekday': [0, 1, 2, 3, 4, 5, 6],
    'hour': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
}

if __name__ == '__main__':
    benchmark = CacheHitrateBenchmark(features)

    # Zugriffsdaten werden geladen, um tiles mit Zugriffsrate von 0 aus dem zufälligen Prefetching auszuschließen
    _access_df = pd.read_csv('./data/synth_access_data.csv', index_col=0)
    _access_tile_ids = _access_df.drop(columns=list(features.keys()) + ['index']).columns
    _access_tiles = []
    for access_tid in _access_tile_ids:
        t_part_x, t_part_y = access_tid.split(';')
        _access_tiles.append((int(t_part_x), int(t_part_y)))

    # Zugriffszahlen für jedes mindestens 1x besuchte Tile berechnen
    _access_counts = {}
    for access_tid in _access_tile_ids:
        t_part_x, t_part_y = access_tid.split(';')
        tile = (int(t_part_x), int(t_part_y))
        _access_counts[tile] = _access_df[[access_tid]].sum().item()

    random_prefetcher = RandomPrefetcher(_access_tiles)
    bayes_prefetcher = BayesPrefetcher('./data/model.pkl')

    hitrate_generator = HitrateGenerator('postgresql://postgres:toor@localhost:5432/postgres')

    benchmark.add_cache('RandomPrefetcher', random_prefetcher.choose_tiles)
    benchmark.add_cache('BayesPrefetcher', bayes_prefetcher.choose_tiles)

    bench_result = benchmark.benchmark_random(hitrate_generator.make_hit_gen, n_random=200, n_reruns=5)

    for alg_name, hit_rate in bench_result.items():
        print("Algorithm: {} | Hit Rate%: {:.2f}".format(alg_name, hit_rate))

    alg_names, alg_hitrates = zip(*bench_result.items())
    alg_hitrates = np.array(alg_hitrates) * 100
    plt.bar(alg_names, alg_hitrates, color=['tab:orange', 'tab:blue', 'tab:green'])
    for i in range(len(alg_names)):
        plt.text(i-0.05, alg_hitrates[i]+0.2, "{:.1f}".format(alg_hitrates[i]))

    plt.title("Cache Hit-Rates%")
    plt.savefig("05_benchmark_result.png")
    plt.show()