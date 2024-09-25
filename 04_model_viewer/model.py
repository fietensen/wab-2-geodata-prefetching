import collections
import math
import pandas as pd
import pickle
import os

class MultinomialNBClassifier:
    def __init__(self, df: pd.DataFrame, tile_x_range: tuple[int, int], tile_y_range: tuple[int, int], data_classes: dict[str, list]):
        self.__priors = collections.defaultdict(int)
        self.__likelihoods = collections.defaultdict(int)

        df_tiles = df.drop(columns=data_classes.keys()).columns
        tile_identifiers = {}
        df_len = len(df)

        # Zähler für A-priori-Wahrscheinlichkeiten
        for tile_x in range(tile_x_range[0], tile_x_range[1]):
            for tile_y in range(tile_y_range[0], tile_y_range[1]):
                tile = (tile_x, tile_y)
                t_id = ";".join([str(tile_x), str(tile_y)])
                tile_identifiers[tile] = t_id

                if not t_id in df_tiles:
                    self.__priors[tile] = math.log(df_len)
                else:
                    self.__priors[tile] = math.log((df.get(t_id) + 1).sum())

        # Zahl der Tile-Zugriffe + Laplace Glättung
        tile_access_count_total = math.log(df[df_tiles].sum().sum() + (tile_x_range[1]-tile_x_range[0]) * (tile_y_range[1]-tile_y_range[0]))*df_len
        for tile in self.__priors.keys():
            self.__priors[tile] -= tile_access_count_total
            if self.__priors[tile] > 0:
                print(tile, self.__priors[tile], tile_identifiers[tile] in df_tiles)
        
        print("Calculated priors")

        # partielle Likelihood-Wahrscheinlichkeiten
        for dc_name, dc_options in data_classes.items():
            for dc_value in dc_options:
                opt_df = df[df[dc_name] == dc_value]
                for tile in self.__priors.keys():
                    t_id = tile_identifiers[tile]
                    if not t_id in df_tiles:
                        self.__likelihoods[(tile, (dc_name, dc_value))] = math.log(
                            1 / df_len
                        )
                        continue
                    
                    self.__likelihoods[(tile, (dc_name, dc_value))] = math.log(
                        (opt_df.get(t_id).sum() + 1)
                        /
                        (df.get(t_id).sum() + df_len)
                    )

        print("Calculated partial likelihoods")
    
    def save(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)

    def load(path):
        with open(path, 'rb') as fp:
            return pickle.load(fp)

    def predict(self, data: dict[str, int]):
        probabilities = {}

        for tile, P_tile in self.__priors.items():
            P_tile_given_c = P_tile
            for cpair in data.items():
                # P(tile)P(data|tile) = log(P(tile))+log(P(data|tile))
                P_tile_given_c += self.__likelihoods[(tile, cpair)]
            
            probabilities[tile] = P_tile_given_c

        # Normalisieren der Wahrscheinlichkeiten für Darstellung auf der interaktiven Karte 
        max_p = max(probabilities.values())
        for pk in probabilities.keys():
            probabilities[pk] -= max_p
            probabilities[pk] = math.e**probabilities[pk]
        
        return probabilities


# Wird dieses Skript direkt aufgerufen, wird das Modell zwischengespeichert
if __name__ == '__main__':
    MODEL_PATH = './data/model.pkl'

    mnbc = None

    if os.path.isfile(MODEL_PATH+'AA'):
        mnbc = MultinomialNBClassifier.load(MODEL_PATH)
    else:
        df = pd.read_csv('./data/synth_access_data.csv')
        print("Read CSV")
        mnbc = MultinomialNBClassifier(df, (0, 195), (0, 104), {
            'temp': [0, 1, 2, 3, 4],
            'snow': [0, 1, 2, 3, 4],
            'wspd': [0, 1, 2, 3, 4],
            'coco': [0, 1, 2],
            'vacation': [0, 1],
            'holiday': [0, 1],
            'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'weekday': [0, 1, 2, 3, 4, 5, 6],
            'hour': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        })
        print(max(mnbc.predict({'wspd': 2, 'snow': 0}, return_log=True).values()))

        print("Saving model @ '{}'".format(MODEL_PATH))
        mnbc.save(MODEL_PATH)