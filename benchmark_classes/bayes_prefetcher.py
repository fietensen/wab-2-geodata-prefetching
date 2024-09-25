import sys
sys.path.append("./04_model_viewer")

from model import MultinomialNBClassifier

class BayesPrefetcher:
    def __init__(self, model_path):
        self.__model = MultinomialNBClassifier.load(model_path)


    def choose_tiles(self, ext_factors, num_caches):
        predictions = self.__model.predict(ext_factors)
        predictions,_ = zip(*sorted(predictions.items(), key=lambda i: i[1], reverse=True))

        return predictions[:num_caches]