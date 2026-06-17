import csv
from dataclasses import dataclass
from numpy import format_float_positional
import os
import statistics

import models

def base_cnn(model):
    base = os.path.split(model)[1].split("~")[0]
    if base not in models.available_models():
        raise AttributeError(base + " is not a valid base CNN")
    return base

def create_model_name(base_cnn: str, batch_size: int,
                      learning_rate: float, dropout: float):
    return "~".join([base_cnn,
                     str(batch_size),
                     format_float_positional(learning_rate, trim='-'),
                     str(dropout)])

@dataclass
class StatsResults:
    model: str
    acc: [float]
    epochs: int
    batch_size: int
    dropout: float
    learning_rate: float
    patience: int

    def accuracy(self):
        return statistics.mean(self.acc)

    def base(self):
        return base_cnn(self.model)

    def img_dims(self):
        conf = models.get_model_config(self.base())
        return(int(conf["image_width"]), int(conf["image_height"]))

    def __lt__(self, other):
        return self.accuracy() < other.accuracy()

    def __repr__(self):
        ret = self.model + " (acc=" + str(self.accuracy())

        for cat, acc in enumerate(self.acc):
            ret += ", cat_" + str(cat) + "=" + str(acc)

        if len(self.acc) == 2:
            # Calculate the Macro-F1 score while we're here.
            tp = self.acc[1]
            fn = 1 - tp
            tn = self.acc[0]
            fp = 1 - tn

            f1 = (2*tp) / (2*tp + fp + fn)
            ret += ", f1=" + str(f1)

        ret += ", base=" + self.base() + ")"
        return ret

# extract models from the model stats CSV file
def loadModels(stats_file, base_model=None):
    model_stats = []
    with open(stats_file, 'r') as csvfile:
        csvdata = csv.DictReader(csvfile)
        for row in csvdata:
            # Optionally filter by base CNN
            if base_model is not None and base_cnn(row['model']) != base_model:
                continue

            # pull out all of the accuracy stats
            accuracies = []
            cat = 0
            while ("accuracy_" + str(cat)) in row:
                accuracies.append(float(row["accuracy_" + str(cat)]))
                cat += 1

            results = StatsResults(model = row['model'],
                                   acc = accuracies,
                                   epochs = int(row['epochs']),
                                   batch_size = int(row['batch_size']),
                                   dropout = float(row['dropout']),
                                   learning_rate = float(row['learning_rate']),
                                   patience = int(row['patience']))
            model_stats.append(results)

    return model_stats

# EOF
