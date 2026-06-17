#!/usr/bin/env python

# hackety hack - tensorflow has lots of warnings. We don't want to see them.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import tensorflow as tf

from train_thyself import runAllBaseCNNs, ModelArgs, MODEL_TYPES

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--batch_size", help="TensorFlow batch size",
                           default=16, type=int)
    argparser.add_argument("--learning_rate", help="TensorFlow learning rate",
                           default=0.0001, type=float)
    argparser.add_argument("--dropout", help="Dropout rate",
                           default=0.2, type=float)
    argparser.add_argument("--model_type", help="Type of model to run",
                           default="refer", choices=MODEL_TYPES)
    argparser.add_argument("--cpu", help="don't use GPU for computations",
                           action="store_true")
    argparser.add_argument("--heatmap_dir", type=str,
                           default=os.path.join("..", "heatmaps_refer"),
                           help="generate heatmaps in this directory")
    argparser.add_argument("--heatmap_img_dir", type=str,
                           default=os.path.join("..", "refer_heatmap_images"),
                           help="path to images used to generate heatmaps")
    argparser.add_argument("--coordinates_csv", type=str,
                           default=os.path.join("..", "coordinates.csv"),
                           help="path to CSV file containing mac and disc coordinates")
    args = argparser.parse_args()

    m_args = ModelArgs(batch_size = args.batch_size,
                       learning_rate = args.learning_rate,
                       dropout = args.dropout,
                       heatmap_dir = args.heatmap_dir,
                       heatmap_img_dir = args.heatmap_img_dir,
                       coordinates_csv = args.coordinates_csv)

    runAllBaseCNNs(m_args, args.model_type, cpu=args.cpu)

# EOF
