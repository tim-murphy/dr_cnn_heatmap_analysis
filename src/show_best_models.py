#!/usr/bin/env python

import os
import sys

from model_common import loadModels

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:", sys.argv[0], "<stats_file_csv>")
        sys.exit(1)

    stats_file = sys.argv[1]
    if not os.path.exists(stats_file):
        print("ERROR: stats file does not exist:", stats_file, file=sys.stderr)
        sys.exit(1)

    model_stats = loadModels(stats_file)
    model_stats.sort(reverse=False)

    for model in model_stats:
        print(model)

# EOF
