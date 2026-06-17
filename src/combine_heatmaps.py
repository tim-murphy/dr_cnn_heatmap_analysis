#!/usr/bin/env python

# take two heatmap data files and combine them

import csv
import numpy as np
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage:", sys.argv[0], "<output_csv> <input1_csv> <input2_csv> [<input3_csv> [...]]")
        sys.exit(1)

    inputs = []
    for i in range(2, len(sys.argv)):
        inputs.append(sys.argv[i])

    outfile = sys.argv[1]

    # sanity checks
    args_ok = True
    for i in inputs:
        if not os.path.isfile(i):
            print("ERROR: input file does not exist:", i, file=sys.stdout)
            args_ok = False

    if not args_ok:
        sys.exit(1)

    # do the joining!
    combined = None
    for i in inputs:
        print("Loading", i)
        with open(i, 'r') as csvfile:
            raw_data = np.array(list(csv.reader(csvfile, delimiter=',')), dtype=np.uint32)

            if combined is None:
                combined = raw_data
            else:
                combined += raw_data

    print("Writing to", outfile)
    np.savetxt(outfile, combined.astype(np.uint32), fmt="%i", delimiter=",")

    print("Done! Have a nice day :)")

# EOF
