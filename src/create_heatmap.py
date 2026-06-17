#!/usr/bin/env python

import csv
import cv2
import math
import numpy as np
import os
import sys

from csv_to_heatmap import csv_to_heatmap

# constants
RIGHT_EYE = 0
LEFT_EYE = 1
BOTH_EYES = 2
SIDE_LABELS = { RIGHT_EYE: "right",\
                LEFT_EYE: "left",\
                BOTH_EYES: "both" }

# set to True to create a CSV file after every image processed. Useful for
# creating animations and stuff.
SAVE_INTERMEDIATE_DATA = False
INTERMEDIATE_DIR = "int_data"
if SAVE_INTERMEDIATE_DATA and not os.path.exists(INTERMEDIATE_DIR):
    os.makedirs(INTERMEDIATE_DIR)

# distance (in pixels) from the optic nerve to the macular in each scaled image
NERVE_MAC_DIST = 250

# Vertical distance (in pixels) from the optic nerve to the macular in each
# scaled image. Used to rotate each image to the same orientation.
MAC_DROP = int(float(NERVE_MAC_DIST) * 0.1)
MAC_ANGLE = math.degrees(math.atan(float(MAC_DROP) / float(NERVE_MAC_DIST)))

# the (x,y) coordinate of the optic nerve on the heatmap canvas
# note: (NERVE_COORD, NERVE_COORD) is the coordinate to use.
# note: this canvas will be necessarily huge because the photos have large
#       borders which cannot be stripped until after processing is finished
NERVE_COORD = 1000

DRAW_QUADS = False
QUAD_BOX_SIZE = (200, 200)

# trim - the resulting image will have large black borders, so cut this
# much off each side (measured in pixels)
SUPERIOR=0
NASAL=1
INFERIOR=2
TEMPORAL=3
TRIM = [450, 600, 450, 300]

# directory structure for images
IMAGE_SUBDIR = "image"
LESION_LABELS_GRADING = { # "actual_1": "Mild NDPR (actual)",\
                          # "actual_2": "Moderate NDPR (actual)",\
                          # "actual_3": "Severe NDPR (actual)",\
                          # "actual_4": "Proliferative DR (actual)",\
                          "estimated_0": "No DR (predicted)",\
                          "estimated_1": "Mild NDPR (predicted)",\
                          "estimated_2": "Moderate NDPR (predicted)",\
                          "estimated_3": "Severe NDPR (predicted)",\
                          "estimated_4": "Proliferative DR (predicted)" }

LESION_LABELS_REFER = { # "actual_0": "No referable DR (actual)",
                        # "actual_1": "Referable DR (actual)",
                        "estimated_0": "No referable DR (predicted)",
                        "estimated_1": "Referable DR (predicted)" }

LESION_LABELS = LESION_LABELS_REFER # FIXME make this selection dynamic

# minimum pixel count to include in the data - used to intensify the heatmap
THRESHOLD = False
LESION_THRESHOLD = { "EX": 5,\
                     "HE": 3,\
                     "MA": 1,\
                     "SE": 2,\
                     "IRMA": 1,\
                     "NVD": 1,\
                     "NVE": 1,\
                     "VB": 1,\
                     "ALL": 10 }

class CoordsData:
    filename = None
    nerve_xy = None
    mac_xy = None

    def __init__(self, filename, nerve_xy, mac_xy):
        self.filename = filename
        self.nerve_xy = nerve_xy
        self.mac_xy = mac_xy

    def showImage(self):
        image = cv2.imread(self.filename)
        cv2.imshow(self.filename, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __repr__(self):
        return "[filename=" + self.filename + \
               ", nerve_xy=(" + str(self.nerve_xy) + ")" + \
               ", mac_xy=(" + str(self.mac_xy) + ")]"

def parseCoordsFile(filename, image_dir):
    coords = list()

    with open(filename) as csvfile:
        coords_data = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(coords_data):
            # ignore the first row - header
            if (index == 0):
                continue

            coords.append(CoordsData(os.path.join(image_dir, IMAGE_SUBDIR, row[0]),
                          (int(row[1]), int(row[2])),
                          (int(row[3]), int(row[4]))))

    return coords

def trimImageArrays(image):
    colour = (True if image.ndim == 5 else False)

    orig_x_len = len(image[0][0])
    orig_y_len = len(image[0][0][0])
    x_len = orig_x_len - TRIM[NASAL] - TRIM[TEMPORAL]
    y_len = orig_y_len - TRIM[SUPERIOR] - TRIM[INFERIOR]

    trimmed = None
    if colour:
        trimmed = np.zeros((3, len(LESION_LABELS) + 1, x_len, y_len, 3), dtype=np.uint32)
    else:
        trimmed = np.zeros((3, len(LESION_LABELS) + 1, x_len, y_len), dtype=np.uint32)

    for i, l in enumerate(LESION_LABELS):
        trimmed[RIGHT_EYE][i] = image[RIGHT_EYE][i][TRIM[SUPERIOR]:orig_x_len-TRIM[INFERIOR],\
                                                    TRIM[TEMPORAL]:orig_y_len-TRIM[NASAL]]
        trimmed[LEFT_EYE][i] = image[LEFT_EYE][i][TRIM[SUPERIOR]:orig_x_len-TRIM[INFERIOR],\
                                                  TRIM[NASAL]:orig_y_len-TRIM[TEMPORAL]]
        trimmed[BOTH_EYES][i] = image[BOTH_EYES][i][TRIM[SUPERIOR]:orig_x_len-TRIM[INFERIOR],\
                                                    TRIM[TEMPORAL]:orig_y_len-TRIM[NASAL]]

        # scale
        for eye in (RIGHT_EYE, LEFT_EYE, BOTH_EYES):
            heatmap_scale = 255.0 / float(max(1, trimmed[eye][i].max()))
            trimmed[eye][i] = trimmed[eye][i] * heatmap_scale

    return trimmed.astype('uint8')

def printUsage():
    print("Usage: " + sys.argv[0] + " <coordinates_csv> <image_dir> [<outdir=heatmaps> [<outfile_suffix>]]")

def addText(image, position, text):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    return image

# Scale the image such that the distance between the nerve
# and macula is NERVE_MAC_DIST. Return the scaled (x,y) position
# of the nerve in the scaled image, as well as the scaling factor
# and rotation required.
# @param image_data CoordsData object
def scaleImage(image_data):
    # Calculate the distace from the nerve to the mac. Ye Olde Pythagoras.
    x = abs(image_data.nerve_xy[0] - image_data.mac_xy[0])
    y = abs(image_data.nerve_xy[1] - image_data.mac_xy[1])
    orig_dist = int(math.sqrt((x*x) + (y*y)))

    if (orig_dist == 0):
        print("ERROR: invalid tagging for image (ignoring): " + image_data.filename)
        return None, None

    # Calculate the angle from the nerve to the mac
    angle = math.degrees(math.atan(float(y) / float(x)))

    # Image rotation required (in degrees)
    rotation = MAC_ANGLE - angle

    # left eyes rotate the opposite direction
    if rightOrLeft(image_data) == LEFT_EYE:
        rotation = angle - MAC_ANGLE

    scaling_factor =  float(NERVE_MAC_DIST) / float(orig_dist)

    new_nerve_xy = (int(float(image_data.nerve_xy[0]) * scaling_factor),
                    int(float(image_data.nerve_xy[1]) * scaling_factor))

    return new_nerve_xy, scaling_factor, rotation

# @param image_data CoordsData object
def rightOrLeft(image_data):
    if (image_data.nerve_xy[0] > image_data.mac_xy[0]):
        return RIGHT_EYE
    return LEFT_EYE

# args is <script_name> <coordinates_csv> <image_dir> [<outdir=heatmaps> [<outfile_suffix=""> [<cmap=cividis>]]]"
# This is nasty business - convert to sane arguments please.
def create_heatmap(args):
    cli_args_valid = True

    coords_csv = os.path.abspath(args[1])
    if (not os.path.exists(coords_csv)):
        print("ERROR: coordinates_csv file \"" + coords_csv + "\" does not exist")
        cli_args_valid = False

    image_dir = os.path.abspath(args[2])
    if (not os.path.exists(image_dir)):
        print("ERROR: image_dir path \"" + image_dir + "\" does not exist")
        cli_args_valid = False

    outdir = "heatmaps"
    if len(args) > 3:
        outdir = args[3]
    os.makedirs(outdir, exist_ok=True)

    out_suffix = ""
    if len(args) > 4 and args[4]: # if args[4] == True, string is not empty
        out_suffix = "_" + args[4]

    cmap = "cividis"
    if len(args) > 5:
        cmap = args[5]

    if (not cli_args_valid):
        sys.exit(1)

    print("Using CSV file \"" + coords_csv + "\"")
    print("with image dir \"" + image_dir + "\"")

    coords_data = parseCoordsFile(coords_csv, image_dir)

    # Initialise the data array (nerve at (NERVE_COORD,NERVE_COORD) which
    # is the middle of our matrix
    # stored as [side][lesion][x][y]
    heatmap_data = np.zeros((3, len(LESION_LABELS) + 1, NERVE_COORD * 2, NERVE_COORD * 2), dtype=np.uint32)

    for r, record in enumerate(coords_data):
        # give the user an idea of how we're tracking
        print("Extracting lesion data [", r+1, "/", len(coords_data), "]\r",
              sep="", end="", flush=True)

        side = rightOrLeft(record)

        # Calculate the scaling factor and scaled nerve position. This determines
        # how to scale the lesion coordinates, and how far to translate them to
        # make sure everything lines up.
        nerve_xy_scaled, scaling_factor, rotation = scaleImage(record)

        if (nerve_xy_scaled == None or scaling_factor == None):
            print("ERROR: ignoring file: " + record.filename)
            continue

        # Load the lesion file(s)
        for index, lesion in enumerate(LESION_LABELS):
            # "ALL" is generated by us from the other lesion types
            if (lesion == "ALL"):
                continue

            lesion_image_path = os.path.join(image_dir, lesion, os.path.split(record.filename)[1])

            # lesion files are .tif, not .png, so we need to replace the extension
            lesion_image_path = os.path.splitext(lesion_image_path)[0] + ".tif"

            if (not os.path.exists(lesion_image_path)):
                # print("ERROR: lesion file does not exist (ignoring): " + lesion_image_path)
                continue

            # load the image...
            lesion_orig = cv2.imread(lesion_image_path, 0)

            # ...scale it...
            lesion_scaled = cv2.resize(lesion_orig, None,
                                       fx=scaling_factor,
                                       fy=scaling_factor)

            # ...rotate it...
            rot_matrix = cv2.getRotationMatrix2D(nerve_xy_scaled, rotation, 1.0)
            img_dims = (len(lesion_scaled[0]), len(lesion_scaled))
            lesion_scaled = cv2.warpAffine(lesion_scaled, rot_matrix, img_dims)

            # ...and mark it in our heatmap matrix
            y_from = NERVE_COORD - nerve_xy_scaled[1]
            y_to = y_from + len(lesion_scaled)
            x_from = NERVE_COORD - nerve_xy_scaled[0]
            x_to = x_from + len(lesion_scaled[0])

            if (x_from < 0 or y_from < 0 or x_to > len(heatmap_data[side][index]) or y_to > len(heatmap_data[side][index][0])):
                print("ERROR:", lesion, "mapping outside of bounds (ignoring):", os.path.basename(record.filename))
                continue

            heatmap_data[side][index][y_from:y_to, x_from:x_to] += lesion_scaled

            # add data to our composite heatmaps as well - represented as right side,
            # so need to mirror left data.
            if (side == LEFT_EYE):
                lesion_scaled = np.fliplr(lesion_scaled)
                x_from = NERVE_COORD - len(lesion_scaled[0]) + nerve_xy_scaled[0]
                x_to = x_from + len(lesion_scaled[0])
            heatmap_data[BOTH_EYES][index][y_from:y_to, x_from:x_to] += lesion_scaled

            # if we are saving progress for each image, do that here
            if SAVE_INTERMEDIATE_DATA:
                frame_number = f'{r:04}'
                trimmed = trimImageArrays(heatmap_data)
                for i, l in enumerate(LESION_LABELS):
                    fname = os.path.join(INTERMEDIATE_DIR, "lesion_count_" + SIDE_LABELS[BOTH_EYES] + "_" + l + out_suffix + "_" + frame_number + ".csv")
                    np.savetxt(fname, trimmed[BOTH_EYES][i], fmt="%i", delimiter=",")

    print()

    # write the heatmap data to file
    trimmed = trimImageArrays(heatmap_data)

    for side in [RIGHT_EYE, LEFT_EYE, BOTH_EYES]:
        for i, l in enumerate(LESION_LABELS):
            print("Generating", ("right", "left", "composite")[side], LESION_LABELS[l], "CSV file")
            np.savetxt(os.path.join(outdir, "lesion_count_" + SIDE_LABELS[side] + "_" + l + out_suffix + ".csv"), trimmed[side][i], fmt="%i", delimiter=",")

    # with a README
    with open(os.path.join(outdir, "README_csv.txt"), "w") as f:
        print("How to interpret the CSV files", file=f)
        print("==============================", file=f)
        print("", file=f)
        print("Each file contains the number of lesions found at each pixel co-ordinate.", file=f)
        print("Note that this uses the screen standard of (0, 0) located at the top left corner.", file=f)
        print("", file=f)
        print("For the right eye and composite images:", file=f)
        print("  Optic nerve position = (", NERVE_COORD - TRIM[TEMPORAL], ",", NERVE_COORD - TRIM[SUPERIOR], ")", file=f)
        print("  Macular position = (", NERVE_COORD - TRIM[TEMPORAL] - NERVE_MAC_DIST, ",", NERVE_COORD - TRIM[SUPERIOR] + MAC_DROP, ")", file=f)
        print("", file=f)
        print("For the left eye:", file=f)
        print("  Optic nerve position = (", NERVE_COORD - TRIM[NASAL], ",", NERVE_COORD - TRIM[SUPERIOR], ")", file=f)
        print("  Macular position = (", NERVE_COORD - TRIM[NASAL] + NERVE_MAC_DIST, ",", NERVE_COORD - TRIM[SUPERIOR] + MAC_DROP, ")", file=f)

    # We now have a giant array with count values. Convert to a uint8 array with
    # normalised values.
    heatmap_image = np.zeros((3, len(LESION_LABELS) + 1, NERVE_COORD * 2, NERVE_COORD * 2), dtype=np.uint8)
    heatmap_coloured = np.zeros((3, len(LESION_LABELS) + 1, NERVE_COORD * 2, NERVE_COORD * 2, 3), dtype=np.uint8)
    for side in [RIGHT_EYE, LEFT_EYE, BOTH_EYES]:
        for index, lesion in enumerate(LESION_LABELS):
            # first do some thresholding
            if THRESHOLD:
                # calculate the threshold. Doubled for the combined image.
                threshold = LESION_THRESHOLD[lesion]
                if (side == BOTH_EYES): threshold *= 2
                heatmap_data[side][index] = np.where(heatmap_data[side][index] < threshold, 0, heatmap_data[side][index])

            print("Generating", ("right", "left", "composite")[side], LESION_LABELS[lesion], "heatmap...", end='')
            heatmap_scale = 255.0 / float(max(1, heatmap_data[side][index].max()))
            heatmap_image[side][index] = heatmap_data[side][index] * heatmap_scale

            # convert to a coloured heatmap
            mac_coord = (NERVE_COORD - ((1, -1, 1)[side] * NERVE_MAC_DIST),
                        NERVE_COORD + MAC_DROP)
            heatmap_coloured[side][index] = csv_to_heatmap(heatmap_image[side][index], cmap,
                                                           annotation_colour_bgr = (255, 255, 255),
                                                           show_annotations = True,
                                                           nerve_coords = (NERVE_COORD, NERVE_COORD),
                                                           macula_coords = mac_coord)

            # also draw in the quads used for stats
            if (DRAW_QUADS):
                cv2.line(heatmap_coloured[side][index],
                         (mac_coord[0] - QUAD_BOX_SIZE[0], mac_coord[1]),
                         (mac_coord[0] + QUAD_BOX_SIZE[0], mac_coord[1]),
                         annotation_colour, 2)
                cv2.line(heatmap_coloured[side][index],
                         (mac_coord[0], mac_coord[1] - QUAD_BOX_SIZE[1]),
                         (mac_coord[0], mac_coord[1] + QUAD_BOX_SIZE[1]),
                         annotation_colour, 2)
                cv2.rectangle(heatmap_coloured[side][index],
                              (mac_coord[0] - QUAD_BOX_SIZE[0], mac_coord[1] - QUAD_BOX_SIZE[1]),
                              (mac_coord[0] + QUAD_BOX_SIZE[0], mac_coord[1] + QUAD_BOX_SIZE[1]),
                              annotation_colour, 2)

            print("done")

    # trim the black edges from the images
    trimmed = trimImageArrays(heatmap_coloured)

    # add some descriptive text
    for i, l in enumerate(LESION_LABELS):
        addText(trimmed[RIGHT_EYE][i], (30,50), "Right Eye - " + LESION_LABELS[l])
        addText(trimmed[LEFT_EYE][i], (30,50), "Left Eye - " + LESION_LABELS[l])
        addText(trimmed[BOTH_EYES][i], (30,50), "Combined - " + LESION_LABELS[l])

    # and we're done! put all the heatmaps together
    stacks = []
    for index, lesion in enumerate(LESION_LABELS):
        s = np.hstack((trimmed[RIGHT_EYE][index],\
                       trimmed[LEFT_EYE][index],\
                       trimmed[BOTH_EYES][index]))
        cv2.imwrite(os.path.join(outdir, "heatmap_" + lesion + out_suffix + ".png"), s)
        stacks.append(s)

    composite = None
    for s in stacks:
        if (composite is None):
            composite = s
        else:
            composite = np.vstack((composite, s))

    cv2.imwrite(os.path.join(outdir, "heatmap" + out_suffix + ".png"), composite)

if __name__ == '__main__':
    # FIXME convert to argparse
    if (len(sys.argv) not in [3,4,5,6]):
        printUsage()
        sys.exit(1)

    create_heatmap(sys.argv)

# EOF
