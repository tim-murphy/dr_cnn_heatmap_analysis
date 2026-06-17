
#!/usr/bin/env python

# Create a heatmap collage from a bunch of individual heatmaps.
# Written by Tim Murphy <tim.murphy@canberra.edu.au> 2026

import argparse
import cv2
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap_dir", type=str, required=True,
                        help="Directory containing heatmap images.")
    parser.add_argument("--outfile_png", type=str, required=False,
                        help="If set, will write the collated image to this path.")
    parser.add_argument("--heatmap_width_px", type=int, default=1100,
                        help="Width of input heatmaps (default 1100).")
    parser.add_argument("--heatmap_height_px", type=int, default=1100,
                        help="Height of input heatmaps (default: 1100)")
    parser.add_argument("--canvas_width_n", type=int, default=4,
                        help="Canvas width in number of heatmaps.")
    parser.add_argument("--canvas_height_n", type=int, default=7,
                        help="Canvas height in number of heatmaps.")
    args = parser.parse_args()

    # Validate command line arguments.
    if not os.path.isdir(args.heatmap_dir):
        raise ValueError("Heatmap directory does not exist: " + args.heatmap_dir)
        
    if args.heatmap_width_px < 1 or args.heatmap_height_px < 1 or\
       args.canvas_width_n < 1 or args.canvas_width_n < 1:
        raise ValueError("Invalid canvas parameters: widths and heights must be > 0.")

    COLLAGE = [
        ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"],
        ["convnext_xlarge", "densenet121", "densenet169", "densenet201"],
        ["efficientnetv2-b0", "efficientnetv2-b1", "efficientnetv2-b2", "efficientnetv2-b3"],
        ["inception_resnet_v2", "inception_v3", "nasnet_large", "nasnet_mobile"],
        ["resnet50", "resnet101", "resnet152", "vgg16"],
        ["resnet50v2", "resnet101v2", "resnet152v2", "vgg19"],
        ["_mobilenet", "mobilenetv2", "xception", "combined"]
    ]

    width = args.heatmap_width_px * args.canvas_width_n
    height = args.heatmap_height_px * args.canvas_height_n
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # The next ensures the walk is non-recursive.
    _, _, heatmap_png = next(os.walk(args.heatmap_dir))

    for y, rows in enumerate(COLLAGE):
        for x, cell_img in enumerate(rows):
            # Find the image which belongs in this cell.
            for png in heatmap_png:
                if cell_img in png:
                    # Found!
                    print("Adding ", png, " at (", x, ",", y, ")", sep="")

                    x_min = (x * args.heatmap_width_px)
                    x_max = (x_min + args.heatmap_width_px)
                    y_min = (y * args.heatmap_height_px)
                    y_max = (y_min + args.heatmap_height_px)

                    png_path = os.path.join(args.heatmap_dir, png)
                    canvas[y_min:y_max, x_min:x_max] = cv2.imread(png_path)
                    break

    if args.outfile_png is None:
        canvas_small = cv2.resize(canvas, (int(width / 10), int(height / 10))) 
        cv2.imshow('Collated heatmaps', canvas_small)
        cv2.waitKey(0) # Wait for a key press to close the window
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(args.outfile_png, canvas)
        print("Image written to", args.outfile_png)

    print()
    print("All done! Have a nice day :)")
# EOF