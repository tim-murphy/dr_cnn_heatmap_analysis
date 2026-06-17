# Helper function to convert a string into a cv2 heatmap.
# Written by Tim Murphy <tim.murphy@canberra.edu.au> 2026

import cv2

def string_to_cmap(cmapstr: str):
    c = cmapstr.lower()
    if c == "autumn":
        return cv2.COLORMAP_AUTUMN
    elif c == "bone":
        return cv2.COLORMAP_BONE
    elif c == "jet":
        return cv2.COLORMAP_JET
    if c == "winter":
        return cv2.COLORMAP_WINTER
    elif c == "rainbow":
        return cv2.COLORMAP_RAINBOW
    elif c == "ocean":
        return cv2.COLORMAP_OCEAN
    if c == "summer":
        return cv2.COLORMAP_SUMMER
    elif c == "spring":
        return cv2.COLORMAP_SPRING
    elif c == "cool":
        return cv2.COLORMAP_COOL
    if c == "hsv":
        return cv2.COLORMAP_HSV
    elif c == "pink":
        return cv2.COLORMAP_PINK
    elif c == "hot":
        return cv2.COLORMAP_HOT
    elif c == "parula":
        return cv2.COLORMAP_PARULA
    elif c == "magma":
        return cv2.COLORMAP_MAGMA
    elif c == "inferno":
        return cv2.COLORMAP_INFERNO
    elif c == "plasma":
        return cv2.COLORMAP_PLASMA
    elif c == "viridis":
        return cv2.COLORMAP_VIRIDIS
    elif c == "cividis":
        return cv2.COLORMAP_CIVIDIS
    elif c == "twilight":
        return cv2.COLORMAP_TWILIGHT
    elif c == "twilight_shifted":
        return cv2.COLORMAP_TWILIGHT_SHIFTED
    elif c == "turbo":
        return cv2.COLORMAP_TURBO
    elif c == "deepgreen":
        return cv2.COLORMAP_DEEPGREEN
    else:
        raise ValueError("Invalid cmap:" + cmapstr)

# EOF
