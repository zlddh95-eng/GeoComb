from __future__ import annotations

CLASS_ORDER = [
    "Egg",
    "Larvae",
    "Capped_pupae",
    "Pollen",
    "Capped_honey",
    "Uncapped_honey",
    "Empty_cell",
    "Unbuilt_cell",
]

CLASS_SHORTCUTS = {
    1: "Egg",
    2: "Larvae",
    3: "Capped_pupae",
    4: "Pollen",
    5: "Capped_honey",
    6: "Uncapped_honey",
    7: "Empty_cell",
    8: "Unbuilt_cell",
}

# BGR values to match the existing OpenCV overlay convention.
CLASS_COLORS_BGR = {
    "Egg": (0, 255, 255),
    "Larvae": (255, 255, 255),
    "Capped_pupae": (255, 0, 255),
    "Pollen": (0, 165, 255),
    "Capped_honey": (0, 215, 255),
    "Uncapped_honey": (0, 255, 0),
    "Empty_cell": (255, 0, 0),
    "Unbuilt_cell": (160, 160, 160),
}

CLASS_COLORS_RGB = {
    key: (value[2], value[1], value[0])
    for key, value in CLASS_COLORS_BGR.items()
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}
