# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:43:22 2025

@author: Uporabnik
"""

import random

# Define the new texture list
new_textures = [
    "BarrelTexture_459","BarrelTexture_460","BarrelTexture_461","BarrelTexture_462","BarrelTexture_463","BarrelTexture_464","BarrelTexture_465","BarrelTexture_466","BarrelTexture_467","BarrelTexture_468","BarrelTexture_469","BarrelTexture_470",
"BarrelTexture_471","BarrelTexture_472","BarrelTexture_473","BarrelTexture_474","BarrelTexture_475","BarrelTexture_476","BarrelTexture_477","BarrelTexture_478","BarrelTexture_479","BarrelTexture_480","BarrelTexture_481","BarrelTexture_482",
"BarrelTexture_483","BarrelTexture_484","BarrelTexture_485","BarrelTexture_486"
]

# Generate objects
new_result_strings = []

for texture in new_textures:
    variable = texture.replace("Texture", "")
    random_value = round(random.uniform(0.80, 1.20), 2)
    new_result_strings.append(f"""{variable}: {{
        name: "{variable}",
        category: "chest",
        element: "BARREL",
        scale: {random_value} / (2 ** 1),
        glueToFloor: true,
        texture: "{texture}",
        material: MATERIAL.standardShine,
        rotateToNorth: 0
    }}""")

new_concatenated_result = ",\n".join(new_result_strings)
new_concatenated_result
