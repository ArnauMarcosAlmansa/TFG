import os
import shutil

filters = [
    "NOFILTER",
    "BP850-27",
    "BP635-27",
    "BP590-27",
    "BP525-27",
    "BP505-27",
    "BP470-27",
    "BP324-27",
    "BP550-27",
]


source_path = "/home/amarcos/workspace/TFG/scripts/MSNerf/"
destination_path = "/home/amarcos/workspace/TFG/scripts/MSNerfCorrected/"
alturas_dirs = sorted(os.listdir(source_path))
os.makedirs(destination_path, exist_ok=True)


for altura_dir in alturas_dirs:
    angle_dirs = sorted(os.listdir(source_path + altura_dir))
    os.makedirs(destination_path + altura_dir, exist_ok=True)
    for angle_dir in angle_dirs:
        os.makedirs(destination_path + altura_dir + "/" + angle_dir, exist_ok=True)
        photos = sorted(os.listdir(source_path + altura_dir + "/" + angle_dir))
        for i, (photo, filter) in enumerate(zip(photos, filters)):
            shutil.copy2(
                source_path + altura_dir + f"/{angle_dir}/{photo}",
                destination_path + altura_dir + f"/{angle_dir}/{i}-{filter}.tiff"
            )
