#!/usr/bin/env python3

import os
from utility import get_paths

base = "./../data/new"

settings_custom = "./../settings/label_matching/label_matching_settings_per_image_openai.json"
settings_coco = "./../settings/label_matching/label_matching_settings_coco_openai.json"

# settings_custom = "./../settings/label_matching/label_matching_settings_per_image_openrouter.json"
# settings_coco = "./../settings/label_matching/label_matching_settings_coco_openrouter.json"

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Datasets:")
    datasets = get_paths(base, "match", verbose=False)
    if len(datasets) == 0:
        print("None")

    args = ""
    for dataset in sorted(datasets):
        assert "/custom/" in dataset or "/coco/" in dataset
        assert not ("/custom/" in dataset and "/coco/" in dataset)
        if "/custom/" in dataset:
            args += f' "{os.path.realpath(dataset)}" "{os.path.realpath(settings_custom)}"'
        elif "/coco/" in dataset:
            args += f' "{os.path.realpath(dataset)}" "{os.path.realpath(settings_coco)}"'

    if args != "":
        cmd = f"ros2 run vlm_gist fiftyone_label_match{args}"
        print(f"\nCommand:\n{cmd}")

        ack = input("\nDo you want to execute the command directly (y/n)? ")
        if ack.lower() in ["y", "yes"]:
            os.system(cmd)
