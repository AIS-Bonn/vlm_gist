#!/usr/bin/env python3

import os
from utility import get_paths

base = "./../data/new"
settings = "./../settings/detection/detection_settings_mmgd_large.json"

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Datasets:")
    datasets = get_paths(base, "detect", verbose=False)
    if len(datasets) == 0:
        print("None")

    args = ""
    for dataset in datasets:
        args += f' "{os.path.realpath(dataset)}" "{os.path.realpath(settings)}"'

    if args != "":
        cmd = f"ros2 run vlm_gist fiftyone_detect{args}"
        print(f"\nCommand:\n{cmd}")

        ack = input("\nDo you want to execute the command directly (y/n)? ")
        if ack.lower() in ["y", "yes"]:
            os.system(cmd)

