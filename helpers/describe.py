#!/usr/bin/env python3

import os

datasets = ["./../data/datasets/vlm_gist"]
# datasets = ["./../data/datasets/coco_500"]
# datasets = ["./../data/datasets/vlm_gist", "./../data/datasets/coco_500"]

settings = "./../settings/description/pending"

repeat = 1

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    settings = [os.path.join(settings, setting) for setting in sorted(os.listdir(settings)) if setting[-5:] == ".json"]

    print(f"Datasets: ({len(datasets)})")
    if len(datasets) > 0:
        print("\n".join(datasets))
    else:
        print("None")

    print(f"\nSettings: ({len(settings)})")
    if len(settings) > 0:
        print("\n".join(settings))
    else:
        print("None")

    print(f"\nRepeat: {repeat}")

    args = ""
    for dataset in datasets:
        dataset = os.path.realpath(dataset)
        for setting in settings:
            setting = os.path.realpath(setting)
            for _ in range(repeat):
                args += f' "{dataset}" "{setting}"'

        if args != "":
            cmd = f"ros2 run vlm_gist fiftyone_describe{args}"
            print(f"\nCommand:\n{cmd}")

            ack = input("\nDo you want to execute the command directly (y/n)? ")
            if ack.lower() in ["y", "yes"]:
                os.system(cmd)

