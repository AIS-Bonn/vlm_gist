import os

datasets = ["./../data/datasets/vlm_gist", "./../data/datasets/coco_500"]

settings = "./description/pending/"
settings = [os.path.join(settings, setting) for setting in sorted(os.listdir(settings)) if setting[-5:] == ".json"]

cmd = ""
for dataset in datasets:
    dataset = os.path.abspath(dataset)
    for setting in settings:
        setting = os.path.abspath(setting)
        cmd += f' "{dataset}" "{setting}"'
cmd = cmd.lstrip()

print("ros2 run vlm_gist fiftyone_describe", cmd)
