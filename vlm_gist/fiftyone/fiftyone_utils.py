#!/usr/bin/env python3

# STANDARD

import os
import datetime

# CUSTOM

from nimbro_utils.lazy import SelfShutdown

def import_fiftyone(logger=None):
    if logger is not None:
        logger.info("Importing fiftyone")
    try:
        import fiftyone
    except ImportError:
        raise SelfShutdown("Failed to import module 'fiftyone'.")
    if logger is not None:
        logger.info("Listing loaded datasets")
    for dataset in fiftyone.list_datasets():
        if logger is not None:
            logger.info(f"Removing dataset '{dataset}'")
        fiftyone.load_dataset(dataset).delete()
    return fiftyone

def load_dataset(fiftyone, dataset_path, load_masks=False, crowd_mode=None, logger=None):
    if logger is not None:
        logger.info(f"Loading dataset '{dataset_path}' with{'' if load_masks else 'out'} masks")

    # retrieve dataset type

    for dataset_type in ["FiftyOneDataset", "COCODetectionDataset"]:
        if dataset_type == "FiftyOneDataset":
            if not os.path.isdir(os.path.join(dataset_path, "data")):
                continue
            if not os.path.isfile(os.path.join(dataset_path, "metadata.json")):
                continue
            if not os.path.isfile(os.path.join(dataset_path, "samples.json")):
                continue
            break
        elif dataset_type == "COCODetectionDataset":
            if not os.path.isdir(os.path.join(dataset_path, "test")):
                continue
            if not os.path.isfile(os.path.join(dataset_path, "annotations", "instances_Test.json")):
                continue
            break
    else:
        raise SelfShutdown(f"Failed to retrieve dataset type. Content in '{dataset_path}' does not match any known structure.")

    if logger is not None:
        logger.info(f"Dataset is of type '{dataset_type}'")

    # load dataset

    if dataset_type == "FiftyOneDataset":

        if not os.path.isdir(os.path.join(dataset_path, "data")):
            raise SelfShutdown("Dataset does not contain 'data' folder.")
        if not os.path.isfile(os.path.join(dataset_path, "metadata.json")):
            raise SelfShutdown("Dataset does not contain 'metadata.json' file.")
        if not os.path.isfile(os.path.join(dataset_path, "samples.json")):
            raise SelfShutdown("Dataset does not contain 'samples.json' file.")

        dataset = fiftyone.Dataset.from_dir(
            dataset_dir=dataset_path,
            dataset_type=fiftyone.types.FiftyOneDataset,
            name=os.path.basename(os.path.normpath(dataset_path)),
        )

        # set crowd attribute handling mode
        if crowd_mode is None:
            crowd_mode = "fix"

    elif dataset_type == "COCODetectionDataset":

        image_dir = os.path.join(dataset_path, "test")
        if not os.path.isdir(image_dir):
            raise SelfShutdown(f"Image directory '{image_dir}' does not exist.")

        labels_path = os.path.join(dataset_path, "annotations", "instances_Test.json")
        if not os.path.isfile(labels_path):
            raise SelfShutdown(f"Annotation file '{labels_path}' does not exist.")

        dataset = fiftyone.Dataset.from_dir(
            dataset_type=fiftyone.types.COCODetectionDataset,
            data_path=image_dir,
            labels_path=labels_path,
            name=os.path.basename(os.path.normpath(dataset_path)),
            label_types=["segmentations"] if load_masks else ["detections"]
        )

        # set crowd attribute handling mode
        if crowd_mode is None:
            crowd_mode = "leave"

    else:
        raise NotImplementedError(f"Unknown dataset type '{dataset_type}'")

    # apply crowd attribute handling mode
    if crowd_mode not in ['leave', 'remove', 'fix']:
        raise NotImplementedError(f"Unknown crowd mode '{crowd_mode}'")
    elif crowd_mode == 'leave':
        pass
    else:
        if logger is not None:
            logger.info(f"Handling crowd attribute with mode '{crowd_mode}'")
        for sample in dataset.iter_samples(progress=True):
            for field in ['ground_truth']:
                if field in sample:
                    if not sample[field] is None:
                        any_is_crowd = any(detection.iscrowd for detection in sample[field].detections)
                        if any_is_crowd:
                            processed_detections = []
                            for detection in sample[field].detections:
                                processed_detection = detection.copy()
                                if processed_detection.iscrowd == 1:
                                    if crowd_mode == 'fix':
                                        processed_detection.iscrowd = 0
                                        processed_detections.append(processed_detection)
                                else:
                                    processed_detections.append(processed_detection)
                            sample[field] = fiftyone.Detections(detections=processed_detections)
                            sample.save()

    dataset.info["loaded_dataset_name"] = os.path.basename(dataset_path.rstrip("/"))
    dataset.save()

    return dataset

def get_export_path(export_path, dataset_path=None):
    if dataset_path is not None:
        export_path = export_path.replace("dataset_path", dataset_path)
    export_path = os.path.abspath(export_path)
    export_path = export_path.replace("_stamp", "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3])
    return export_path

def save_dataset(fiftyone, dataset, export_dir, logger=None):
    if logger is not None:
        logger.info(f"Saving dataset to '{export_dir}'")
    dataset.export(export_dir=export_dir, dataset_type=fiftyone.types.FiftyOneDataset)
    if logger is not None:
        logger.info(f"Successfully saved dataset to '{export_dir}'")

def show_dataset(fiftyone, dataset, logger=None, *, address="127.0.0.1", port=None):
    if logger is not None:
        logger.info(f"Launching app: {fiftyone.list_datasets()}")

    # Ensure dataset is persisted + saved so it is always selectable
    if not getattr(dataset, "persistent", False):
        dataset.persistent = True
        dataset.save()

    # Close any stale app and start fresh
    fiftyone.close_app()

    # Launch the App already attached to the dataset
    session = fiftyone.launch_app(
        dataset,
        address=address,
        port=port
    )

    # Tweak color scheme
    cs = session.color_scheme
    cs.color_by = "instance"
    dataset.app_config.color_scheme = cs
    dataset.save()
    session.refresh()

    if logger is not None:
        logger.info(f"Dataset can be viewed at 'http://{session.server_address}:{session.server_port}'")

    return session
