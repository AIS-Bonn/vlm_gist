#!/usr/bin/env python3

# STANDARD

import os
import sys
import json
import shutil
import traceback

import numpy as np

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.lazy import Detection
from vlm_gist.fiftyone.fiftyone_utils import import_fiftyone, load_dataset, get_export_path, save_dataset

from nimbro_utils.lazy import start_and_spin_node, SelfShutdown, Logger, ParameterHandler, read_json, escape

### <Parameter Defaults>

severity = 20
save_mask_with_detections = True
show_description_as_label = True
export_path = "dataset_path/../fo_stamp_detect"

### </Parameter Defaults>

class FiftyOneDetect(Node):

    def __init__(self, name="fiftyone_detect", *, context=None, dataset_paths, settings_paths, **kwargs):
        super().__init__(name, context=context, **kwargs)
        self._logger = Logger(self)

        self.dataset_paths = []
        for path in dataset_paths:
            path = os.path.abspath(path)
            if os.path.isdir(path):
                self.dataset_paths.append(path)
            else:
                message = f"Dataset '{path}' does not exist."
                self._logger.error(message)
                raise SelfShutdown(message)

        if settings_paths is None:
            self.settings_paths = None
        else:
            self.settings_paths = []
            for path in settings_paths:
                path = os.path.abspath(path)
                if os.path.isfile(path):
                    success, message, settings = read_json(file_path=path, logger=self._logger)
                    if success:
                        self.settings_paths.append(path)
                    else:
                        raise SelfShutdown(message)
                else:
                    message = f"Settings file '{path}' does not exist."
                    self._logger.error(message)
                    raise SelfShutdown(message)

        # Declare parameters

        self.parameter_handler = ParameterHandler(self)

        self.parameter_handler.declare(
            name="severity",
            dtype=int,
            default_value=severity,
            description="Logging severity of node logger.",
            read_only=False,
            range_min=10,
            range_max=50,
            range_step=10
        )

        self.parameter_handler.declare(
            name="save_mask_with_detections",
            dtype=bool,
            default_value=save_mask_with_detections,
            description="Save masks with the exported dataset.",
            read_only=True
        )

        self.parameter_handler.declare(
            name="show_description_as_label",
            dtype=bool,
            default_value=show_description_as_label,
            description="Show descriptions as labels instead of object names.",
            read_only=True
        )

        self.parameter_handler.declare(
            name="export_path",
            dtype=str,
            default_value=export_path,
            description="Path to save new dataset.",
            read_only=True
        )

        self.parameter_handler.deactivate_declarations()

        self.detection = Detection(
            node=self,
            metadata_timer=False,
            logger_severity=self.parameters.severity
        )

        self.timer_state = self.create_timer(0.0, self.state_machine, callback_group=MutuallyExclusiveCallbackGroup())

        self._logger.info("Node started")

    def __del__(self):
        self._logger.info("Node shutdown")

    def filter_parameter(self, name, value, is_declared):
        message = None

        if name == "severity":
            self.get_logger().set_settings(settings={'severity': value})

        return value, message

    def state_machine(self):
        self.timer_state.cancel()
        errors, export_paths = {}, []
        for i, path in enumerate(self.dataset_paths):
            self._logger.info(f"Handling dataset '{i + 1}' of '{len(self.dataset_paths)}': '{path}'")
            try:
                if self.settings_paths is not None:
                    success, message, settings = read_json(file_path=self.settings_paths[i], logger=self._logger)
                    if success:
                        success, message, _ = self.detection.set_settings(settings=settings)
                        if not success:
                            raise SelfShutdown(message)
                    else:
                        raise SelfShutdown(message)
                success, message, _ = self.detection.set_settings(
                    settings={
                        'segmentation_model_id_secondary': None,
                        'retry': True,
                        'leave_images_in_place': True,
                        'keep_image_name': True,
                        'png_max_pixels': None,
                        'crop_masks': True,
                        'data_path': path,
                        'image_folder': "detection_edits",
                        'metadata_file': "detections.json",
                        'metadata_write_no_paths': True
                    }
                )
                if not success:
                    raise SelfShutdown(message)

                fiftyone = import_fiftyone(logger=self._logger)
                dataset = load_dataset(fiftyone=fiftyone, dataset_path=path, logger=self._logger)
                dataset = self.detect_dataset(fiftyone=fiftyone, dataset=dataset)
                export_path = get_export_path(export_path=self.parameters.export_path, dataset_path=path)
                save_dataset(fiftyone=fiftyone, dataset=dataset, export_dir=export_path, logger=self._logger)

                # move detections file
                self.detection.save_metadata()
                detection_source = os.path.join(path, self.detection._settings['metadata_file'])
                detection_target = os.path.join(export_path, self.detection._settings['metadata_file'])
                self._logger.info(f"Moving detections metadata file '{detection_source}' to dataset folder '{detection_target}'")
                shutil.move(detection_source, detection_target)

                # move edited detection images
                images_source = os.path.join(path, self.detection._settings['image_folder'])
                if os.path.isdir(images_source):
                    images_target = os.path.join(export_path, self.detection._settings['image_folder'])
                    self._logger.info(f"Moving edited detection images '{images_source}' to '{images_target}'")
                    shutil.move(images_source, images_target)

                # copy description file
                description_source = os.path.join(path, "descriptions.json") # TODO retrieve name
                description_target = os.path.join(export_path, "descriptions.json") # TODO retrieve name
                self._logger.info(f"Copying descriptions metadata file '{description_source}' to dataset folder '{description_target}'")
                shutil.copy(description_source, description_target)

                # copy edited description images
                images_source = os.path.join(path, "description_edits") # TODO retrieve name
                if os.path.isdir(images_source):
                    images_target = os.path.join(export_path, "description_edits") # TODO retrieve name
                    self._logger.info(f"Copying edited description images '{images_source}' to '{images_target}'")
                    shutil.copytree(images_source, images_target)

                export_paths.append(export_path)
                self._logger.info(f"Validate dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_validate -- {export_path}{escape['end']}")
                self._logger.info(f"Label match dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_label_match -- {export_path}{escape['end']}")
                self._logger.info(f"Evaluate dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_eval -- {export_path}{escape['end']}")
                self._logger.info(f"Visualize dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_show -- {export_path}{escape['end']}")

                dataset.delete()
            except SelfShutdown as e:
                self._logger.error(f"{e}")
            except Exception as e:
                self._logger.error(f"{repr(e)}\n{traceback.format_exc()}")

        if len(self.dataset_paths) > 1:
            self._logger.info("Summary:")
            if len(errors) > 0:
                for path in errors:
                    self._logger.error(f"Error while handling '{path}' ({i}): {errors[path]}")
            if len(export_paths) > 0:
                self._logger.info(f"Successfully exported datasets: {" ".join([f"'{path}'" for path in export_paths])}")
                self._logger.info(f"Evaluate all successful datasets: {escape['cyan']}ros2 run vlm_gist fiftyone_eval -- {" ".join([f"'{path}'" for path in export_paths])}{escape['end']}")

        self._logger.info("Node stopped")

    def detect_dataset(self, fiftyone, dataset):
        # obtain detector prompts
        self._logger.info(f"Parsing structured description of '{len(dataset)}' images")
        key_description = 'description' # TODO retrieve
        key_object_name = 'object_name' # TODO retrieve
        file_paths = dataset.values("filepath")
        file_prompts = []

        structured_descriptions = [sample.structured_description if sample.has_field("structured_description") else None for sample in dataset]
        if all(structured_description is None for structured_description in structured_descriptions):
            self._logger.warn("The dataset does not contain any structured descriptions.")
            mute_missing_descriptions = True
        else:
            mute_missing_descriptions = False
        warnings = []
        for i, _ in enumerate(dataset.iter_samples(progress=True)):
            prompts = []
            if structured_descriptions[i] is None and not mute_missing_descriptions:
                warnings.append(f"Image '{file_paths[i]}' does not contain a structured description.")
            else:
                try:
                    structured_descriptions[i] = json.loads(structured_descriptions[i])
                    prompts = [structured_descriptions[i][j][key_description] for j in range(len(structured_descriptions[i]))]
                    if len(prompts) == 0:
                        raise ValueError("Structured description does not contain any object instances.")
                except Exception as e:
                    warnings.append(f"Failed to parse structured description of image '{file_paths[i]}': {e}")
                    structured_descriptions[i] = None
                    prompts = []
            file_prompts.append(prompts)
        for warning in warnings:
            self._logger.warn(warning)

        # detect
        success, message, labels, bboxes, confidences, masks, track_ids, metadata = self.detection.get(
            images=file_paths,
            prompts=file_prompts
        )
        if not success:
            raise SelfShutdown(message)

        # collect full dictionaries for each detection
        self._logger.info("Collecting instance descriptions of all detections")
        instance_descriptions = []
        for i, _ in enumerate(dataset.iter_samples(progress=True)):
            image_descriptions = []
            for label in labels[i]:
                if structured_descriptions[i] is not None:
                    for instance in structured_descriptions[i]:
                        if instance[key_description] == label:
                            image_descriptions.append(instance)
                            break
                    else:
                        raise SelfShutdown(f"Failed to retrieve instance description of detector prompt '{label}' of image '{file_paths[i]}'.")
            instance_descriptions.append(image_descriptions)

        # write to dataset
        self._logger.info("Writing detections to dataset")
        for i, sample in enumerate(dataset.iter_samples(progress=True)):
            image_detections = []
            detections_direct = []
            detections_over = []
            for j in range(len(labels[i])):
                # format bounding box
                width = sample.metadata.width
                height = sample.metadata.height
                bbox = bboxes[i][j]
                bbox = [bbox[0] / width, bbox[1] / height, (bbox[2] - bbox[0]) / width, (bbox[3] - bbox[1]) / height]

                # collect
                image_detections.append(fiftyone.core.labels.Detection(
                    label=labels[i][j] if self.parameters.show_description_as_label else instance_descriptions[i][j][key_object_name],
                    bounding_box=bbox,
                    confidence=confidences[i][j],
                    mask=np.array(masks[i][j]) if self.parameters.save_mask_with_detections else None,
                    attributes={k: fiftyone.core.labels.Attribute(value=v) for k, v in instance_descriptions[i][j].items()}
                ))
                if metadata[i]['is_over_detection'][j]:
                    detections_over.append(image_detections[-1])
                else:
                    detections_direct.append(image_detections[-1])

            if len(image_detections) > 0:
                sample["detections"] = fiftyone.Detections(detections=image_detections)
            if len(detections_direct) > 0:
                sample["detections_direct"] = fiftyone.Detections(detections=detections_direct)
            if len(detections_over) > 0:
                sample["detections_over"] = fiftyone.Detections(detections=detections_over)
            sample["num_detections"] = len(labels[i])
            if len(file_prompts[i]) > 0:
                sample["num_detections_unique"] = len(set(labels[i]))
                sample["num_detections_missing"] = metadata[i]['num_missing_prompts']
                sample["num_detections_over"] = metadata[i]['num_over_detections']
            sample["detection_info"] = json.dumps(metadata[i], indent=4)
            sample.save()

        return dataset

def main():
    if len(sys.argv) < 2 or (len(sys.argv) > 2 and len(sys.argv) % 2 == 0):
        print("Usage: ros2 run vlm_gist fiftyone_detect -- <dataset_path>\n   or: ros2 run vlm_gist fiftyone_detect -- <dataset_path> <settings_path>\n   or: ros2 run vlm_gist fiftyone_detect -- <dataset_path_1> <settings_path_1> ...")
        return

    dataset_paths = []
    if len(sys.argv) == 2:
        dataset_paths.append(sys.argv[1])
        settings_paths = None
    else:
        settings_paths = []
        for i in range(1, len(sys.argv), 2):
            dataset_paths.append(sys.argv[i])
            settings_paths.append(sys.argv[i + 1])

    start_and_spin_node(FiftyOneDetect, node_args={'dataset_paths': dataset_paths, 'settings_paths': settings_paths})

if __name__ == '__main__':
    main()
