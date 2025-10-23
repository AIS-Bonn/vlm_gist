#!/usr/bin/env python3

# STANDARD

import os
import sys
import json
import random
import shutil
import traceback

import numpy as np

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.lazy import Validation
from vlm_gist.fiftyone.fiftyone_utils import import_fiftyone, load_dataset, get_export_path, save_dataset

from nimbro_utils.lazy import start_and_spin_node, SelfShutdown, Logger, ParameterHandler, read_json, escape

### <Parameter Defaults>

severity = 20
parallel_completions = 0
keep_detection_results = True
save_mask_with_detections = True
show_description_as_label = True
export_path = "dataset_path/../fo_stamp_validate"

### </Parameter Defaults>

class FiftyOneValidate(Node):

    def __init__(self, name="fiftyone_validate", *, context=None, dataset_paths, settings_paths, **kwargs):
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
            name="parallel_completions",
            dtype=int,
            default_value=parallel_completions,
            description="Number of completions used in parallel. Set 0 to use CPU count.",
            read_only=True,
            range_min=0,
            range_max=1000,
            range_step=1
        )

        self.parameter_handler.declare(
            name="keep_detection_results",
            dtype=bool,
            default_value=keep_detection_results,
            description="Backup detections before validation.",
            read_only=True
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

        self.validation = Validation(
            node=self,
            parallel_completions=self.parameters.parallel_completions if self.parameters.parallel_completions > 0 else None,
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
            self._logger.info(f"Handling experiment '{i + 1}' of '{len(self.dataset_paths)}': '{path}' / '{self.settings_paths[i]}'")
            try:
                if self.settings_paths is not None:
                    success, message, settings = read_json(file_path=self.settings_paths[i], logger=self._logger)
                    if success:
                        success, message, _ = self.validation.set_settings(settings=settings)
                        if not success:
                            raise SelfShutdown(message)
                    else:
                        raise SelfShutdown(message)
                success, message, _ = self.validation.set_settings(
                    settings={
                        'retry': True,
                        'timeout': None,
                        'leave_images_in_place': True,
                        'keep_image_name': True,
                        'png_max_pixels': None,
                        'crop_masks': True,
                        'usage_skip': False,
                        'usage_delay': 1.0,
                        'data_path': path,
                        'image_folder': "validation_edits",
                        'crop_folder': "validation_crops",
                        'metadata_file': "validations.json",
                        'metadata_write_no_paths': True
                    }
                )
                if not success:
                    raise SelfShutdown(message)

                fiftyone = import_fiftyone(logger=self._logger)
                dataset = load_dataset(fiftyone=fiftyone, dataset_path=path, logger=self._logger)
                dataset = self.validate_dataset(fiftyone=fiftyone, dataset=dataset)
                export_path = get_export_path(export_path=self.parameters.export_path, dataset_path=path)
                save_dataset(fiftyone=fiftyone, dataset=dataset, export_dir=export_path, logger=self._logger)

                # move validations file
                self.validation.save_metadata()
                validation_source = os.path.join(path, self.validation._settings['metadata_file'])
                validation_target = os.path.join(export_path, self.validation._settings['metadata_file'])
                self._logger.info(f"Moving validations metadata file '{validation_source}' to dataset folder '{validation_target}'")
                shutil.move(validation_source, validation_target)

                # move edited validation images
                images_source = os.path.join(path, self.validation._settings['image_folder'])
                if os.path.isdir(images_source):
                    images_target = os.path.join(export_path, self.validation._settings['image_folder'])
                    self._logger.info(f"Moving edited validation images '{images_source}' to '{images_target}'")
                    shutil.move(images_source, images_target)

                # move validation crops
                images_source = os.path.join(path, self.validation._settings['crop_folder'])
                if os.path.isdir(images_source):
                    images_target = os.path.join(export_path, self.validation._settings['crop_folder'])
                    self._logger.info(f"Moving validation crops '{images_source}' to '{images_target}'")
                    shutil.move(images_source, images_target)

                # copy detections file
                detection_source = os.path.join(path, "detections.json")
                detection_target = os.path.join(export_path, "detections.json")
                self._logger.info(f"Copying detections metadata file '{detection_source}' to dataset folder '{detection_target}'")
                shutil.copy(detection_source, detection_target)

                # copy edited detection images
                images_source = os.path.join(path, "detection_edits")
                if os.path.isdir(images_source):
                    images_target = os.path.join(export_path, "detection_edits")
                    self._logger.info(f"Copying edited detection images '{images_source}' to '{images_target}'")
                    shutil.copytree(images_source, images_target)

                # copy description file
                description_source = os.path.join(path, "descriptions.json")
                description_target = os.path.join(export_path, "descriptions.json")
                self._logger.info(f"Copying descriptions metadata file '{description_source}' to dataset folder '{description_target}'")
                shutil.copy(description_source, description_target)

                # copy label matching file
                matches_source = os.path.join(path, self.label_matching._settings['metadata_file'])
                if os.path.isfile(matches_source):
                    matches_target = os.path.join(export_path, self.label_matching._settings['metadata_file'])
                    self._logger.info(f"Copying label matching metadata file '{matches_source}' to dataset folder '{matches_target}'")
                    shutil.copy(matches_source, matches_target)

                # copy edited description images
                images_source = os.path.join(path, "description_edits")
                if os.path.isdir(images_source):
                    images_target = os.path.join(export_path, "description_edits")
                    self._logger.info(f"Copying edited description images '{images_source}' to '{images_target}'")
                    shutil.copytree(images_source, images_target)

                export_paths.append(export_path)
                self._logger.info(f"Label match dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_label_match -- {export_path}{escape['end']}")
                self._logger.info(f"Evaluate dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_eval -- {export_path}{escape['end']}")
                self._logger.info(f"Visualize dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_show -- {export_path}{escape['end']}")

                dataset.delete()
            except SelfShutdown as e:
                self._logger.error(f"{e}")
            except Exception as e:
                errors[path] = f"{repr(e)}\n{traceback.format_exc()}"
                self._logger.error(errors[path])

        if len(self.dataset_paths) > 1:
            self._logger.info("Summary:")
            if len(errors) > 0:
                for path in errors:
                    self._logger.error(f"Error while handling experiment '{i}' ('{path}' / '{self.settings_paths[i]}'): {errors[path]}")
            if len(export_paths) > 0:
                self._logger.info(f"Successfully exported datasets: {" ".join([f"'{path}'" for path in export_paths])}")
                self._logger.info(f"Evaluate all successful datasets: {escape['cyan']}ros2 run vlm_gist fiftyone_eval -- {" ".join([f"'{path}'" for path in export_paths])}{escape['end']}")

        self._logger.info("Node stopped")

    def validate_dataset(self, fiftyone, dataset):
        # collect validation inputs
        self._logger.info(f"Collecting validation input of '{len(dataset)}' images")
        key_description = 'description' # TODO retrieve
        key_object_name = 'object_name' # TODO retrieve
        detections = dataset.values("detections.detections", missing_value=None)
        if all(detections_image is None for detections_image in detections):
            raise SelfShutdown("The dataset does not contain any detections")
        structured_descriptions = dataset.values("structured_description", missing_value=None)
        if all(structured_description is None for structured_description in structured_descriptions):
            raise SelfShutdown("The dataset does not contain any structured descriptions")
        file_paths = dataset.values("filepath")
        file_mask = [None for _ in range(len(file_paths))]
        validate_paths, validate_structured_descriptions, labels, bboxes, confidences, masks = [], [], [], [], [], []
        for i, sample in enumerate(dataset.iter_samples(progress=True)):
            image_detections = detections[i]
            if image_detections is None:
                self._logger.warn(f"Image '{file_paths[i]}' does not contain any detections")
            elif structured_descriptions[i] is None:
                self._logger.error(f"Image '{file_paths[i]}' does not contain a structured descriptions even through it contains detections")
            else:
                try:
                    structured_descriptions[i] = json.loads(structured_descriptions[i])
                    if len(structured_descriptions[i]) == 0:
                        raise ValueError("Structured description does not contain any instances.")
                except Exception as e:
                    self._logger.warn(f"Failed to parse structured description of image '{file_paths[i]}': {repr(e)}")
                    structured_descriptions[i] = None
                else:
                    # this assumes that object_name is unique while choosing at random if description is not, i.e. the mapping from description to object_name is ambiguous
                    name_to_description = {item[key_object_name]: item[key_description] for item in structured_descriptions[i]}
                    description_to_names = {}
                    for name in name_to_description:
                        if name_to_description[name] in description_to_names:
                            description_to_names[name_to_description[name]].append(name)
                        else:
                            description_to_names[name_to_description[name]] = [name]

                    assert len(image_detections) > 0
                    labels_image, bboxes_image, confidences_image, masks_image = [], [], [], []
                    for detection in image_detections:
                        if detection['label'] not in description_to_names:
                            self._logger.error(f"Image '{file_paths[i]}' contains detection with label '{detection['label']}' that is not contained the corresponding structured description")
                            break

                        labels_image.append(description_to_names[detection['label']][random.randrange(len(description_to_names[detection['label']]))])

                        # format bounding box
                        width = sample.metadata.width
                        height = sample.metadata.height
                        bbox = detection['bounding_box']
                        x1 = int(round(bbox[0] * width))
                        y1 = int(round(bbox[1] * height))
                        x2 = int(round(x1 + bbox[2] * width))
                        y2 = int(round(y1 + bbox[3] * height))
                        bbox = [x1, y1, x2, y2]

                        bboxes_image.append(bbox)
                        confidences_image.append(detection['confidence'])
                        masks_image.append(detection['mask'])
                    else:
                        file_mask[i] = len(validate_paths)
                        validate_paths.append(file_paths[i])
                        validate_structured_descriptions.append(structured_descriptions[i])
                        labels.append(labels_image)
                        bboxes.append(bboxes_image)
                        confidences.append(confidences_image)
                        masks.append(masks_image)
        if len(validate_paths) == 0:
            raise SelfShutdown("All samples contained in the dataset lack either a structured description or a detection")

        # validate
        self._logger.info(f"Validating detections of '{len(validate_paths)}' images")
        success, message, labels, bboxes, confidences, masks, track_ids, metadata = self.validation.get(
            images=validate_paths,
            structured_descriptions=validate_structured_descriptions,
            identifiers=key_object_name,
            labels=labels,
            bboxes=bboxes,
            confidences=confidences,
            masks=masks
        )
        self.validation.release_completions()
        if not success:
            raise SelfShutdown(message)

        # collect full dictionaries for each validated detection
        self._logger.info("Collecting instance descriptions of all validated detections")
        instance_descriptions = []
        for i, _ in enumerate(dataset.iter_samples(progress=True)):
            if file_mask[i] is not None:
                self._logger.info(f"structured_descriptions[i]: {structured_descriptions[i]}")
                image_descriptions = []
                for label in labels[file_mask[i]]:
                    for instance in structured_descriptions[i]:
                        if instance[key_object_name] == label:
                            image_descriptions.append(instance)
                            break
                    else:
                        raise SelfShutdown(f"Failed to retrieve instance description of object '{label}' of image '{file_paths[i]}'")
                instance_descriptions.append(image_descriptions)

        # write to dataset
        self._logger.info("Writing validated detections to dataset")

        if self.parameters.keep_detection_results:
            dataset.rename_sample_field("detections", "detections_before_validation")
        else:
            dataset.delete_sample_field("detections")
        if "detections_direct" in dataset.get_field_schema():
            dataset.delete_sample_field("detections_direct")
        if "detections_over" in dataset.get_field_schema():
            dataset.delete_sample_field("detections_over")

        for i, sample in enumerate(dataset.iter_samples(progress=True)):
            if file_mask[i] is not None:
                image_detections = []
                for j in range(len(labels[file_mask[i]])):
                    # format bounding box
                    width = sample.metadata.width
                    height = sample.metadata.height
                    bbox = bboxes[file_mask[i]][j]
                    bbox = [bbox[0] / width, bbox[1] / height, (bbox[2] - bbox[0]) / width, (bbox[3] - bbox[1]) / height]

                    # collect
                    image_detections.append(fiftyone.core.labels.Detection(
                        label=instance_descriptions[file_mask[i]][j][key_description] if self.parameters.show_description_as_label else labels[file_mask[i]][j],
                        bounding_box=bbox,
                        confidence=confidences[file_mask[i]][j],
                        mask=np.array(masks[file_mask[i]][j]) if self.parameters.save_mask_with_detections else None,
                        attributes={k: fiftyone.core.labels.Attribute(value=v) for k, v in instance_descriptions[file_mask[i]][j].items()}
                    ))
                sample["detections"] = fiftyone.Detections(detections=image_detections)
                sample["num_detections_valid"] = len(labels[file_mask[i]])
                sample["num_detections_valid_unique"] = len(set(labels[file_mask[i]]))
                sample["num_validation_keep"] = len(metadata[file_mask[i]]['summary']['keep'])
                sample["num_validation_update"] = len(metadata[file_mask[i]]['summary']['update'])
                sample["num_validation_reject"] = len(metadata[file_mask[i]]['summary']['reject'])
                sample["num_validation_error"] = len(metadata[file_mask[i]]['summary']['error'])
                sample["validation_info"] = json.dumps(metadata[file_mask[i]], indent=4)
                sample.save()

        return dataset

def main():
    if len(sys.argv) < 2 or (len(sys.argv) > 2 and len(sys.argv) % 2 == 0):
        print("Usage: ros2 run vlm_gist fiftyone_validate -- <dataset_path>\n   or: ros2 run vlm_gist fiftyone_validate -- <dataset_path> <settings_path>\n   or: ros2 run vlm_gist fiftyone_validate -- <dataset_path_1> <settings_path_1> ...")
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

    start_and_spin_node(FiftyOneValidate, node_args={'dataset_paths': dataset_paths, 'settings_paths': settings_paths})

if __name__ == '__main__':
    main()
