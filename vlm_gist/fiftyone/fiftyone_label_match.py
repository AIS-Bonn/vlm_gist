#!/usr/bin/env python3

# STANDARD

import os
import sys
import json
import shutil
import traceback

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.lazy import LabelMatching
from vlm_gist.fiftyone.fiftyone_utils import import_fiftyone, load_dataset, get_export_path, save_dataset

from nimbro_utils.lazy import start_and_spin_node, SelfShutdown, Logger, ParameterHandler, read_json, escape

### <Parameter Defaults>

severity = 20
parallel_completions = 0
keep_detection_results = True
save_mask_with_detections = True
export_path = "dataset_path/../fo_stamp_label_match"

### </Parameter Defaults>

class FiftyOneLabelMatch(Node):

    def __init__(self, name="fiftyone_label_match", *, context=None, dataset_paths, settings_paths, **kwargs):
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
            description="Backup detections before label-matching.",
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
            name="export_path",
            dtype=str,
            default_value=export_path,
            description="Path to save new dataset.",
            read_only=True
        )

        self.parameter_handler.deactivate_declarations()

        self.label_matching = LabelMatching(
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
            self._logger.info(f"Handling dataset '{i + 1}' of '{len(self.dataset_paths)}': '{path}'")
            try:
                if self.settings_paths is not None:
                    success, message, settings = read_json(file_path=self.settings_paths[i], logger=self._logger)
                    if success:
                        success, message, _ = self.label_matching.set_settings(settings=settings)
                        if not success:
                            raise SelfShutdown(message)
                    else:
                        raise SelfShutdown(message)
                success, message, _ = self.label_matching.set_settings(
                    settings={
                        'usage_skip': False,
                        'usage_delay': 1.0,
                        'data_path': path,
                        'metadata_file': "label_matches.json"
                    }
                )
                if not success:
                    raise SelfShutdown(message)

                fiftyone = import_fiftyone(logger=self._logger)
                dataset = load_dataset(fiftyone=fiftyone, dataset_path=path, logger=self._logger)
                dataset = self.label_match_dataset(fiftyone=fiftyone, dataset=dataset)
                export_path = get_export_path(export_path=self.parameters.export_path, dataset_path=path)
                save_dataset(fiftyone=fiftyone, dataset=dataset, export_dir=export_path, logger=self._logger)

                # move label matching file
                self.label_matching.save_metadata()
                matches_source = os.path.join(path, self.label_matching._settings['metadata_file'])
                if os.path.isfile(matches_source):
                    matches_target = os.path.join(export_path, self.label_matching._settings['metadata_file'])
                    self._logger.info(f"Moving label matching metadata file '{matches_source}' to dataset folder '{matches_target}'")
                    shutil.move(matches_source, matches_target)

                # copy validations file
                validation_source = os.path.join(path, "validations.json")
                if os.path.isfile(validation_source):
                    validation_target = os.path.join(export_path, "validations.json")
                    self._logger.info(f"Copying validations metadata file '{validation_source}' to dataset folder '{validation_target}'")
                    shutil.copy(validation_source, validation_target)

                # copy edited validation images
                images_source = os.path.join(path, "validation_edits")
                if os.path.isdir(images_source):
                    images_target = os.path.join(export_path, "validation_edits")
                    self._logger.info(f"Copying edited validation images '{images_source}' to '{images_target}'")
                    shutil.copytree(images_source, images_target)

                # copy validation crops
                images_source = os.path.join(path, "validation_crops")
                if os.path.isdir(images_source):
                    images_target = os.path.join(export_path, "validation_crops")
                    self._logger.info(f"Copying validation crops '{images_source}' to '{images_target}'")
                    shutil.copytree(images_source, images_target)

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
                if os.path.isfile(description_source):
                    description_target = os.path.join(export_path, "descriptions.json")
                    self._logger.info(f"Copying descriptions metadata file '{description_source}' to dataset folder '{description_target}'")
                    shutil.copy(description_source, description_target)

                # copy edited description images
                images_source = os.path.join(path, "description_edits")
                if os.path.isdir(images_source):
                    images_target = os.path.join(export_path, "description_edits")
                    self._logger.info(f"Copying edited description images '{images_source}' to '{images_target}'")
                    shutil.copytree(images_source, images_target)

                export_paths.append(export_path)
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

    def label_match_dataset(self, fiftyone, dataset):
        if 'detections' not in dataset.get_field_schema():
            self._logger.warn("The dataset does not contain any detections")
            return dataset

        self._logger.info("Handling sample fields")
        if self.parameters.keep_detection_results:
            dataset.set_values("detections_before_label_matching", dataset.values("detections"))
        if "detections_direct" in dataset.get_field_schema():
            dataset.delete_sample_field("detections_direct")
        if "detections_over" in dataset.get_field_schema():
            dataset.delete_sample_field("detections_over")
        if "detections_before_validation" in dataset.get_field_schema():
            dataset.delete_sample_field("detections_before_validation")

        self._logger.info("Obtaining label matching data")
        key_object_name = 'object_name' # TODO retrieve
        key_description = 'description' # TODO retrieve
        file_paths = dataset.values("filepath")
        image_names = [os.path.basename(os.path.normpath(path)) for path in file_paths]
        detections = dataset.values("detections.detections", missing_value=None)
        if all(detections_image is None for detections_image in detections):
            self._logger.warn("The dataset does not contain any detections")
            return dataset

        ground_truth = dataset.values("ground_truth.detections", missing_value=None)
        file_mask = [None for _ in range(len(file_paths))]
        labels = []
        descriptions = []
        identifiers = []
        if self.label_matching._settings['use_per_set_targets']:
            dataset_targets = []
        else:
            dataset_targets = None
        for i, _ in enumerate(dataset.iter_samples(progress=True)):
            if detections[i] is None or len(detections[i]) == 0:
                self._logger.warn(f"Image '{file_paths[i]}' does not contain any detections")
            elif ground_truth[i] is None and self.label_matching._settings['use_per_set_targets']:
                self._logger.warn(f"Image '{file_paths[i]}' does not contain any ground truth")
            else:
                file_mask[i] = len(labels)
                labels.append([detection.attributes[key_object_name].value if key_object_name in detection.attributes else detection.label for detection in detections[i]])
                descriptions.append([detection.attributes[key_description].value if key_description in detection.attributes else None for detection in detections[i]])
                identifiers.append(image_names[i])
                if self.label_matching._settings['use_per_set_targets']:
                    dataset_targets.append([detection.label for detection in ground_truth[i]])

        self._logger.info("Performing label matching")
        success, message, matching_results, metadata = self.label_matching.get(
            labels=labels,
            descriptions=descriptions,
            identifiers=identifiers,
            dataset_targets=dataset_targets
        )
        self.label_matching.release_completions()
        if not success:
            raise SelfShutdown(message)

        self._logger.info("Applying label matching to dataset")
        for i, sample in enumerate(dataset.iter_samples(progress=False)):
            image_detections = []
            if file_mask[i] is not None:
                for j in range(len(detections[i])):
                    if matching_results[file_mask[i]][j] is not None:
                        detection = detections[i][j]
                        detection.label = matching_results[file_mask[i]][j]
                        if not self.parameters.save_mask_with_detections:
                            detection.masks = None
                        image_detections.append(detection)
                sample["detections"] = fiftyone.Detections(detections=image_detections)
                sample["num_detections_matched"] = len(image_detections)
                sample["label_matching_info"] = json.dumps(metadata[file_mask[i]], indent=4)
            sample.save()

        return dataset

def main():
    if len(sys.argv) < 2 or (len(sys.argv) > 2 and len(sys.argv) % 2 == 0):
        print("Usage: ros2 run vlm_gist fiftyone_label_match -- <dataset_path>\n   or: ros2 run vlm_gist fiftyone_label_match -- <dataset_path> <settings_path>\n   or: ros2 run vlm_gist fiftyone_label_match -- <dataset_path_1> <settings_path_1> ...")
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

    start_and_spin_node(FiftyOneLabelMatch, node_args={'dataset_paths': dataset_paths, 'settings_paths': settings_paths})

if __name__ == '__main__':
    main()
