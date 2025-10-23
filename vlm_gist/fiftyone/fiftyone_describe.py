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

from vlm_gist.lazy import StructuredDescription
from vlm_gist.fiftyone.fiftyone_utils import import_fiftyone, load_dataset, get_export_path, save_dataset

from nimbro_utils.lazy import start_and_spin_node, SelfShutdown, Logger, ParameterHandler, read_json, escape, get_package_path

### <Parameter Defaults>

severity = 20
parallel_completions = 0
export_path = os.path.join(get_package_path("vlm_gist"), "data", "fo_stamp_describe")

### </Parameter Defaults>

class FiftyOneDescribe(Node):

    def __init__(self, name="fiftyone_describe", *, context=None, dataset_paths, settings_paths, **kwargs):
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
            name="export_path",
            dtype=str,
            default_value=export_path,
            description="Path to save new dataset.",
            read_only=True
        )

        self.parameter_handler.deactivate_declarations()

        self.structured_description = StructuredDescription(
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
                        success, message, _ = self.structured_description.set_settings(settings=settings)
                        if not success:
                            raise SelfShutdown(message)
                    else:
                        raise SelfShutdown(message)
                success, message, _ = self.structured_description.set_settings(
                    settings={
                        'timeout': None,
                        'leave_images_in_place': True,
                        'keep_image_name': True,
                        'usage_skip': False,
                        'usage_delay': 1.0,
                        'data_path': path,
                        'image_folder': "description_edits",
                        'metadata_file': "descriptions.json",
                        'metadata_write_no_paths': True
                    }
                )
                if not success:
                    raise SelfShutdown(message)

                fiftyone = import_fiftyone(logger=self._logger)
                dataset = load_dataset(fiftyone=fiftyone, dataset_path=path, logger=self._logger)
                dataset = self.describe_dataset(dataset=dataset)
                export_path = get_export_path(export_path=self.parameters.export_path, dataset_path=path)
                save_dataset(fiftyone=fiftyone, dataset=dataset, export_dir=export_path, logger=self._logger)

                # move description file
                self.structured_description.save_metadata()
                description_source = os.path.join(path, self.structured_description._settings['metadata_file'])
                description_target = os.path.join(export_path, self.structured_description._settings['metadata_file'])
                self._logger.info(f"Moving descriptions metadata file '{description_source}' to dataset folder '{description_target}'")
                shutil.move(description_source, description_target)

                # move edited description images
                images_source = os.path.join(path, self.structured_description._settings['image_folder'])
                if os.path.isdir(images_source):
                    images_target = os.path.join(export_path, self.structured_description._settings['image_folder'])
                    self._logger.info(f"Moving edited description images '{images_source}' to '{images_target}'")
                    shutil.move(images_source, images_target)

                export_paths.append(export_path)
                self._logger.info(f"Detect dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_detect -- '{export_path}'{escape['end']}")
                self._logger.info(f"Evaluate dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_eval -- '{export_path}'{escape['end']}")
                self._logger.info(f"Visualize dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_show -- '{export_path}'{escape['end']}")

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

    def describe_dataset(self, dataset):
        file_paths = dataset.values("filepath")
        _, _, structured_descriptions, scene_descriptions, metadatas = self.structured_description.get(images=file_paths)
        self.structured_description.release_completions()

        assert len(structured_descriptions) == len(file_paths), f"{len(structured_descriptions)}, {len(file_paths)}"
        assert len(scene_descriptions) == len(file_paths), f"{len(scene_descriptions)}, {len(file_paths)}"
        assert len(metadatas) == len(file_paths), f"{len(metadatas)}, {len(file_paths)}"

        for sample, scene_description, structured_description, metadata in zip(dataset, scene_descriptions, structured_descriptions, metadatas):
            if scene_description is not None:
                sample["scene_description"] = scene_description
            if structured_description is None:
                sample["num_described_instances"] = 0
            else:
                structured_description_str = json.dumps(structured_description, indent=4)
                sample["structured_description"] = structured_description_str
                sample["num_described_instances"] = len(structured_description)
            if metadata is not None:
                metadata_str = json.dumps(metadata, indent=4)
                sample["description_info"] = metadata_str
            sample.save()

        return dataset

def main():
    if len(sys.argv) < 2 or (len(sys.argv) > 2 and len(sys.argv) % 2 == 0):
        print("Usage: ros2 run vlm_gist fiftyone_describe -- <dataset_path>\n   or: ros2 run vlm_gist fiftyone_describe -- <dataset_path> <settings_path>\n   or: ros2 run vlm_gist fiftyone_describe -- <dataset_path_1> <settings_path_1> ...")
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

    start_and_spin_node(FiftyOneDescribe, node_args={'dataset_paths': dataset_paths, 'settings_paths': settings_paths})

if __name__ == '__main__':
    main()
