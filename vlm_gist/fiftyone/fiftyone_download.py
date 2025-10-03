#!/usr/bin/env python3

# STANDARD

import os
import traceback

import cv2

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.fiftyone.fiftyone_utils import import_fiftyone, get_export_path, save_dataset

from nimbro_utils.lazy import start_and_spin_node, SelfShutdown, Logger, ParameterHandler, write_json, escape, get_package_path

### <Parameter Defaults>

severity = 20
name_or_url = "coco-2017"
split = "validation" # train, validation, test
seed = 666
max_samples = 500
crowd_mode = "remove" # leave, fix, remove
validate = True
export_path = os.path.join(get_package_path("vlm_gist"), "data", "fo_stamp_download")

### </Parameter Defaults>

class FiftyOneDownload(Node):

    def __init__(self, name="fiftyone_download", *, context=None, **kwargs):
        super().__init__(name, context=context, **kwargs)
        self._logger = Logger(self)

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
            name="name_or_url",
            dtype=str,
            default_value=name_or_url,
            description="Argument passed to 'fiftyone.zoo.load_zoo_dataset'.",
            read_only=True
        )

        self.parameter_handler.declare(
            name="split",
            dtype=str,
            default_value=split,
            description="Argument passed to 'fiftyone.zoo.load_zoo_dataset'.",
            read_only=True
        )

        self.parameter_handler.declare(
            name="seed",
            dtype=int,
            default_value=seed,
            description="Argument passed to 'fiftyone.zoo.load_zoo_dataset'.",
            read_only=True
        )

        self.parameter_handler.declare(
            name="max_samples",
            dtype=int,
            default_value=max_samples,
            description="Argument passed to 'fiftyone.zoo.load_zoo_dataset'.",
            read_only=True
        )

        self.parameter_handler.declare(
            name="crowd_mode",
            dtype=str,
            default_value=crowd_mode,
            description="Style of handling 'is_crowd' attribute in ['leave', 'remove', 'fix'].",
            read_only=True
        )

        self.parameter_handler.declare(
            name="validate",
            dtype=bool,
            default_value=validate,
            description="Validate image files after download.",
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

        self.timer_state = self.create_timer(0.0, self.state_machine, callback_group=MutuallyExclusiveCallbackGroup())

        self._logger.info("Node started")

    def __del__(self):
        self._logger.info("Node shutdown")

    def filter_parameter(self, name, value, is_declared):
        message = None

        if name == "severity":
            self.get_logger().set_settings(settings={'severity': value})

        elif name == "crowd_mode":
            valid_modes = ["leave", "remove", "fix"]
            if value not in valid_modes:
                value = None
                message = f"Value must be in {valid_modes}."

        return value, message

    def state_machine(self):
        self.timer_state.cancel()
        try:
            fiftyone = import_fiftyone(logger=self._logger)

            self._logger.info("Downloading dataset")
            dataset = fiftyone.zoo.load_zoo_dataset(
                name_or_url=self.parameters.name_or_url,
                split=self.parameters.split,
                shuffle=False,
                label_types=["detections"],
                seed=self.parameters.seed,
                max_samples=self.parameters.max_samples
            )

            self._logger.info(f"Handling 'is_crowd' attribute with mode '{self.parameters.crowd_mode}'")
            if crowd_mode == "leave":
                pass
            else:
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
                                            if self.parameters.crowd_mode == 'fix':
                                                processed_detection.iscrowd = 0
                                                processed_detections.append(processed_detection)
                                        else:
                                            processed_detections.append(processed_detection)
                                    sample[field] = fiftyone.Detections(detections=processed_detections)
                                    sample.save()

            if self.parameters.validate:
                self._logger.info("Checking image validity")
                for x, sample in enumerate(dataset.iter_samples(progress=True)):
                    img = cv2.imread(sample.filepath)
                    if img is None:
                        raise SelfShutdown(f"File {sample.filepath} is corrupted")

            export_path = get_export_path(export_path=self.parameters.export_path)
            save_dataset(fiftyone=fiftyone, dataset=dataset, export_dir=export_path, logger=self._logger)

            settings_path = os.path.join(export_path, "download.json")
            self._logger.info(f"Saving download settings to '{settings_path}'")
            write_json(
                file_path=settings_path,
                json_object={
                    'name_or_url': self.parameters.name_or_url,
                    'split': self.parameters.split,
                    'seed': self.parameters.seed,
                    'max_samples': self.parameters.max_samples,
                    'export_path': export_path,
                    'validate': self.parameters.validate
                },
                logger=self._logger
            )

            self._logger.info(f"Edit dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_edit -- {export_path}{escape['end']}")
            self._logger.info(f"Label dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_label -- {export_path}{escape['end']}")
            self._logger.info(f"Describe dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_describe -- {export_path}{escape['end']}")
            self._logger.info(f"Visualize dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_show -- {export_path}{escape['end']}")

            dataset.delete()
        except SelfShutdown as e:
            self._logger.error(f"{e}")
        except Exception as e:
            self._logger.error(f"{repr(e)}\n{traceback.format_exc()}")
        finally:
            self._logger.info("Node stopped")

def main():
    start_and_spin_node(FiftyOneDownload)

if __name__ == '__main__':
    main()
