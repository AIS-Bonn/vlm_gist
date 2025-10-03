#!/usr/bin/env python3

# STANDARD

import os
import sys
import traceback

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.fiftyone.fiftyone_utils import import_fiftyone, load_dataset, get_export_path, save_dataset

from nimbro_utils.lazy import start_and_spin_node, SelfShutdown, Logger, ParameterHandler, read_json, write_json, escape

### <Parameter Defaults>

severity = 20
field = "ground_truth"
export_path = "dataset_path/../fo_stamp_edit"

### </Parameter Defaults>

class FiftyOneEdit(Node):

    def __init__(self, name="fiftyone_edit", *, context=None, dataset_path, **kwargs):
        super().__init__(name, context=context, **kwargs)
        self._logger = Logger(self)

        self.session = None
        self.dataset_path = os.path.normpath(dataset_path)
        if not os.path.isdir(self.dataset_path):
            self._logger.error(f"Dataset '{self.dataset_path}' does not exist")
            raise SelfShutdown()

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
            name="field",
            dtype=str,
            default_value=field,
            description="Name of the dataset field to edit.",
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

        return value, message

    def state_machine(self):
        self.timer_state.cancel()
        try:
            fiftyone = import_fiftyone(logger=self._logger)
            dataset = load_dataset(fiftyone=fiftyone, dataset_path=self.dataset_path, logger=self._logger)
            dataset = self.edit_dataset(fiftyone=fiftyone, dataset=dataset)
            export_path = get_export_path(export_path=self.parameters.export_path, dataset_path=self.dataset_path)
            save_dataset(fiftyone=fiftyone, dataset=dataset, export_dir=export_path, logger=self._logger)
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

    def edit_dataset(self, fiftyone, dataset):
        self._logger.info("Collecting labels of all detections")
        labels_per_file = {}
        for i, sample in enumerate(dataset.iter_samples(progress=True)):
            assert sample.filepath not in labels_per_file
            labels_per_file[sample.filepath] = []
            if self.parameters.field not in sample:
                self._logger.warn(f"Sample '{sample.filepath}' does not contain field '{self.parameters.field}'")
            else:
                if sample[self.parameters.field] is None:
                    self._logger.warn(f"Field '{self.parameters.field}' of sample '{sample.filepath}' is 'None'")
                elif sample[self.parameters.field] is None:
                    self._logger.warn(f"Field '{self.parameters.field}' of sample '{sample.filepath}' is empty list")
                else:
                    for j, detection in enumerate(sample[self.parameters.field].detections):
                        labels_per_file[sample.filepath].append(detection.label)

        file_path = os.path.join(self.dataset_path, "edit_labels.json")
        self._logger.info(f"Writing all labels to file '{file_path}'")
        success, message = write_json(file_path=file_path, json_object=labels_per_file, indent=True, logger=self._logger)
        if not success:
            raise SelfShutdown(message)

        input(f"Edit the labels in '{file_path}' and hit enter when you are done!")

        self._logger.info(f"Reading edited labels from file '{file_path}'")
        success, message, labels_per_file = read_json(file_path=file_path, logger=self._logger)
        if not success:
            raise SelfShutdown(message)

        self._logger.info("Writing edited labels to dataset")
        edits = 0
        for i, sample in enumerate(dataset.iter_samples(progress=True)):
            assert sample.filepath in labels_per_file
            if self.parameters.field not in sample:
                self._logger.warn(f"Sample '{sample.filepath}' does not contain field '{self.parameters.field}'")
            else:
                if sample[self.parameters.field] is None:
                    self._logger.warn(f"Field '{self.parameters.field}' of sample '{sample.filepath}' is 'None'")
                elif sample[self.parameters.field] is None:
                    self._logger.warn(f"Field '{self.parameters.field}' of sample '{sample.filepath}' is empty list")
                else:
                    assert len(labels_per_file[sample.filepath]) == len(sample[self.parameters.field].detections), f"{len(labels_per_file[sample.filepath])} != {len(sample[self.parameters.field].detections)}"
                    for j, detection in enumerate(sample[self.parameters.field].detections):
                        if sample[self.parameters.field].detections[j].label != labels_per_file[sample.filepath][j]:
                            self._logger.info(f"Assigning detection '{j}' of sample '{sample.filepath}': '{sample[self.parameters.field].detections[j].label}' -> '{labels_per_file[sample.filepath][j]}'")
                            sample[self.parameters.field].detections[j].label = labels_per_file[sample.filepath][j]
                            edits += 1
                    del labels_per_file[sample.filepath]
                    sample.save()
        assert len(labels_per_file) == 0, f"{len(labels_per_file)}"
        self._logger.info(f"Total number of edits: '{edits}'")

        self._logger.info(f"Deleting file '{file_path}'")
        os.remove(file_path)

        return dataset

def main():
    if len(sys.argv) != 2:
        print("Usage: ros2 run vlm_gist fiftyone_edit -- <dataset_path>")
    else:
        dataset_path = sys.argv[1]
        start_and_spin_node(FiftyOneEdit, node_args={'dataset_path': dataset_path})

if __name__ == '__main__':
    main()
