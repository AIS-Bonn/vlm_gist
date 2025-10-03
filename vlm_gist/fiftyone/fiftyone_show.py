#!/usr/bin/env python3

# STANDARD

import os
import sys
import traceback

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.fiftyone.fiftyone_utils import import_fiftyone, load_dataset, show_dataset

from nimbro_utils.lazy import start_and_spin_node, SelfShutdown, Logger, ParameterHandler

### <Parameter Defaults>

severity = 20

### </Parameter Defaults>

class FiftyOneShow(Node):

    def __init__(self, name="fiftyone_show", *, context=None, dataset_path, **kwargs):
        super().__init__(name, context=context, **kwargs)
        self._logger = Logger(self)

        self.dataset_path = os.path.normpath(dataset_path)
        if not os.path.isdir(self.dataset_path):
            self._logger.error(f"Dataset '{self.dataset_path}' does not exist")
            raise SelfShutdown()

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
            show_dataset(fiftyone=fiftyone, dataset=dataset, logger=self._logger)
        except SelfShutdown as e:
            self._logger.error(f"{e}")
        except Exception as e:
            self._logger.error(f"{repr(e)}\n{traceback.format_exc()}")

def main():
    if len(sys.argv) != 2:
        print("Usage: ros2 run vlm_gist fiftyone_describe -- <dataset_path>")
    else:
        dataset_path = sys.argv[1]
        start_and_spin_node(FiftyOneShow, node_args={'dataset_path': dataset_path})

if __name__ == '__main__':
    main()
