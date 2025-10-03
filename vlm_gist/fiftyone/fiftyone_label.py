#!/usr/bin/env python3

# STANDARD

import os
import sys
import traceback

import cv2
import numpy as np

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.fiftyone.fiftyone_utils import import_fiftyone, load_dataset, get_export_path, save_dataset

from nimbro_api.api_director import ApiDirector
from nimbro_utils.lazy import start_and_spin_node, SelfShutdown, Logger, ParameterHandler, encode_mask, escape

### <Parameter Defaults>

severity = 20
field = "ground_truth"
export_path = "dataset_path/../fo_stamp_label"

### </Parameter Defaults>

class FiftyOneLabel(Node):

    def __init__(self, name="fiftyone_label", *, context=None, dataset_path, **kwargs):
        super().__init__(name, context=context, **kwargs)
        self._logger = Logger(self)

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

        self.api_director = ApiDirector(self._node, {'severity': 30})

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
            dataset = load_dataset(fiftyone=fiftyone, dataset_path=self.dataset_path, load_masks=True, crowd_mode="fix", logger=self._logger)
            export_path = get_export_path(export_path=self.parameters.export_path, dataset_path=self.dataset_path)
            dataset = self.label_dataset(dataset=dataset, export_path=export_path)
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

    def label_dataset(self, dataset):
        self._logger.info("Configuring completions node")

        success, message, completions_id = self.api_director.acquire(retry=True)
        if not success:
            raise SelfShutdown(message)

        settings = {
            'logger_level': "20",
            'probe_api_connection': "False",
            'api_endpoint': "OpenAI",
            'model_name': "gpt-4o",
            # 'api_endpoint': "OpenRouter",
            # 'model_name': "google/gemini-2.0-flash-001",
            'model_temperatur': "1.0",
            'model_top_p': "1.0",
            'model_max_tokens': "1500",
            'model_presence_penalty': "0.0",
            'model_frequency_penalty': "0.0",
            'stream_completion': "True",
            'normalize_text_response': "False",
            'max_tool_calls_per_response': "1",
            'correction_attempts': "2",
            'timeout_chunk': "50.0", # 15.0
            'timeout_completion': "90.0"
        }

        success, message = self.api_director.set_parameters(
            completions_id=completions_id,
            parameter_names=list(settings.keys()),
            parameter_values=list(settings.values()),
            retry=True
        )
        if not success:
            raise SelfShutdown(message)

        nulls = 0
        for i, sample in enumerate(dataset.iter_samples(progress=False)):
            # if i != 47:
            #     continue # TODO remove
            self._logger.info(f"Processing sample '{i + 1}' of '{len(dataset)}' ('{sample.filepath}')")
            if self.parameters.field not in sample:
                self._logger.warn(f"Sample '{sample.filepath}' does not contain field '{self.parameters.field}'")
            else:
                if sample[self.parameters.field] is None:
                    self._logger.warn(f"Field '{self.parameters.field}' of sample '{sample.filepath}' is 'None'")
                elif sample[self.parameters.field] is None:
                    self._logger.warn(f"Field '{self.parameters.field}' of sample '{sample.filepath}' is empty list")
                else:
                    # collect dam prompts
                    masks = []
                    bboxes = []
                    for j, detection in enumerate(sample[self.parameters.field].detections):
                        mask = detection.mask
                        if mask is None:
                            raise SelfShutdown(f"Detection '{j}' of sample '{sample.filepath}' does not contain a mask")
                        else:
                            mask = encode_mask(detection.mask > 0)
                            masks.append(mask)

                            # format bounding box
                            width = sample.metadata.width
                            height = sample.metadata.height
                            bbox = detection.bounding_box
                            bbox = [bbox[0] * width, bbox[1] * height, (bbox[0] + bbox[2]) * width, (bbox[1] + bbox[3]) * height]
                            bbox = np.round(bbox).astype(int).tolist()
                            bboxes.append(bbox)

                    # generate dam descriptions
                    self._logger.info(f"Generating '{len(bboxes)}' DAM descriptions")
                    success, message, dam_descriptions = self.api_director.dam(
                        image=sample.filepath,
                        prompts=[{'mask': mask, 'bbox': bbox} for mask, bbox in zip(masks, bboxes)],
                        max_batch_size=16,
                        retry=True
                    )
                    if not success:
                        raise SelfShutdown(message)

                    # # apply dam descriptions
                    # for j, detection in enumerate(sample[self.parameters.field].detections):
                    #     sample[self.parameters.field].detections[j].label = dam_descriptions[j]

                    # save crops to file
                    self._logger.info("Saving crops and highlights")
                    crop_files, highlight_files = [], []
                    edits = os.path.join(export_path, "edits")
                    os.makedirs(edits, exist_ok=True)
                    img_rgb = cv2.imread(sample.filepath).copy()[..., ::-1]
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    for j, detection in enumerate(sample[self.parameters.field].detections):
                        bbox = bboxes[j]
                        crop = img_bgr[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        crop_file = os.path.join(edits, f"crop_{i}_{j}.png")
                        cv2.imwrite(crop_file, crop)
                        crop_files.append(crop_file)

                        highlight = img_bgr.copy()
                        highlight = cv2.rectangle(highlight, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 5)
                        highlight_file = os.path.join(edits, f"bbox_{i}_{j}.png")
                        cv2.imwrite(highlight_file, highlight)
                        highlight_files.append(highlight_file)

                    # generate vlm descriptions
                    vlm_descriptions = []
                    for j, detection in enumerate(sample[self.parameters.field].detections):
                        self._logger.info(f"Describing detection '{j + 1}' of '{len(sample[self.parameters.field].detections)}' ('{sample.filepath}')")

                        prompt = "You are a visual perception system that identifies and described objects in an image. Be concise and factual."
                        success, message, _, _ = self.api_director.prompt(
                            completions_id=completions_id,
                            text=prompt,
                            role="system",
                            reset_context=True,
                            tool_response_id=None,
                            response_type="none",
                            retry=True
                        )
                        if not success:
                            raise SelfShutdown(message)

                        content = [
                            {'type': "text", 'text': "I need your help describing the object highlighted by the red bounding box in the following image."},
                            {'type': "image_url", 'image_url': {'detail': 'high', 'url': highlight_files[j]}},
                            {'type': "text", 'text': "Here is a close-up of the highlighted object, which should help you assess it in more detail."},
                            {'type': "image_url", 'image_url': {'detail': 'high', 'url': crop_files[j]}},
                            {'type': "text", 'text': (
                                "Also, here is a description of the object obtained from another method:\n"
                                f"'{dam_descriptions[j]}'\n"
                                "Be aware that it might not be correct, so feel free to ignore it if you think it's wrong. "
                                "However, it may still provide helpful information."
                            )},
                            {'type': "text", 'text': (
                                "Please describe this object in a single sentence:\n"
                                "- The description must start with 'A' or 'An' and end with '.'\n"
                                "- The description must contain between 10 and 15 words\n"
                                "- The description must summarize the object's most important attributes, including its type, color, and characteristic properties\n"
                                "- If there are multiple objects visible in the close-up, focus your description on the object in the background that fills the entire area of the crop\n"
                                "- The description must end with a reference to the location of the object, relating it to another close object in the image (left, right, above, below, inside, next, etc.)\n"
                                "- The description must enable the object shown to be uniquely identified so that it cannot be confused with any other object in the complete image, even if they are very similar or of the same type\n"
                                "- Your response must contain only this description\n"
                                "- If you're unsure about the nature of the object, focus your description on visual properties instead of making an excuse"
                                # "- If you're unsure about the object, respond with NULL instead"
                            )}
                        ]
                        success, message, description, _ = self.api_director.prompt(
                            completions_id=completions_id,
                            text={"role": "user", "content": content},
                            role="json",
                            reset_context=False,
                            tool_response_id=None,
                            response_type="text",
                            retry=True
                        )
                        if not success:
                            raise SelfShutdown(message)

                        vlm_descriptions.append(description)

                    # apply vlm descriptions
                    # next_null = 0
                    for j, detection in enumerate(sample[self.parameters.field].detections):
                        # if levenshtein(vlm_descriptions[j], "NULL", normalization=True) == 0:
                        #     sample[self.parameters.field].detections[j].label = str(next_null)
                        #     next_null += 1
                        #     nulls += 1
                        # else:
                        sample[self.parameters.field].detections[j].label = f"{nulls:03}: {vlm_descriptions[j]}"
                        nulls += 1

            sample.save()

        self.api_director.release(completions_id=completions_id)
        # self._logger.info(f"Total number of 'NULL' labels: '{nulls}'")

        return dataset

def main():
    if len(sys.argv) != 2:
        print("Usage: ros2 run vlm_gist fiftyone_label -- <dataset_path>")
    else:
        dataset_path = sys.argv[1]
        start_and_spin_node(FiftyOneLabel, node_args={'dataset_path': dataset_path})

if __name__ == '__main__':
    main()
