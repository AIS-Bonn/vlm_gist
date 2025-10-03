#!/usr/bin/env python3

# STANDARD

import os
import re
import cv2
import copy
import json
import time
import uuid
import datetime
import threading
import collections
import multiprocessing
from queue import Queue

import numpy as np

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.core.common import read_image, set_logger, set_settings, complete_settings, release_completions, format_seconds, save_metadata

from nimbro_api import ApiDirector
from nimbro_utils.lazy import encode_mask, levenshtein_match, get_package_path

default_settings = {
    'vlm': {
        'probe_api_connection': "False",
        'api_endpoint': "OpenRouter",
        'model_name': "google/gemini-2.5-flash",
        'model_temperature': "1.0",
        'model_top_p': "1.0",
        'model_max_tokens': "50",
        'model_presence_penalty': "0.0",
        'model_frequency_penalty': "0.0",
        'model_reasoning_effort': "none",
        'completion_parsers': "[]",
        'stream_completion': "True",
        'normalize_text_completion': "True",
        'correction_attempts': "0",
        'timeout_chunk_first': "15.0",
        'timeout_chunk_next': "10.0",
        'timeout_completion': "30.0"
    },
    'image_resolution_full': ["high", "low"][0],
    'image_resolution_crop': ["high", "low"][0],
    'bbox_scale': 1.0,
    'background_saturation': 1.0,
    'background_brightness': 1.0,
    'prepend_image': True,
    'system_prompt': "You are a robot's visual perception system that identifies and analyzes objects in an image. Be concise and factual.",
    'description_prompt': "Provide a JSON description of all object instances in the image above.",
    'validation_prompt': "Is this cropped image taken from the full image above a tight bounding box around any one of the objects mentioned in the description? If yes, your answer must only contain the one corresponding object_name. Otherwise, respond with '{invalid}'.",
    'image_res': "high",
    'invalid': "n/a",
    'indent': 4,
    'levenshtein_threshold': 2,
    'retry': True, # retry API requests after failure instead of skipping the current image
    'timeout': None, # set None to deactivate or set number of seconds as int or float in which get() must finish
    'leave_images_in_place': False, # attempt to not copy images passed as string (local path or web)
    'keep_image_name': False, # keep original image name when copying instead of using a timestamp based name
    'png_compression_level': 2, # 0 (off) to 9 (max)
    'png_max_pixels': 1920 * 1080, # set None to deactivate re-scaling
    'crop_masks': True, # compresses masks by cropping them using their bounding boxes
    'usage_skip': False, # set True to not obtain usage information (tokens & dollars); Only skipped if entire batch wants to skip
    'usage_delay': 1.0, # time in seconds to sleep before requesting usage to ensure was captured; Maximum of batch is used
    'data_path': os.path.join(get_package_path("vlm_gist"), "data", "validations"),
    'image_folder': "validation_edits", # folder within data_path to store images; set None to use data_path directly
    'crop_folder': "validation_crops",
    'metadata_file': "validations.json", # set None to deactivate
    'metadata_write_relative_paths': False, # write relative path between metadata and image instead of absolute image paths to metadata
    'metadata_write_no_paths': True # only write image names to metadata; if True this overwrites 'metadata_write_relative_paths'
}

class Validation:

    def __init__(self, node, parallel_completions=None, metadata_timer=True, settings=None, logger_name=None, logger_severity=20):
        assert isinstance(node, Node), f"Provided argument 'node' is of invalid type '{type(node).__name__}'. Supported type is 'rclpy.node.Node'."
        assert parallel_completions is None or (isinstance(parallel_completions, int) and parallel_completions > 0), f"Provided argument 'parallel_completions' is of invalid type '{type(parallel_completions).__name__}'. Supported types are 'None' and 'int > 0'."
        assert isinstance(metadata_timer, bool), f"Provided argument 'metadata_timer' is of unsupported type '{type(metadata_timer).__name__}'. Supported type is 'bool'."
        assert logger_name is None or isinstance(logger_name, str), f"Provided argument 'logger_name' is of unsupported type '{type(logger_name).__name__}'. Supported types are 'None' and 'str'."

        self._node = node

        # logger
        if logger_name is None:
            logger_name = (f"{self._node.get_namespace()}.{self._node.get_name()}.validation").replace("/", "")
            if logger_name[0] == ".":
                logger_name = logger_name[1:]
        self.set_logger = set_logger.__get__(self)
        success, message = self.set_logger(logger_name, logger_severity)
        assert success, message

        # parallel completions
        if parallel_completions is None:
            parallel_completions = multiprocessing.cpu_count() - 1
        self._status_lock = threading.Lock()
        self._completions = [{'ID': None, 'status': "NOT_ACQUIRED"} for _ in range(parallel_completions)]
        self.release_completions = release_completions.__get__(self)
        self._logger.debug(f"Using '{parallel_completions}' parallel completion{'s' if parallel_completions != 1 else ''}")

        # metadata
        self._metadata_queue = Queue(maxsize=0)
        self.save_metadata = save_metadata.__get__(self)
        self._metadata_timer = self._node.create_timer(10.0, self.save_metadata, callback_group=MutuallyExclusiveCallbackGroup(), autostart=metadata_timer)

        # settings
        self._default_settings = default_settings
        self.complete_settings = complete_settings.__get__(self)
        self.set_settings = set_settings.__get__(self)
        success, message, _ = self.set_settings(settings)
        assert success, message

        # ApiDirector
        if hasattr(self._node, 'api_director'):
            assert isinstance(self._node.api_director, ApiDirector), f"Expected existing attribute 'api_director' of parent node to be of type 'nimbro_api.api_director.ApiDirector' but it is of type '{type(self._node.api_director).__name__}'!"
        else:
            self._logger.debug("Adding 'ApiDirector' to parent node")
            self._node.api_director = ApiDirector(self._node, {'severity': 30})

    def __del__(self):
        self.release_completions()

    def get(self, images, structured_descriptions, identifiers, labels, bboxes, confidences=None, masks=None, track_ids=None, settings=None):
        stamp_start = datetime.datetime.now()

        # read settings
        settings = copy.deepcopy(settings)
        num_images = len(images) if isinstance(images, list) else 1
        if isinstance(settings, dict):
            settings = [settings] * num_images
        elif settings is None:
            settings = [None] * num_images
        elif not isinstance(settings, list):
            message = f"Provided argument 'settings' is of unsupported type '{type(settings).__name__}'. Supported types are 'None', 'dict', and 'list'."
            self._logger.error(message)
            return False, message, None, None, None, None, None, None
        elif not len(settings) == num_images:
            message = f"Expected number of settings '{len(settings)}' to match the number of images '{num_images}'."
            self._logger.error(message)
            return False, message, None, None, None, None, None, None
        for i in range(len(settings)):
            if settings[i] is None:
                settings[i] = copy.deepcopy(self._settings)
            else:
                success, message, settings[i] = self.complete_settings(settings[i])
                if not success:
                    return False, message, None, None, None, None, None, None
        timeout_idx = [i for i in range(num_images) if settings[i]['timeout'] is not None]
        self._logger.debug(f"Image IDs that require timeout: {timeout_idx}")

        # read images
        images = copy.deepcopy(images)
        image_paths_metadata = []
        image_paths = []
        no_batch = False
        if not isinstance(images, list):
            no_batch = True
            images = [images]
        for i in range(len(images)):
            image_folder = settings[i]['data_path'] if settings[i]['image_folder'] is None else os.path.join(settings[i]['data_path'], settings[i]['image_folder'])
            success, message, image, path = read_image( # TODO scale bounding box in/out when images get rescaled
                self,
                image=images[i],
                data_path=image_folder,
                leave_in_place=settings[i]['leave_images_in_place'],
                keep_name=settings[i]['keep_image_name'],
                png_compression_level=settings[i]['png_compression_level'],
                png_max_pixels=settings[i]['png_max_pixels']
            )
            if success:
                image_paths.append(path)
                images[i] = image
                if settings[i]['metadata_write_no_paths']:
                    image_paths_metadata.append(os.path.basename(path))
                elif settings[i]['metadata_write_relative_paths']:
                    image_paths_metadata.append(os.path.relpath(path, settings[i]['data_path']))
            else:
                return False, message, None, None, None, None, None, None
        self._logger.debug(f"Absolute image paths: {image_paths}")
        self._logger.debug(f"Metadata image paths: {image_paths_metadata}")
        image_paths_metadata_counter = dict(collections.Counter(image_paths_metadata))
        for key in copy.deepcopy(image_paths_metadata_counter):
            if image_paths_metadata_counter[key] == 1:
                del image_paths_metadata_counter[key]
        if len(image_paths_metadata_counter) > 0:
            message = f"The image paths/names supposed to be written to metadata according to settings are not unique: {image_paths_metadata_counter}"
            self._logger.error(message)
            return False, message, None, None, None, None, None, None

        # read identifiers
        identifiers = copy.deepcopy(identifiers)
        if isinstance(identifiers, str):
            identifiers = [identifiers] * num_images
        elif not isinstance(identifiers, list):
            message = f"Provided argument 'identifiers' is of unsupported type '{type(identifiers).__name__}'. Supported type is 'list'."
            self._logger.error(message)
            return False, message, None, None, None, None, None, None
        for identifier in identifiers:
            if not isinstance(identifier, str):
                message = f"Provided argument 'identifiers' contains identifier of unsupported type '{type(identifier).__name__}'. Supported type is 'str'."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            elif len(identifier) == 0:
                message = "Provided argument 'identifiers' contains empty string as identifier. Supported are non-empty strings."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
        self._logger.debug(f"identifiers: {identifiers}")

        # read structured descriptions
        structured_descriptions = copy.deepcopy(structured_descriptions)
        if not isinstance(structured_descriptions, list):
            message = f"Provided argument 'structured_descriptions' is of unsupported type '{type(structured_descriptions).__name__}'. Supported type is 'list'."
            self._logger.error(message)
            return False, message, None, None, None, None, None, None
        is_list = [isinstance(item, list) for item in structured_descriptions]
        is_dict = [isinstance(item, dict) for item in structured_descriptions]
        if not (all(is_list) or all(is_dict)):
            message = f"Provided argument 'structured_descriptions' contains unsupported set of types {[type(item).__name__ for item in structured_descriptions]}. Supported are either only elements of type 'dict' or of type 'list'."
            self._logger.error(message)
            return False, message, None, None, None, None, None, None
        if all(is_dict):
            structured_descriptions = [structured_descriptions] * num_images
        elif not len(structured_descriptions) == num_images:
            message = f"Expected number of structured descriptions '{len(structured_descriptions)}' to match the number of images '{num_images}'."
            self._logger.error(message)
            return False, message, None, None, None, None, None, None
        structured_descriptions_str = []
        for i, description in enumerate(structured_descriptions):
            if len(description) == 0:
                message = "Provided argument 'structured_descriptions' contains an empty structured description. Each structured description must feature at least one object instance."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            identifier_values = []
            for instance in description:
                if not isinstance(instance, dict):
                    message = f"Provided argument 'structured_descriptions' contains object instance of unsupported type '{type(instance).__name__}'. Supported type is 'dict'."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
                if identifiers[i] not in instance:
                    message = f"Provided argument 'structured_descriptions' contains structured description '{i}' that does not feature the expected identifier '{identifiers[i]}'."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
                identifier_values.append(instance[identifiers[i]])
            identifier_values = dict(collections.Counter(identifier_values))
            for value in copy.deepcopy(identifier_values):
                if identifier_values[value] == 1:
                    del identifier_values[value]
            if len(identifier_values) > 0:
                message = f"Argument 'structured_descriptions' contains structured description '{i}' that features duplicate values {identifier_values} for identifier '{identifiers[i]}'."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            try:
                structured_descriptions_str.append(json.dumps(description, indent=settings[i]['indent']))
            except Exception as e:
                message = f"Failed to dump structured description '{i}' to string: {repr(e)}"
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
        self._logger.debug(f"structured_descriptions: {structured_descriptions}")
        # self._logger.debug(f"structured_descriptions_str: {structured_descriptions_str}")

        # read bboxes
        bboxes = copy.deepcopy(bboxes)
        if isinstance(bboxes, np.ndarray):
            bboxes = bboxes.tolist()
        elif not isinstance(bboxes, (list, tuple)):
            message = f"Provided argument 'bboxes' is of unsupported type '{type(bboxes).__name__}'. Supported type are 'list', 'tuple', and 'numpy.ndarray'."
            self._logger.error(message)
            return False, message, None, None, None, None, None, None
        if all(isinstance(detection, (list, tuple, np.ndarray)) and len(detection) == 4 for detection in bboxes):
            bboxes = [bboxes] * num_images
        elif len(bboxes) != num_images:
            message = f"Expected number of detection sets '{len(bboxes)}' to match the number of images '{num_images}'."
            self._logger.error(message)
            return False, message, None, None, None, None, None, None
        for i in range(len(bboxes)):
            if isinstance(bboxes[i], np.ndarray):
                bboxes[i] = bboxes[i].tolist()
            elif not isinstance(bboxes[i], (list, tuple)):
                message = f"Provided argument 'bboxes' contains detection of unsupported type '{type(bboxes[i]).__name__}'. Supported type are 'list', 'tuple', and 'numpy.ndarray'."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            for j in range(len(bboxes[i])):
                if isinstance(bboxes[i][j], np.ndarray):
                    bboxes[i][j] = bboxes[i][j].tolist()
                elif not isinstance(bboxes[i][j], (list, tuple)):
                    message = f"Provided argument 'bboxes' contains detection of unsupported type '{type(bboxes[i][j]).__name__}'. Supported type are 'list', 'tuple', and 'numpy.ndarray'."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
                if len(bboxes[i][j]) != 4:
                    message = f"Provided argument 'bboxes' contains detection with length '{len(bboxes[i][j])}'. Each detection must have exactly 4 coordinates."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
                for coord in bboxes[i][j]:
                    if not isinstance(coord, int):
                        message = f"Provided argument 'bboxes' contains detection with coordinate of unsupported type '{type(coord).__name__}'. Supported type is 'int'."
                        self._logger.error(message)
                        return False, message, None, None, None, None, None, None
        self._logger.debug(f"bboxes: {bboxes}")
        # check there is at least one detection per image

        # read labels
        labels = copy.deepcopy(labels)
        if not isinstance(labels, (list, tuple)):
            message = f"Provided argument 'labels' is of unsupported type '{type(labels).__name__}'. Supported type are 'list' and 'tuple'."
            self._logger.error(message)
            return False, message, None, None, None, None, None, None
        if all(isinstance(label, str) for label in labels):
            labels = [labels] * len(bboxes)
        elif len(labels) != num_images:
            message = f"Expected number of label sets '{len(labels)}' to match the number of images '{num_images}'."
            self._logger.error(message)
            return False, message, None, None, None, None, None, None
        for i in range(len(labels)):
            if not isinstance(labels[i], (list, tuple)):
                message = f"Provided argument 'labels' contains label set of unsupported type '{type(labels[i]).__name__}'. Supported type are 'list', 'tuple', and 'numpy.ndarray'."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            if len(labels[i]) != len(bboxes[i]):
                message = f"Expected number of labels '{len(labels[i])}' for image '{i}' to match the number of detections '{len(bboxes[i])}'."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            for j in range(len(labels[i])):
                if not isinstance(labels[i][j], str):
                    message = f"Provided argument 'labels' contains label of unsupported type '{type(labels[i][j]).__name__}'. Supported type is 'str'."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
                if not len(labels[i][j]) > 0:
                    message = "Provided argument 'labels' contains label that is an empty string. Supported are non-empty strings."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
        self._logger.debug(f"labels: {labels}")

        # read confidences
        if confidences is not None:
            confidences = copy.deepcopy(confidences)
            if isinstance(confidences, np.ndarray):
                confidences = confidences.tolist()
            elif not isinstance(confidences, (list, tuple)):
                message = f"Provided argument 'confidences' is of unsupported type '{type(confidences).__name__}'. Supported type are 'None', 'list', 'tuple', and 'numpy.ndarray'."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            if all(isinstance(confidence, float) for confidence in confidences):
                confidences = [confidences] * len(bboxes)
            elif len(confidences) != num_images:
                message = f"Expected number of confidence sets '{num_images}' to match the number of images '{num_images}'."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            for i in range(len(confidences)):
                if isinstance(confidences[i], np.ndarray):
                    confidences[i] = confidences[i].tolist()
                elif not isinstance(confidences[i], (list, tuple)):
                    message = f"Provided argument 'confidences' contains confidence set of unsupported type '{type(confidences[i]).__name__}'. Supported type are 'list', 'tuple', and 'numpy.ndarray'."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
                if len(confidences[i]) != len(bboxes[i]):
                    message = f"Expected number of confidences '{len(confidences[i])}' for image '{i}' to match the number of detections '{len(bboxes[i])}'."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
                for j in range(len(confidences[i])):
                    if not isinstance(confidences[i][j], float):
                        message = f"Provided argument 'confidences' contains confidence of unsupported type '{type(confidences[i][j]).__name__}'. Supported type is 'float'."
                        self._logger.error(message)
                        return False, message, None, None, None, None, None, None
            self._logger.debug(f"confidences: {confidences}")

        # read track_ids
        if track_ids is not None:
            track_ids = copy.deepcopy(track_ids)
            if isinstance(track_ids, np.ndarray):
                track_ids = track_ids.tolist()
            elif not isinstance(track_ids, (list, tuple)):
                message = f"Provided argument 'track_ids' is of unsupported type '{type(track_ids).__name__}'. Supported type are 'None', 'list', 'tuple', and 'numpy.ndarray'."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            if all(isinstance(confidence, int) for confidence in track_ids):
                track_ids = [track_ids] * len(bboxes)
            elif len(track_ids) != num_images:
                message = f"Expected number of confidence sets '{num_images}' to match the number of images '{num_images}'."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            for i in range(len(track_ids)):
                if isinstance(track_ids[i], np.ndarray):
                    track_ids[i] = track_ids[i].tolist()
                elif not isinstance(track_ids[i], (list, tuple)):
                    message = f"Provided argument 'track_ids' contains confidence set of unsupported type '{type(track_ids[i]).__name__}'. Supported type are 'list', 'tuple', and 'numpy.ndarray'."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
                if len(track_ids[i]) != len(bboxes[i]):
                    message = f"Expected number of track_ids '{len(track_ids[i])}' for image '{i}' to match the number of detections '{len(bboxes[i])}'."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
                for j in range(len(track_ids[i])):
                    if not isinstance(track_ids[i][j], int):
                        message = f"Provided argument 'track_ids' contains confidence of unsupported type '{type(track_ids[i][j]).__name__}'. Supported type is 'int'."
                        self._logger.error(message)
                        return False, message, None, None, None, None, None, None
            self._logger.debug(f"track_ids: {track_ids}")

        # read masks
        if masks is not None:
            masks = copy.deepcopy(masks)
            if isinstance(masks, np.ndarray):
                masks = masks.tolist()
            elif not isinstance(masks, (list, tuple)):
                message = f"Provided argument 'masks' is of unsupported type '{type(masks).__name__}'. Supported type are 'None', 'list', 'tuple', and 'numpy.ndarray'."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            if all(isinstance(mask, (list, tuple, np.ndarray)) and all(isinstance(mask_dim, (list, tuple, np.ndarray)) and all(isinstance(coord, (bool, np.bool_)) for coord in mask_dim) for mask_dim in mask) for mask in masks):
                masks = [masks] * len(bboxes)
            elif len(masks) != num_images:
                message = f"Expected number of mask sets '{len(masks)}' to match the number of images '{num_images}'."
                self._logger.error(message)
                return False, message, None, None, None, None, None, None
            for i in range(len(masks)):
                if isinstance(masks[i], np.ndarray):
                    masks[i] = masks[i].tolist()
                elif not isinstance(masks[i], (list, tuple)):
                    message = f"Provided argument 'masks' contains mask set of unsupported type '{type(masks[i]).__name__}'. Supported type are 'list', 'tuple', and 'numpy.ndarray'."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
                if len(masks[i]) != len(bboxes[i]):
                    message = f"Expected number of masks '{len(masks[i])}' for image '{i}' to match the number of detections '{len(bboxes[i])}'."
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None
                for j in range(len(masks[i])):
                    if isinstance(masks[i][j], np.ndarray):
                        masks[i][j] = masks[i][j].tolist()
                    elif not isinstance(masks[i][j], (list, tuple)):
                        message = f"Provided argument 'masks' contains mask of unsupported type '{type(masks[i][j]).__name__}'. Supported type are 'list', 'tuple', and 'numpy.ndarray'."
                        self._logger.error(message)
                        return False, message, None, None, None, None, None, None
                    for k in range(len(masks[i][j])):
                        if isinstance(masks[i][j][k], np.ndarray):
                            masks[i][j] = masks[i][j][k].tolist()
                        elif not isinstance(masks[i][j][k], (list, tuple)):
                            message = f"Provided argument 'masks' contains first-dimension element of unsupported type '{type(masks[i][j][k]).__name__}'. Supported type are 'list', 'tuple', and 'numpy.ndarray'."
                            self._logger.error(message)
                            return False, message, None, None, None, None, None, None
                        for o in range(len(masks[i][j][k])):
                            if not isinstance(masks[i][j][k][o], bool):
                                message = f"Provided argument 'masks' contains second-dimension element of unsupported type '{type(masks[i][j][k][o]).__name__}'. Supported type is 'bool'."
                                self._logger.error(message)
                                return False, message, None, None, None, None, None, None

        # extract crops
        self._logger.info("Extracting bounding box crops")
        crops = []
        for i in range(num_images):
            image_crops = []

            image_bg = cv2.cvtColor(images[i].copy(), cv2.COLOR_BGR2HSV)
            image_bg[:, :, 1] = settings[i].get('background_saturation', 1.0) * image_bg[:, :, 1]
            image_bg = cv2.cvtColor(image_bg, cv2.COLOR_HSV2BGR)
            image_bg = np.round(settings[i].get('background_brightness', 1.0) * image_bg).astype(np.uint8)

            for j in range(len(bboxes[i])):
                masked_image = images[i].copy()
                if masks is not None:
                    if masks[i] is not None:
                        if masks[i][j] is not None:
                            mask_np = np.array(masks[i][j], dtype=bool) # TODO inefficient: repeated when writing metadata
                            if mask_np.shape[0] != masked_image.shape[0] or mask_np.shape[1] != masked_image.shape[1]:
                                # mask was cropped with bounding box
                                temp = np.ndarray(shape=(masked_image.shape[0], masked_image.shape[1]), dtype=bool)
                                temp[bboxes[i][j][1]:bboxes[i][j][3], bboxes[i][j][0]:bboxes[i][j][2]] = mask_np
                                mask_np = temp
                            masked_image[~mask_np] = image_bg[~mask_np]

                bbox = np.array(bboxes[i][j])
                center = (bbox[:2] + bbox[2:]) / 2
                size = bbox[2:] - bbox[:2]
                scaled_size = size * settings[i].get('bbox_scale', 1.0)
                bbox = np.concatenate([center - scaled_size / 2, center + scaled_size / 2])
                bbox = np.clip(bbox, [0, 0, 0, 0], [masked_image.shape[1], masked_image.shape[0], masked_image.shape[1], masked_image.shape[0]])
                bbox = np.round(bbox).astype(int)
                crop = masked_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                max_size = settings[i].get('max_crop_area')
                if max_size is not None:
                    if size[0] * size[1] > max_size:
                        crop = crop.copy()
                        aspect_ratio = crop.shape[1] / crop.shape[0]
                        new_height = np.sqrt(max_size / aspect_ratio)
                        new_width = new_height * aspect_ratio
                        new_height = int(np.round(new_height))
                        new_width = int(np.round(new_width))
                        crop = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        self._logger.warn(f"Crop of size '{size[0] * size[1]}' of image '{i}' exceeded '{max_size}', rescaled to '{new_height * new_width}'")

                file_name = f"{os.path.basename(image_paths[i])[:-4]}_crop_{j}.png"
                crop_folder_path = os.path.join(settings[i]['data_path'], settings[i]['crop_folder'])
                crop_file_path = os.path.join(crop_folder_path, file_name)
                try:
                    if not os.path.exists(crop_folder_path):
                        self._logger.debug(f"Creating crop folder '{crop_folder_path}'")
                        try:
                            os.makedirs(crop_folder_path)
                        except Exception as e:
                            raise Exception(f"Failed to create crop folder '{crop_folder_path}': {repr(e)}") from e
                    cv2.imwrite(crop_file_path, crop)
                except Exception as e:
                    message = f"Failed to save crop '{j}' of image '{i}' to file '{crop_file_path}': {repr(e)}"
                    self._logger.error(message)
                    return False, message, None, None, None, None, None, None

                image_crops.append(crop_file_path)
            crops.append(image_crops)
        self._logger.debug(f"crops: {crops}")

        # prepare uuids
        uuids = []
        for i in range(len(labels)):
            if settings[i]['usage_skip']:
                uuids.append([None] * len(labels[i]))
            else:
                uuid_detection = []
                for i in range(len(labels[i])):
                    uuid_detection.append(uuid.uuid4().hex)
                uuids.append(uuid_detection)
        self._logger.debug(f"UUIDs: {uuids}")

        # validate

        bboxes_per_image = [len(bboxes[i]) for i in range(num_images)]
        num_total_detections = sum(bboxes_per_image)
        map_flat_idx = {}
        i = 0
        for image_index, num_detections in enumerate(bboxes_per_image):
            for detection_index in range(num_detections):
                map_flat_idx[i] = (image_index, detection_index)
                i += 1

        validation_responses_raw = [[None for _ in bboxes[i]] for i in range(num_images)]
        validation_responses_error = [[[] for _ in bboxes[i]] for i in range(num_images)]

        self._logger.info(f"Validating '{num_images}' image{'s' if num_images != 1 else ''} with '{num_total_detections}' detection{'s' if num_total_detections != 1 else ''}")

        metadata = [None] * len(images)
        for i, image in enumerate(image_paths_metadata):
            metadata[i] = {
                'stamp_batch_start': stamp_start.isoformat(),
                'structured_description': structured_descriptions[i],
                'identifier': identifiers[i],
                'settings': settings[i]
            }
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'stamp_batch_start', metadata[i]['stamp_batch_start']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'structured_description', structured_descriptions[i]))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'identifier', identifiers[i]))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'settings', settings[i]))
            metadata[i]['parallel_completions'] = len(self._completions)
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'parallel_completions', len(self._completions)))
            metadata[i]['batch_size'] = len(images)
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'batch_size', len(images)))
            if len(images) > 1:
                metadata[i]['batch_images'] = image_paths_metadata
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'batch_images', image_paths_metadata))
            for j in range(len(bboxes[i])):
                metadata[i][str(j)] = {
                    'crop': crops[i][j],
                    'bbox': bboxes[i][j],
                    'label': labels[i][j],
                    'uuid': uuids[i][j]
                }
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'crop', crops[i][j]))
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'bbox', bboxes[i][j]))
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'label', labels[i][j]))
                if confidences is not None:
                    metadata[i][str(j)]['confidence'] = confidences[i][j]
                    self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'confidence', confidences[i][j]))
                if track_ids is not None:
                    metadata[i][str(j)]['track_id'] = track_ids[i][j]
                    self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'track_id', track_ids[i][j]))
                if masks is not None:
                    if masks[i] is not None:
                        if masks[i][j] is not None:
                            mask = np.array(masks[i][j], dtype=bool)
                            if mask.shape[0] != images[i].shape[0] or mask.shape[1] != images[i].shape[1]: # is crop
                                if settings[i]['crop_masks']:
                                    metadata[i][str(j)]['mask_cropped'] = True
                                else:
                                    metadata[i][str(j)]['mask_cropped'] = False
                                    full_mask = np.zeros((images[i].shape[0], images[i].shape[1]), dtype=bool)
                                    x0, y0, x1, y1 = bboxes[i][j]
                                    full_mask[y0:y1, x0:x1] = mask
                                    mask = full_mask
                            else:
                                if settings[i]['crop_masks']:
                                    metadata[i][str(j)]['mask_cropped'] = True
                                    x0, y0, x1, y1 = bboxes[i][j]
                                    mask = mask[y0:y1, x0:x1]
                                else:
                                    metadata[i][str(j)]['mask_cropped'] = False
                            metadata[i][str(j)]['mask'] = encode_mask(mask)
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'mask_cropped', metadata[i][str(j)]['mask_cropped']))
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'mask', metadata[i][str(j)]['mask']))

        status_order = ['UNPROCESSED', 'GENERATING_VALIDATION', 'ERROR', 'FINISHED']
        progress = [[{'status': "UNPROCESSED"} for _ in bboxes[i]] for i in range(num_images)]
        while True:
            # status and termination

            status = [[item['status'] for item in sublist] for sublist in progress]
            info = [item for sublist in status for item in sublist]
            info = dict(collections.Counter(info))
            info = {key: info[key] for key in status_order if key in info}
            info['TOTAL'] = num_total_detections
            num_done = sum(sum(item in ["FINISHED", "ERROR"] for item in sublist) for sublist in status)

            stamp_now = datetime.datetime.now()
            time_running = (stamp_now - stamp_start).total_seconds()
            time_left = f" (~{format_seconds((time_running / num_done) * (num_total_detections - num_done))} remaining)" if num_done > 0 else ""

            if num_done == num_total_detections:
                self._logger.info(f"Progress: {info} after {format_seconds(time_running)}")
                break

            self._logger.info(f"Progress: {info} after {format_seconds(time_running)}{time_left}", throttle_duration_sec=1.0)
            # self._logger.debug(f"{progress}", throttle_duration_sec=1.0)

            if len(timeout_idx) > 0:
                for i in timeout_idx:
                    if time_running > settings[i]['timeout']:
                        message = f"Timeout while processing '{image_paths[i]}' after '{settings[i]['timeout']}s'."
                        self._logger.error(message)
                        progress[i]['status'] = {'status': "ERROR"}
                        metadata[i]['stamp_error'] = stamp_now.isoformat()
                        metadata[i]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_error', metadata[i]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_error', message))

            # collect finished jobs
            forward = False
            for i in range(len(status)):
                for j in range(len(status[i])):
                    if status[i][j] == "GENERATING_VALIDATION":
                        success, _, completion_result = self._node.api_director.async_get(async_id=progress[i][j]['async_id'], mute_timeout_logging=True, timeout=0.0)
                        if success:
                            forward = True
                            del progress[i][j]['async_id']
                            self._completions[progress[i][j]['completions_ID']]['status'] = "IDLE"
                            success, message, completion = completion_result
                            if success:
                                metadata[i][str(j)]['completion'] = completion
                                validation_responses_raw[i][j] = completion['text']
                                metadata[i][str(j)]['stamp_validation_end'] = datetime.datetime.now().isoformat()
                                metadata[i][str(j)]['duration_validation'] = (datetime.datetime.fromisoformat(metadata[i][str(j)]['stamp_validation_end']) - datetime.datetime.fromisoformat(metadata[i][str(j)]['stamp_validation_start'])).total_seconds()
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], str(j), 'completion', metadata[i][str(j)]['completion']))
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], str(j), 'stamp_validation_end', metadata[i][str(j)]['stamp_validation_end']))
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], str(j), 'duration_validation', metadata[i][str(j)]['duration_validation']))
                                progress[i][j] = {'status': "FINISHED"}
                            elif settings[i]['retry']:
                                self._logger.info(f"Retrying validation of detection '{j}' from image '{i}' after failure")
                                metadata[i][str(j)]['failed_completions'] = metadata[i].get('failed_completions', []) + [completion]
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], str(j), 'failed_completions', metadata[i][str(j)]['failed_completions']))
                                progress[i][j] = {'status': "UNPROCESSED"}
                            else:
                                validation_responses_error[i][j].append(message)
                                metadata[i][str(j)]['stamp_error'] = datetime.datetime.now().isoformat()
                                metadata[i][str(j)]['report_error'] = message
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], str(j), 'stamp_error', metadata[i][str(j)]['stamp_error']))
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], str(j), 'report_error', message))
                                progress[i][j] = {'status': "ERROR"}

            if forward:
                continue

            # find next job
            for i in range(num_total_detections):
                if status[map_flat_idx[i][0]][map_flat_idx[i][1]] == "UNPROCESSED":
                    job = {'image_ID': map_flat_idx[i][0], 'detection_ID': map_flat_idx[i][1], 'task': "VALIDATION", "UUID": uuids[map_flat_idx[i][0]][map_flat_idx[i][1]]}
                    break
            else:
                # wait for job
                time.sleep(0.01)
                continue

            # completions allocation
            self._status_lock.acquire()
            status = [item['status'] for item in self._completions]
            # find idle
            try:
                completions_i = status.index("IDLE")
            except ValueError:
                # find not acquired
                try:
                    completions_i = status.index("NOT_ACQUIRED")
                except ValueError:
                    completions_i = None
                else:
                    # acquire
                    success, message, completions_id = self._node.api_director.acquire(reset_parameters=False, reset_context=False, retry=False)
                    if success:
                        self._completions[completions_i] = {'ID': completions_id, 'status': "ACQUIRED"}
                    else:
                        completions_i = None
            else:
                self._completions[completions_i]['status'] = "ACQUIRED"
            self._status_lock.release()
            if completions_i is None:
                # wait for available completion
                time.sleep(0.01)
                continue
            else:
                job['completions_ID'] = completions_i

            # do job
            self._logger.debug(f"Generating '{job['task']}' for detection '{job['detection_ID']}' of '{image_paths[job['image_ID']]}' using completions '{self._completions[job['completions_ID']]['ID']}'")
            if job['task'] == "VALIDATION":
                metadata[job['image_ID']][str(job['detection_ID'])]['stamp_validation_start'] = datetime.datetime.now().isoformat()
                self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'stamp_validation_start', metadata[job['image_ID']][str(job['detection_ID'])]['stamp_validation_start']))
                # set parameters
                success, message, async_id = self._node.api_director.async_set_parameters(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    parameter_names=list(settings[job['image_ID']]['vlm'].keys()),
                    parameter_values=list(settings[job['image_ID']]['vlm'].values()),
                    retry=False,
                    succeed_async_id=None
                )
                if not success:
                    if not settings[job['image_ID']]['retry']:
                        metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']][str(job['detection_ID'])]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'stamp_error', metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'report_error', message))
                        progress[job['image_ID']][job['detection_ID']]['status'] = "ERROR"
                        validation_responses_error[job['image_ID']][job['detection_ID']].append(message)
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add system prompt
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text=settings[job['image_ID']]['system_prompt'],
                    role="system",
                    reset_context=True,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if not settings[job['image_ID']]['retry']:
                        metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']][str(job['detection_ID'])]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'stamp_error', metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'report_error', message))
                        progress[job['image_ID']][job['detection_ID']]['status'] = "ERROR"
                        validation_responses_error[job['image_ID']][job['detection_ID']].append(message)
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add image prompt
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text={"role": "user", "content": [{'type': "image_url", 'image_url': {'detail': settings[job['image_ID']]['image_resolution_full'], 'url': image_paths[job['image_ID']]}}]},
                    role="json",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if not settings[job['image_ID']]['retry']:
                        metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']][str(job['detection_ID'])]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'stamp_error', metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'report_error', message))
                        progress[job['image_ID']][job['detection_ID']]['status'] = "ERROR"
                        validation_responses_error[job['image_ID']][job['detection_ID']].append(message)
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add description prompt
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text=settings[job['image_ID']]['description_prompt'],
                    role="user",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if not settings[job['image_ID']]['retry']:
                        metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']][str(job['detection_ID'])]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'stamp_error', metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'report_error', message))
                        progress[job['image_ID']][job['detection_ID']]['status'] = "ERROR"
                        validation_responses_error[job['image_ID']][job['detection_ID']].append(message)
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add description response
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text=structured_descriptions_str[job['image_ID']],
                    role="assistant",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if not settings[job['image_ID']]['retry']:
                        metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']][str(job['detection_ID'])]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'stamp_error', metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'report_error', message))
                        progress[job['image_ID']][job['detection_ID']]['status'] = "ERROR"
                        validation_responses_error[job['image_ID']][job['detection_ID']].append(message)
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add crop prompt
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text={"role": "user", "content": [{'type': "image_url", 'image_url': {'detail': settings[job['image_ID']]['image_resolution_crop'], 'url': crops[job['image_ID']][job['detection_ID']]}}]},
                    role="json",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if not settings[job['image_ID']]['retry']:
                        metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']][str(job['detection_ID'])]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'stamp_error', metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'report_error', message))
                        progress[job['image_ID']][job['detection_ID']]['status'] = "ERROR"
                        validation_responses_error[job['image_ID']][job['detection_ID']].append(message)
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add validation prompt
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text=settings[job['image_ID']]['validation_prompt'].format(invalid=settings[job['image_ID']]['invalid']),
                    role="user",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="text",
                    identifier=job['UUID'],
                    retry=False,
                    succeed_async_id=async_id
                )
                if success:
                    progress[job['image_ID']][job['detection_ID']] = {'status': "GENERATING_VALIDATION", 'async_id': async_id, 'completions_ID': job['completions_ID']}
                else:
                    if not settings[job['image_ID']]['retry']:
                        metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']][str(job['detection_ID'])]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'stamp_error', metadata[job['image_ID']][str(job['detection_ID'])]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], str(job['detection_ID']), 'report_error', message))
                        progress[job['image_ID']][job['detection_ID']]['status'] = "ERROR"
                        validation_responses_error[job['image_ID']][job['detection_ID']].append(message)
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue

        # parse responses
        self._logger.debug(f"Parsing validation responses: {validation_responses_raw}")
        validation_responses = [[None for _ in bboxes[i]] for i in range(num_images)]
        instances = []
        for i in range(len(images)):
            instances.append([instance[identifiers[i]] for instance in structured_descriptions[i]])
            for j in range(len(bboxes[i])):
                if validation_responses_raw[i][j] is not None:
                    response, report = self._parse_validation_response(
                        response=validation_responses_raw[i][j],
                        identifier=identifiers[i],
                        targets=instances[i],
                        settings=settings[i]
                    )
                    if response is not None:
                        validation_responses[i][j] = response
                    validation_responses_error[i][j] += report

        # assign actions
        self._logger.debug(f"Assigning actions to validation responses: {validation_responses}")
        assignments = []
        for i, image in enumerate(image_paths_metadata):
            if len(images) > 1:
                self._logger.info(f"Assigning actions to validation responses for image '{i + 1}' of '{len(images)}' ('{image_paths[i]}')")
            keep, update, reject, error, actions = self._assign_actions(
                responses=validation_responses[i],
                instances=instances[i],
                identifier=identifiers[i],
                labels=labels[i],
                confidences=confidences[i],
                settings=settings[i]
            )
            labels_validated, durations = [], []
            for j in range(len(bboxes[i])):
                if validation_responses_raw[i][j] != validation_responses[i][j]:
                    metadata[i][str(j)]['validation_raw'] = validation_responses_raw[i][j]
                    self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'validation_raw', validation_responses_raw[i][j]))
                metadata[i][str(j)]['validation'] = validation_responses[i][j]
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'validation', validation_responses[i][j]))
                if len(validation_responses_error[i][j]) > 0:
                    metadata[i][str(j)]['report'] = validation_responses_error[i][j]
                    self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'report', validation_responses_error[i][j]))
                metadata[i][str(j)]['action'] = actions[j]
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'action', actions[j]))
                label_validated = None
                if j in keep:
                    label_validated = labels[i][j]
                elif str(j) in update:
                    label_validated = update[str(j)]
                labels_validated.append(label_validated)
                metadata[i][str(j)]['label_validated'] = label_validated
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'label_validated', label_validated))
                durations.append(metadata[i][str(j)].get('duration_validation'))
            durations_np = np.array(durations, dtype=float)
            assignments.append({
                'keep': keep,
                'update': update,
                'reject': reject,
                'error': error,
                'actions': actions,
                'labels': labels[i],
                'labels_validated': labels_validated,
                'duration_validation': durations,
                'duration_validation_min': float(np.nanmin(durations_np)),
                'duration_validation_max': float(np.nanmax(durations_np)),
                'duration_validation_std': float(np.nanstd(durations_np)),
                'duration_validation_mean': float(np.nanmean(durations_np)),
                'completion_errors': [None if len(validation_responses_error[i][j]) == 0 else validation_responses_error[i][j] for j in range(len(bboxes[i]))],
                'completion_responses_raw': [validation_responses_raw[i][j] for j in range(len(bboxes[i]))],
                'completion_responses': [validation_responses[i][j] for j in range(len(bboxes[i]))]
            })
            metadata[i]['summary'] = assignments[-1]
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'summary', assignments[-1]))

        # construct results
        labels_out, bboxes_out = [], []
        if confidences is None:
            confidences_out = None
        else:
            confidences_out = []
        if track_ids is None:
            track_ids_out = None
        else:
            track_ids_out = []
        if masks is None:
            masks_out = None
        else:
            masks_out = []
        for i in range(len(images)):
            labels_image, bboxes_image, confidences_image, track_ids_image, masks_image = [], [], [], [], []
            for j in range(len(bboxes[i])):
                if j in assignments[i]['keep']:
                    labels_image.append(labels[i][j])
                    bboxes_image.append(bboxes[i][j])
                    if confidences is not None:
                        confidences_image.append(confidences[i][j])
                    if track_ids is not None:
                        track_ids_image.append(track_ids[i][j])
                    if masks is not None:
                        mask = masks[i][j]
                        masks_image.append(masks[i][j])
                elif str(j) in assignments[i]['update']:
                    labels_image.append(assignments[i]['update'][str(j)])
                    bboxes_image.append(bboxes[i][j])
                    if confidences is not None:
                        confidences_image.append(confidences[i][j])
                    if track_ids is not None:
                        track_ids_image.append(track_ids[i][j])
                    if masks is not None:
                        masks_image.append(masks[i][j])
                elif j in assignments[i]['reject']:
                    pass
                elif j in assignments[i]['error']:
                    pass
                else:
                    assert False, (i, j, assignments[i]['actions'])
            labels_out.append(labels_image)
            bboxes_out.append(bboxes_image)
            if confidences is not None:
                confidences_out.append(confidences_image)
            if track_ids is not None:
                track_ids_out.append(track_ids_image)
            if masks is not None:
                masks_out.append(masks_image)

        # retrieve usage
        skip = all(settings[i]['usage_skip'] for i in range(len(settings)))
        if not skip:
            max_wait = max([settings[i]['usage_delay'] for i in range(len(settings))])
            if max_wait > 0.0:
                self._logger.debug(f"Sleeping '{max_wait}s' before requesting usage")
                time.sleep(max_wait)
            success, message, total_usage = self._node.api_director.get_usage(stamp_start=stamp_start)
            if success:
                history = total_usage.get('completions', {}).get('history', [])
                for i in range(len(labels)):
                    warnings_image = []
                    usage_image = {
                        'tokens_input_cached': 0,
                        'tokens_input_uncached': 0,
                        'tokens_input': 0,
                        'tokens_output': 0,
                        'dollars_input': 0.0,
                        'dollars_output': 0.0,
                        'dollars_total': 0.0
                    }
                    for j in range(len(labels[i])):
                        warnings_detection = []
                        usage_detection = []
                        for item in history:
                            if item.get('identifier') == uuids[i][j]:
                                usage_detection.append(item)
                        usage_formatted = {
                            'tokens_input_cached': 0,
                            'tokens_input_uncached': 0,
                            'tokens_input': 0,
                            'tokens_output': 0,
                            'dollars_input': 0.0,
                            'dollars_output': 0.0,
                            'dollars_total': 0.0
                        }
                        if len(usage_detection) == 0:
                            message = f"Cannot find usage for task 'validation' of detection '{j}' of image '{i}'."
                            warnings_detection.append(message)
                            self._logger.warn(message)
                        else:
                            for usage in usage_detection:
                                usage_formatted['tokens_input_cached'] += usage.get('tokens_input_cached', 0)
                                usage_formatted['tokens_input_uncached'] += usage.get('tokens_input_uncached', 0)
                                usage_formatted['tokens_output'] += usage.get('tokens_output', 0)
                                usage_formatted['dollars_input'] += usage.get('dollars_input', 0.0)
                                usage_formatted['dollars_output'] += usage.get('dollars_output', 0.0)
                                usage_formatted['dollars_total'] += usage.get('dollars_total', 0.0)
                            usage_formatted['tokens_input'] = usage_formatted['tokens_input_cached'] + usage_formatted['tokens_input_uncached']
                        for key in copy.deepcopy(usage_formatted):
                            if usage_formatted[key] == 0 or usage_formatted[key] == 0.0:
                                del usage_formatted[key]
                            else:
                                usage_image[key] += usage_formatted[key]
                        if len(warnings_detection) == 0:
                            if 'tokens_input_cached' not in usage_formatted and 'tokens_input_uncached' not in usage_formatted:
                                message = f"Cannot find input tokens for task 'validation' of detection '{j}' of image '{i}'."
                                warnings_detection.append(message)
                                self._logger.warn(message)
                            if 'tokens_output' not in usage_formatted:
                                message = f"Cannot find output tokens for task 'validation' of detection '{j}' of image '{i}'."
                                warnings_detection.append(message)
                                self._logger.warn(message)
                        if len(warnings_detection) > 0:
                            usage_formatted['warnings'] = warnings_detection
                            warnings_image.append(warnings_detection)
                        else:
                            warnings_image.append(None)
                        if usage_formatted != {}:
                            metadata[i][str(j)]['usage'] = usage_formatted
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, str(j), 'usage', metadata[i][str(j)]['usage']))

                    for key in copy.deepcopy(usage_image):
                        if usage_image[key] == 0 or usage_image[key] == 0.0:
                            del usage_image[key]
                    if not all(warning is None for warning in warnings_image):
                        usage_image['warnings'] = warnings_image
                    if usage_image != {}:
                        metadata[i]['summary']['usage'] = usage_image
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'summary', metadata[i]['summary']))

        # log
        status = [[item['status'] for item in sublist] for sublist in progress]
        info = [item for sublist in status for item in sublist]
        info_counter = dict(collections.Counter(info))
        assert set(info_counter.keys()) <= {"FINISHED", "ERROR"}, info
        if "ERROR" in info_counter:
            success = info_counter['ERROR'] < len(info)
            if len(info) > 1:
                message = f"Successfully processed '{len(info) - info_counter['ERROR']}' out of '{len(info)}' detections ('{info_counter['ERROR']}' error{'s' if info_counter['ERROR'] != 1 else ''})."
                self._logger.warn(message)
            else:
                message = "Failed to validate detection."
                self._logger.error(message)
        else:
            success = True
            if len(info) > 1:
                message = f"Successfully processed all '{len(info)}' detections."
            else:
                message = "Successfully processed detection."
            self._logger.info(message)
        stamp_end = datetime.datetime.now()
        for i, image in enumerate(image_paths_metadata):
            metadata[i]['validations_total'] = len(info)
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'validations_total', metadata[i]['validations_total']))
            metadata[i]['validations_finished'] = len(info) - info_counter.get('ERROR', 0)
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'validations_finished', metadata[i]['validations_finished']))
            metadata[i]['validations_errors'] = info_counter.get('ERROR', 0)
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'validations_errors', metadata[i]['validations_errors']))
            metadata[i]['stamp_batch_end'] = stamp_end.isoformat()
            metadata[i]['duration_batch'] = (stamp_end - stamp_start).total_seconds()
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'stamp_batch_end', metadata[i]['stamp_batch_end']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image, 'duration_batch', metadata[i]['duration_batch']))

        if no_batch:
            labels_out = labels_out[0]
            bboxes_out = bboxes_out[0]
            if confidences is not None:
                confidences_out = confidences_out[0]
            if masks is not None:
                masks_out = masks_out[0]
            if track_ids is not None:
                track_ids_out = track_ids_out[0]
            metadata = metadata[0]

        return success, message, labels_out, bboxes_out, confidences_out, masks_out, track_ids_out, metadata

    def _parse_validation_response(self, response, identifier, targets, settings):
        report = []

        # handle {'identifier': "actual response"}
        try:
            j = json.loads(response)
        except Exception:
            pass
        else:
            if isinstance(j, dict):
                if identifier in j:
                    report.append("Response is JSON that contains the identifier.")
                    response = j[identifier]

        # handle "identifier: actual response"
        response_replaced = response.replace(identifier, "")
        if response_replaced != response:
            report.append("Response contains the identifier.")
            response = response_replaced

        # TODO normalize

        # handle spelling errors, special characters, etc.
        response_matched = levenshtein_match(response, targets + [settings['invalid']], threshold=settings.get('levenshtein_threshold', 0), normalization=settings.get('levenshtein_normalization', True))
        if response_matched is None:
            report.append("Failed to match response to target.")
        elif response_matched != response:
            report.append("Matched response to target.")

        return response_matched, report

    def _assign_actions(self, responses, instances, identifier, labels, confidences, settings):
        # log

        self._logger.info(f"Descriptions: {instances}")
        self._logger.info(f"Detections:   {labels}")
        self._logger.info(f"Validations:  {responses}")

        duplicate_responses = dict(collections.Counter(responses))
        for response in copy.deepcopy(duplicate_responses):
            if duplicate_responses[response] == 1:
                del duplicate_responses[response]
        self._logger.info(f"Duplicate validations:  {duplicate_responses}")

        ### handle tracks

        keep = []
        reject = []
        update = {}
        error = []
        actions = {}

        used_labels = []
        handled_idx = []

        # group

        descriptions_stripped = [re.sub(r'[0-9]+$', '', description).rstrip() for description in instances]

        description_groups = {}

        for i in range(len(descriptions_stripped)):
            if descriptions_stripped[i] not in description_groups:
                description_groups[descriptions_stripped[i]] = []
            description_groups[descriptions_stripped[i]].append(instances[i])

        description_groups_inv = {}

        for group in copy.deepcopy(description_groups):
            if len(description_groups[group]) == 1:
                del description_groups[group]
            else:
                for label in description_groups[group]:
                    description_groups_inv[label] = group

        self._logger.info(f"Groups: {description_groups}")

        # reject completion errors and invalid responses

        for i, validation in enumerate(responses):
            if validation is None:
                error.append(i)
                actions[i] = f"Reject detection '{labels[i]}' after completion error."
                handled_idx.append(i)
            elif validation == settings['invalid']:
                reject.append(i)
                actions[i] = f"Reject invalidated detection '{labels[i]}'."
                handled_idx.append(i)

        # keep unique agreements and remember duplicate agreements

        duplicate_agreements = []

        for i, validation in enumerate(responses):
            if validation != labels[i]:
                continue

            mask_detection = np.array([validation == name for name in labels])
            mask_validation = np.array([validation == name for name in responses])
            agreement = np.logical_and(mask_detection, mask_validation)
            agreement_sum = np.sum(agreement)

            if agreement_sum == 1:
                keep.append(i)
                actions[i] = f"Accept unique validation of detection '{validation}'."
                used_labels.append(validation)
                handled_idx.append(i)
            elif agreement_sum > 1:
                duplicate_agreements.append(i)

        # sort duplicate agreements by descending confidence if available

        if confidences is not None:
            order = np.argsort([confidences[i] for i in duplicate_agreements])[::-1]
            duplicate_agreements = [duplicate_agreements[i] for i in order]

        # assign duplicate agreements

        for i in duplicate_agreements:
            if responses[i] not in used_labels:
                keep.append(i)
                actions[i] = f"Accept best duplicate validation of detection '{responses[i]}'."
                used_labels.append(responses[i])
                handled_idx.append(i)
            else:
                if responses[i] in description_groups_inv:
                    for alternative in description_groups[description_groups_inv[responses[i]]]:
                        if (alternative not in used_labels) and (alternative not in responses):
                            update[str(i)] = alternative
                            actions[i] = f"Update duplicate validation of detection '{responses[i]}' to unassigned group member '{alternative}' not featured in all validation responses."
                            used_labels.append(alternative)
                            handled_idx.append(i)
                            break
                    else:
                        for alternative in description_groups[description_groups_inv[responses[i]]]:
                            if alternative not in used_labels:
                                update[str(i)] = alternative
                                actions[i] = f"Update duplicate validation of detection '{responses[i]}' to unassigned group member '{alternative}' featured '{duplicate_responses[responses[i]]}' time{'s' if duplicate_responses[responses[i]] != 1 else ''} in all validation responses."
                                used_labels.append(alternative)
                                handled_idx.append(i)
                                break
                        else:
                            reject.append(i)
                            actions[i] = f"Reject duplicate validation of detection '{responses[i]}' for which all group members are already assigned."
                            handled_idx.append(i)
                else:
                    reject.append(i)
                    actions[i] = f"Reject duplicate validation of detection '{responses[i]}' that cannot be associated to a group."
                    handled_idx.append(i)

        not_handled = [i for i in range(len(responses)) if i not in handled_idx]

        # sort unassigned responses by descending confidence if available

        if confidences is not None:
            order = np.argsort([confidences[i] for i in not_handled])[::-1]
            not_handled = [not_handled[i] for i in order]

        # keep unassigned responses if possible

        for i in copy.deepcopy(not_handled):
            if responses[i] not in used_labels:
                update[str(i)] = responses[i]
                actions[i] = f"Update detection '{labels[i]}' to {'best' if responses[i] in duplicate_responses else ''} correction '{responses[i]}'."
                used_labels.append(responses[i])
                not_handled.remove(i)
                handled_idx.append(i)

        # validation unassigned responses

        for i in copy.deepcopy(not_handled):
            if responses[i] in description_groups_inv:
                if labels[i] in description_groups[description_groups_inv[responses[i]]] and labels[i] not in used_labels:
                    keep.append(i)
                    actions[i] = f"Accept detection label '{labels[i]}' which is an unassigned group member of group '{description_groups_inv[responses[i]]}' for validation response '{responses[i]}'."
                    used_labels.append(labels[i])
                else:
                    for alternative in description_groups[description_groups_inv[responses[i]]]:
                        if alternative not in used_labels:
                            update[str(i)] = alternative
                            actions[i] = f"Update detection '{labels[i]}' with already used validation response '{responses[i]}' to unassigned group member '{alternative}' not featured in all validation responses."
                            used_labels.append(alternative)
                            break
                    else:
                        reject.append(i)
                        actions[i] = f"Reject detection '{labels[i]}' with already used validation response '{responses[i]}' for which all group members of group '{description_groups_inv[responses[i]]}' are already assigned."
            else:
                reject.append(i)
                actions[i] = f"Reject detection '{labels[i]}' with already used validation response '{responses[i]}' that cannot be associated to a group."

            not_handled.remove(i)
            handled_idx.append(i)

        assert len(not_handled) == 0, f"{len(not_handled)}"
        assert len(handled_idx) == len(responses), f"{len(handled_idx)}"

        # log

        if len(keep) == 0:
            self._logger.error(f"Keep: {[labels[i] for i in keep]} ({len(keep)})")
        else:
            self._logger.info(f"Keep: {[labels[i] for i in keep]} ({len(keep)})")
        update_str = {labels[int(i)]: update[i] for i in update}
        if len(update) == 0:
            self._logger.info(f"Update: {update_str} ({len(update)})")
        else:
            self._logger.warn(f"Update: {update_str} ({len(update)})")
        if len(reject) == 0:
            self._logger.info(f"Reject: {[labels[i] for i in reject]} ({len(reject)})")
        else:
            self._logger.warn(f"Reject: {[labels[i] for i in reject]} ({len(reject)})")
        if len(error) == 0:
            self._logger.info(f"Error: {[labels[i] for i in error]} ({len(error)})")
        else:
            self._logger.error(f"Error: {[labels[i] for i in error]} ({len(error)})")
        actions = [actions[i] for i in range(len(actions))]
        self._logger.info(f"Actions: {actions}")

        return keep, update, reject, error, actions
