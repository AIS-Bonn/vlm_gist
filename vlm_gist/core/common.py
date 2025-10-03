#!/usr/bin/env python3

# STANDARD

import os
import copy
import json
import datetime

import cv2
import numpy as np

# ROS

import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# CUSTOM

from nimbro_utils.lazy import download_image, read_json, write_json, get_package_path

test_image_path = os.path.join(get_package_path("vlm_gist"), "data/datasets/vlm_gist/data/00009.jpg")

def read_image(self, image, data_path, leave_in_place, keep_name, png_compression_level, png_max_pixels):
    image_path = None
    image_name = None
    if isinstance(image, Image):
        if hasattr(self._node, 'cv_bridge'):
            if not isinstance(self._node.cv_bridge, CvBridge):
                message = f"Cannot convert Image message because the parent node features an attribute 'cv_bridge' of type '{type(self._node.cv_bridge).__name__}' instead of 'cv_bridge.CvBridge'."
                self._logger.error(message)
                return False, message, None, None
        else:
            self._logger.debug("Adding 'CvBridge' to parent node")
            self._node.cv_bridge = CvBridge()

        self._logger.debug("Converting Image message to NumPy image")
        image = self._node.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
    elif isinstance(image, str):
        if leave_in_place:
            image_path = image
        if os.path.isfile(image):
            if keep_name:
                image_name = os.path.basename(os.path.normpath(image))
            self._logger.debug(f"Reading image from file '{image}'")
            try:
                image = cv2.imread(image, cv2.IMREAD_COLOR)
            except Exception as e:
                message = f"Failed to read image from path '{image}': {repr(e)}"
                self._logger.error(message)
                return False, message, None, None
        else:
            self._logger.debug(f"Interpreting '{image}' as web address since it is not a valid file path")
            success, message, image = download_image(url=image, rgb=False, retry=1, logger=self._logger)
            if not success:
                return False, message, None, None
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        message = f"Provided argument 'image' is of unsupported type '{type(image).__name__}'. Supported types are 'str (path)', 'np.ndarray', and 'sensor_msgs.msg.Image'."
        self._logger.error(message)
        return False, message, None, None

    # check validity
    if not len(image.shape) == 3:
        message = f"Provided image has invalid shape {image.shape}, which must be 3-dimensional."
        self._logger.error(message)
        return False, message, None, None
    elif image.shape[2] != 3:
        message = f"Provided image has invalid number of channels '{image.shape[2]}', which must be '3'."
        self._logger.error(message)
        return False, message, None, None
    elif image.dtype != np.uint8:
        message = f"Provided image has invalid datatype '{image.dtype}', which must be 'np.uint8'."
        self._logger.error(message)
        return False, message, None, None

    # re-scale
    if isinstance(png_max_pixels, int):
        image_pixels = image.shape[0] * image.shape[1]
        if image_pixels > png_max_pixels:
            ratio = np.sqrt(png_max_pixels / image_pixels)
            old_shape = image.shape
            image = cv2.resize(image, (int(ratio * image.shape[1]), int(ratio * image.shape[0])), interpolation=cv2.INTER_AREA)
            self._logger.debug(f"Re-scaled image with shape {old_shape} by factor '{ratio:.3f}' to shape {image.shape} ensuring the maximum number of pixels '{png_max_pixels}'")
            if image_path is not None and leave_in_place:
                self._logger.warn(f"Copying image '{image_path}' after re-scaling")
                image_path = None

    # write
    if image_path is None:
        if not os.path.isdir(data_path):
            self._logger.debug(f"Creating data folder '{data_path}'")
            try:
                os.makedirs(data_path)
            except Exception as e:
                message = f"Failed to create data folder '{data_path}': {repr(e)}"
                self._logger.error(message)
                return False, message, None, None

        if not keep_name:
            image_name = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]}.png"
        image_path = os.path.join(data_path, image_name)

        if os.path.exists(image_path):
            message = f"Cannot write image '{image_path}' because the file already exists."
            self._logger.error(message)
            return False, message, None, None

        self._logger.debug(f"Writing image '{image_path}' with shape {image.shape} of type '{image.dtype}' in range {[np.min(image), np.max(image)]}")

        try:
            cv2.imwrite(image_path, image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression_level])
        except Exception as e:
            message = f"Failed to save image '{image_path}': {repr(e)}"
            self._logger.error(message)
            return False, message, None, None

    return True, "Successfully read image.", image, image_path

def set_logger(self, name=None, severity=None):
    if name is not None and not isinstance(name, str):
        message = f"Provided argument 'name' is of invalid type '{type(name).__name__}'. Supported types are 'None' and 'str'."
        self._logger.error(message)
        return False, message, self._settings

    if name is not None:
        assert isinstance(name, str), f"Expected 'name' to be of type 'str' but it is of type '{type(name).__name__}'!"
        self._logger = rclpy.logging.get_logger(name)
        self._logger.debug(f"Set logger name to '{name}'")
        message = "Successfully updated logger name."

    if severity is not None and severity not in [10, 20, 30, 40, 50]:
        message = f"Provided argument 'severity' is of invalid type '{type(severity).__name__}'. Supported types are 'None' and 'int in [10, 20, 30, 40, 50]'."
        self._logger.error(message)
        return False, message, self._settings

    if severity is not None:
        assert severity in [10, 20, 30, 40, 50], "Expected 'severity' to be in [10, 20, 30, 40, 50] but it is not!"
        rclpy.logging.set_logger_level(self._logger.name, rclpy.logging.LoggingSeverity(severity))
        self._logger.debug(f"Set logger severity to '{severity}'")
        message = "Successfully updated logger severity."

    if name is not None and severity is not None:
        message = "Successfully updated logger name and severity."

    return True, message

def set_settings(self, settings):
    if hasattr(self, '_settings'):
        init = False
    else:
        init = True
    if settings is None:
        self._settings = copy.deepcopy(self._default_settings)
    elif not isinstance(settings, dict):
        message = f"Provided argument 'settings' is of invalid type '{type(settings).__name__}'. Supported types are 'None' and 'dict'."
        self._logger.error(message)
        return False, message, None
    else:
        success, message, completed_Settings = self.complete_settings(settings)
        if success:
            self._settings = completed_Settings
        else:
            return False, message, None

    if init:
        self._logger.debug(f"Initialized settings:\n{json.dumps(self._settings, indent=4)}")
        return True, "Successfully initialized settings.", copy.deepcopy(self._settings)
    else:
        self._logger.debug(f"Updated settings:\n{json.dumps(self._settings, indent=4)}")
        return True, "Successfully updated settings.", copy.deepcopy(self._settings)

def complete_settings(self, settings):
    if not isinstance(settings, dict):
        message = f"Provided argument 'settings' is of invalid type '{type(settings).__name__}'. Supported type is 'dict'."
        self._logger.error(message)
        return False, message, None

    if hasattr(self, '_settings'):
        completed = copy.deepcopy(self._settings)
    else:
        completed = copy.deepcopy(self._default_settings)

    for key in settings:
        if key not in list(self._default_settings.keys()):
            message = f"The name '{key}' does not refer to a valid setting. Valid settings are in {list(self._default_settings.keys())}."
            self._logger.error(message)
            return False, message, None
        completed[key] = settings[key]

    return True, "Successfully completed settings.", completed

def release_completions(self):
    r, f = 0, 0
    self._status_lock.acquire()
    for i in range(len(self._completions)):
        if self._completions[i]['status'] != "NOT_ACQUIRED":
            self._node.api_director.interrupt(self._completions[i]['ID'], retry=False)
            success, _ = self._node.api_director.release(self._completions[i]['ID'], retry=False)
            if success:
                self._completions[i] = {'ID': None, 'status': "NOT_ACQUIRED"}
                r += 1
            else:
                f += 1
    self._status_lock.release()
    if r > 0:
        self._logger.debug(f"Released '{r}' completion{'s' if r != 1 else ''}")
    if f > 0:
        self._logger.warn(f"Failed to released '{f}' completion{'s' if f != 1 else ''}")

def format_seconds(total_seconds):
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02}hours"
    elif minutes > 0:
        return f"{minutes}:{seconds:02}min"
    else:
        return f"{total_seconds:.1f}s"

def save_metadata(self):
    self._logger.debug("save_metadata callback triggered")
    data_tuples = []
    file_paths = []
    while not self._metadata_queue.empty():
        data_tuple = self._metadata_queue.get_nowait()
        if data_tuple[1] is None:
            continue
        else:
            data_tuples.append(data_tuple)
            file_paths.append(os.path.join(data_tuple[0], data_tuple[1]))
    if len(data_tuples) == 0:
        self._logger.debug("save_metadata callback returning")
        return

    skip_files = []
    metadata_before = {}
    for file_path in set(file_paths):
        # create folder if required
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            self._logger.debug(f"Creating data folder '{folder_path}'")
            try:
                os.makedirs(folder_path)
            except Exception as e:
                self._logger.error(f"Cannot save results to '{file_path}': Failed to create data folder '{folder_path}': {repr(e)}")
                skip_files.append(file_path)
                continue
        # read metadata
        if os.path.isfile(file_path):
            success, message, metadata = read_json(file_path, logger=self._logger)
            if success:
                if isinstance(metadata, dict):
                    metadata_before[file_path] = copy.deepcopy(metadata)
                else:
                    skip_files.append(file_path)
                    self._logger.warn(f"Cannot save results to '{file_path}': Expected content of file to be of type 'dict' but it is of type '{type(metadata).__name__}'.")

    write_files = set()
    metadata = copy.deepcopy(metadata_before)

    for i, data_tuple in enumerate(data_tuples):
        if file_paths[i] in skip_files:
            continue

        if file_paths[i] not in metadata:
            metadata[file_paths[i]] = {}

        if len(data_tuple) == 5:
            _, _, image, name, data = data_tuple
            if not isinstance(image, str):
                if isinstance(image, np.ndarray):
                    self._logger.warn(f"{image.shape}, {name}, {data}")
                else:
                    self._logger.warn(f"{type(image).__name__}, {name}, {data}")
                continue

            if image not in metadata[file_paths[i]]:
                write_files.add(file_paths[i])
                metadata[file_paths[i]][image] = {name: data}
            elif not isinstance(metadata[file_paths[i]][image], dict):
                self._logger.warn(f"Cannot save results to '{file_paths[i]}': Expected value of key '{image}' to be of type 'dict' but it is of type '{type(metadata[file_paths[i]][image]).__name__}'.")
            else:
                write_files.add(file_paths[i])
                metadata[file_paths[i]][image][name] = data
        elif len(data_tuple) == 6:
            _, _, image, detection, name, data = data_tuple
            if detection is None:
                if image not in metadata[file_paths[i]]:
                    write_files.add(file_paths[i])
                    metadata[file_paths[i]][image] = {name: data}
                elif not isinstance(metadata[file_paths[i]][image], dict):
                    self._logger.warn(f"Cannot save results to '{file_paths[i]}': Expected value of key '{image}' to be of type 'dict' but it is of type '{type(metadata[file_paths[i]][image]).__name__}'.")
                else:
                    write_files.add(file_paths[i])
                    metadata[file_paths[i]][image][name] = data
            else:
                if image not in metadata[file_paths[i]]:
                    write_files.add(file_paths[i])
                    metadata[file_paths[i]][image] = {detection: {name: data}}
                elif not isinstance(metadata[file_paths[i]][image], dict):
                    self._logger.warn(f"Cannot save results to '{file_paths[i]}': Expected value of key '{image}' to be of type 'dict' but it is of type '{type(metadata[file_paths[i]][image]).__name__}'.")
                elif detection not in metadata[file_paths[i]][image]:
                    write_files.add(file_paths[i])
                    metadata[file_paths[i]][image][detection] = {name: data}
                elif not isinstance(metadata[file_paths[i]][image][detection], dict):
                    self._logger.warn(f"Cannot save results to '{file_paths[i]}': Expected value of key detection '{detection}' of '{image}' to be of type 'dict' but it is of type '{type(metadata[file_paths[i]][image][detection]).__name__}'.")
                else:
                    write_files.add(file_paths[i])
                    metadata[file_paths[i]][image][detection][name] = data
        else:
            self._logger.warn(f"Cannot save results to '{file_paths[i]}': Expected data in queue to be 3-tuple or 4-tuple.")

    if len(write_files) == 0:
        self._logger.debug("Writing metadata is not required (no file requires update)")
    for file_path in write_files:
        if self._metadata_timer.is_canceled():
            self._logger.info(f"Writing metadata to '{file_path}'")
        success, message = write_json(file_path, metadata[file_path], indent=True, name="metadata file", logger=self._logger)
        if not success:
            if self._metadata_timer.is_canceled():
                try:
                    import pickle
                    with open(f"{file_path}.pkl", "wb") as f:
                        pickle.dump(metadata[file_path], f)
                except Exception as e:
                    self._logger.error(f"Failed to pickle metadata: {repr(e)}")
            if file_path in metadata_before:
                self._logger.info(f"Restoring original metadata file after '{file_path}'")
                write_json(file_path, metadata_before[file_path], indent=True, name="old metadata file", logger=self._logger)
            else:
                self._logger.debug(f"Restoring original metadata file '{file_path}' not required")
    self._logger.debug("save_metadata callback returning")
