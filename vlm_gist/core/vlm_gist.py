#!/usr/bin/env python3

# STANDARD

import os
import copy
import random
import datetime
from queue import Queue

import numpy as np

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.lazy import StructuredDescription, Detection, Validation
from vlm_gist.core.common import read_image, set_logger, set_settings, complete_settings, save_metadata

from nimbro_utils.lazy import encode_mask, get_package_path

default_settings = {
    'skip_validation': False,
    'leave_images_in_place': False, # prevents copying images passed as string (local path or web)
    'keep_image_name': False, # keep original image name when copying instead of using a timestamp based name
    'png_compression_level': 2, # 0 (off) to 9 (max)
    'png_max_pixels': 1920 * 1080, # set None to deactivate re-scaling
    'data_path': os.path.join(get_package_path("vlm_gist"), "data", "vlm_detections"),
    'image_folder': "vlm_gist_edits", # folder within data_path to store images; set None to use data_path directly
    'metadata_file': "vlm_gist.json",
    'metadata_write_relative_paths': False, # write relative path between metadata and image instead of absolute image paths to metadata
    'metadata_write_no_paths': True # only write image names to metadata; if True this overwrites 'metadata_write_relative_paths'
}

class VlmGist:

    def __init__(self, node, parallel_completions=None, metadata_timer=True, settings=None, logger_name=None, logger_severity=20):
        assert isinstance(node, Node), f"Provided argument 'node' is of invalid type '{type(node).__name__}'. Supported type is 'rclpy.node.Node'."
        assert parallel_completions is None or (isinstance(parallel_completions, int) and parallel_completions > 0), f"Provided argument 'parallel_completions' is of invalid type '{type(parallel_completions).__name__}'. Supported types are 'None' and 'int > 0'."
        assert isinstance(metadata_timer, bool), f"Provided argument 'metadata_timer' is of unsupported type '{type(metadata_timer).__name__}'. Supported type is 'bool'."
        assert logger_name is None or isinstance(logger_name, str), f"Provided argument 'logger_name' is of unsupported type '{type(logger_name).__name__}'. Supported types are 'None' and 'str'."

        self._node = node

        # logger
        if logger_name is None:
            logger_name = (f"{self._node.get_namespace()}.{self._node.get_name()}.vlm_gist").replace("/", "")
            if logger_name[0] == ".":
                logger_name = logger_name[1:]
        self.set_logger = set_logger.__get__(self)
        success, message = self.set_logger(logger_name, logger_severity)
        assert success, message

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

        # pipeline
        self.structured_description = StructuredDescription(
            node=self._node,
            parallel_completions=parallel_completions,
            logger_severity=logger_severity
        )
        self.detection = Detection(
            node=self._node,
            logger_severity=logger_severity
        )
        self.validation = Validation(
            node=self._node,
            parallel_completions=parallel_completions,
            logger_severity=logger_severity
        )

    def __del__(self):
        del self.structured_description
        del self.detection
        del self.validation

    def get(self, image, settings=None):
        stamp_start = datetime.datetime.now()

        key_object_name = 'object_name'
        key_description = 'description'

        # read settings
        settings = copy.deepcopy(settings)
        if settings is None:
            settings = copy.deepcopy(self._settings)
        elif isinstance(settings, dict):
            success, message, settings = self.complete_settings(settings)
            if not success:
                return False, message, None, None, None, None, None, None
        else:
            message = f"Provided argument 'settings' is of unsupported type '{type(settings).__name__}'. Supported types are 'None' and 'dict'."
            return False, message, None, None, None, None, None, None

        # read images
        if not isinstance(image, str):
            message = f"Provided argument 'image' is of unsupported type '{type(image).__name__}'. Supported types is 'str'."
            return False, message, None, None, None, None, None, None
        num_images = 1
        image_folder = settings['data_path'] if settings['image_folder'] is None else os.path.join(settings['data_path'], settings['image_folder'])
        success, message, _, image = read_image(
            self,
            image=image,
            data_path=image_folder,
            leave_in_place=settings['leave_images_in_place'],
            keep_name=settings['keep_image_name'],
            png_compression_level=settings['png_compression_level'],
            png_max_pixels=settings['png_max_pixels']
        )
        if not success:
            return False, message, None, None, None, None, None, None
        if settings['metadata_write_no_paths']:
            image_path_metadata = os.path.basename(image)
        elif settings['metadata_write_relative_paths']:
            image_path_metadata = os.path.relpath(image, settings['data_path'])

        # run pipeline

        metadata = {}
        metadata['stamp_start'] = stamp_start.isoformat()
        self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'stamp_start', metadata['stamp_start']))

        metadata['settings'] = settings
        self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'settings', metadata['settings']))

        self._logger.info(f"Processing '{num_images}' image{'s' if num_images != 1 else ''}")

        # obtain structured description
        success, message, structured_description, scene_description, metadata['description'] = self.structured_description.get(
            images=image,
            settings={
                'leave_images_in_place': True,
                'keep_image_name': settings['keep_image_name'],
                'png_compression_level': settings['png_compression_level'],
                'png_max_pixels': settings['png_max_pixels'],
                'data_path': settings['data_path'],
                'metadata_file': None,
                'metadata_write_relative_paths': settings['metadata_write_relative_paths'],
                'metadata_write_no_paths': settings['metadata_write_no_paths'],
            }
        )
        self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'description', metadata['description']))
        self.structured_description.release_completions()
        if not success:
            return False, message, None, None, None, None, None, metadata

        if len(structured_description) == 0:
            instance_descriptions, bboxes, confidences, masks, track_ids = [], [], [], [], []
        else:
            # detect described instances
            prompts = [structured_description[i][key_description] for i in range(len(structured_description))]
            success, message, prompts, bboxes, confidences, masks, track_ids, metadata['detection'] = self.detection.get(
                images=image,
                prompts=prompts,
                settings={
                    'leave_images_in_place': True,
                    'keep_image_name': settings['keep_image_name'],
                    'png_compression_level': settings['png_compression_level'],
                    'png_max_pixels': settings['png_max_pixels'],
                    'data_path': settings['data_path'],
                    'metadata_file': None,
                    'metadata_write_relative_paths': settings['metadata_write_relative_paths'],
                    'metadata_write_no_paths': settings['metadata_write_no_paths'],
                }
            )
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'detection', metadata['detection']))
            if not success:
                return False, message, None, None, None, None, None, metadata

            if len(bboxes) == 0:
                instance_descriptions, bboxes, confidences, masks, track_ids = [], [], [], [], []
            elif settings['skip_validation']:
                # collect full dictionaries of detections
                instance_descriptions = []
                for prompt in prompts:
                    candidates = []
                    for instance in structured_description:
                        if instance[key_description] == prompt:
                            candidates.append(instance)
                    if len(candidates) > 1:
                        self._logger.warn(f"The association between detections and descriptions is ambiguous since the description is featured '{len(candidates)}' times")
                        instance = candidates[random.randrange(len(candidates))]
                    else:
                        instance = candidates[0]
                    instance_descriptions.append(instance)
            else:
                # collect names of described instances
                # this assumes that object_name is unique while choosing at random if description is not, i.e. the mapping from description to object_name is ambiguous
                name_to_description = {item[key_object_name]: item[key_description] for item in structured_description}
                description_to_names = {}
                for name in name_to_description:
                    if name_to_description[name] in description_to_names:
                        description_to_names[name_to_description[name]].append(name)
                    else:
                        description_to_names[name_to_description[name]] = [name]
                labels = [description_to_names[prompt][random.randrange(len(description_to_names[prompt]))] for prompt in prompts]

                # validate detections based on their name
                success, message, labels, bboxes, confidences, masks, track_ids, metadata['validation'] = self.validation.get(
                    images=image,
                    structured_descriptions=structured_description,
                    identifiers=key_object_name,
                    labels=labels,
                    bboxes=bboxes,
                    confidences=confidences,
                    masks=masks,
                    track_ids=track_ids,
                    settings={
                        'leave_images_in_place': True,
                        'keep_image_name': settings['keep_image_name'],
                        'png_compression_level': settings['png_compression_level'],
                        'png_max_pixels': settings['png_max_pixels'],
                        'data_path': settings['data_path'],
                        'metadata_file': None,
                        'metadata_write_relative_paths': settings['metadata_write_relative_paths'],
                        'metadata_write_no_paths': settings['metadata_write_no_paths'],
                    }
                )
                self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'validation', metadata['validation']))
                self.validation.release_completions()
                if not success:
                    return False, message, None, None, None, None, None, metadata

                # collect full dictionaries of validated detections
                instance_descriptions = []
                for label in labels:
                    for instance in structured_description:
                        if instance[key_object_name] == label:
                            instance_descriptions.append(instance)
                            break
                    else:
                        assert False, f"Failed to find instance description for detector prompt '{label}'"

        # collect results
        if scene_description is not None:
            metadata['scene_description'] = scene_description
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'scene_description', metadata['scene_description']))
        metadata['instance_descriptions'] = instance_descriptions
        self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'instance_descriptions', metadata['instance_descriptions']))
        metadata['bboxes'] = bboxes
        self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'bboxes', metadata['bboxes']))
        metadata['confidences'] = confidences
        self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'confidences', metadata['confidences']))
        metadata['masks'] = [encode_mask(np.array(mask) > 0) for mask in masks] # TODO take from metadata
        self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'masks', metadata['masks']))
        metadata['track_ids'] = track_ids
        self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'track_ids', metadata['track_ids']))

        # collect usage
        usage_formatted = {
            'tokens_input_cached': 0,
            'tokens_input_uncached': 0,
            'tokens_input': 0,
            'tokens_output': 0,
            'dollars_input': 0.0,
            'dollars_output': 0.0,
            'dollars_total': 0.0
        }
        usages = [metadata.get('description', {}).get('usage', {}).get('total', {}), metadata.get('validation', {}).get('summary', {}).get('usage', {})]
        for usage in usages:
            for key in usage_formatted:
                usage_formatted[key] += usage.get(key, 0)
        for key in copy.deepcopy(usage_formatted):
            if usage_formatted[key] == 0 or usage_formatted[key] == 0.0:
                del usage_formatted[key]
        warnings = metadata.get('description', {}).get('usage', {}).get('warnings', [])
        for warnings_image in metadata.get('validation', {}).get('summary', {}).get('usage', {}).get('warnings', []):
            if isinstance(warnings_image, list):
                for warning in warnings_image:
                    warnings.append(warning)
        if len(warnings) > 0:
            usage_formatted['warnings'] = warnings
        if usage_formatted != {}:
            metadata['usage'] = usage_formatted
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'usage', metadata['usage']))

        # log
        stamp_end = datetime.datetime.now()
        metadata['stamp_end'] = stamp_end.isoformat()
        self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'stamp_end', metadata['stamp_end']))
        metadata['duration'] = (stamp_end - stamp_start).total_seconds()
        self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], image_path_metadata, 'duration', metadata['duration']))
        message = f"Successfully detected{'' if settings['skip_validation'] else ' & validated'} '{len(instance_descriptions)}' object instance{'s' if len(instance_descriptions) != 1 else ''} in '{metadata['duration']:.3f}s'."
        if len(instance_descriptions) == 0:
            self._logger.warn(message)
        else:
            self._logger.info(message)

        return success, message, instance_descriptions, bboxes, confidences, masks, track_ids, metadata
