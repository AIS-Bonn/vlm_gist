#!/usr/bin/env python3

# STANDARD

import os
import copy
import json
import time
import uuid
import datetime
import threading
import collections
import multiprocessing
from queue import Queue

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.core.common import read_image, set_logger, set_settings, complete_settings, release_completions, format_seconds, save_metadata

from nimbro_api import ApiDirector
from nimbro_utils.lazy import normalize_string, levenshtein_match, get_package_path

default_settings = {
    'vlm_scene_description': {
        'probe_api_connection': "False",
        'api_endpoint': "OpenRouter",
        'model_name': "google/gemini-2.5-flash",
        'model_temperature': "1.0",
        'model_top_p': "1.0",
        'model_max_tokens': "5000",
        'model_presence_penalty': "0.0",
        'model_frequency_penalty': "0.0",
        'model_reasoning_effort': "none",
        'completion_parsers': "[]",
        'stream_completion': "True",
        'normalize_text_completion': "False",
        'correction_attempts': "0",
        'timeout_chunk_first': "10.0",
        'timeout_chunk_next': "5.0",
        'timeout_completion': "30.0"
    },
    'vlm_structured_description': {
        'probe_api_connection': "False",
        'api_endpoint': "OpenRouter",
        'model_name': "google/gemini-2.5-flash",
        'model_temperature': "1.0",
        'model_top_p': "1.0",
        'model_max_tokens': "5000",
        'model_presence_penalty': "0.0",
        'model_frequency_penalty': "0.0",
        'model_reasoning_effort': "none",
        'completion_parsers': "[]",
        'stream_completion': "True",
        'normalize_text_completion': "False",
        'correction_attempts': "0",
        'timeout_chunk_first': "10.0",
        'timeout_chunk_next': "5.0",
        'timeout_completion': "30.0"
    },
    'vlm_decoupled_attribution': {
        'probe_api_connection': "False",
        'api_endpoint': "OpenRouter",
        'model_name': "google/gemini-2.5-flash",
        'model_temperature': "1.0",
        'model_top_p': "1.0",
        'model_max_tokens': "5000",
        'model_presence_penalty': "0.0",
        'model_frequency_penalty': "0.0",
        'model_reasoning_effort': "none",
        'completion_parsers': "[]",
        'stream_completion': "True",
        'normalize_text_completion': "False",
        'correction_attempts': "0",
        'timeout_chunk_first': "10.0",
        'timeout_chunk_next': "5.0",
        'timeout_completion': "30.0"
    },
    'system_prompt_scene_description': "You are a robot's visual perception system that identifies and analyzes objects in an image. Be concise and factual.",
    'system_prompt_scene_description_role': "system",
    'system_prompt_structured_description': "You are a robot's visual perception system that identifies and analyzes objects in an image. Be concise and factual.",
    'system_prompt_structured_description_role': "system",
    'system_prompt_decoupled_attribution': "You are a robot's visual perception system that identifies and analyzes objects in an image. Be concise and factual.",
    'system_prompt_decoupled_attribution_role': "system",
    'image_resolution_scene_description': ["high", "low"][0],
    'image_resolution_structured_description': ["high", "low"][0],
    'image_resolution_decoupled_attribution': ["high", "low"][0],
    # 'scene_prompt': "Please describe the content of the image above. "\
    #                 "Focus your your description on all visible objects. "\
    #                 "Be concise and answer with at most 10 sentences. "\
    #                 "If you are unsure about identifying an object, make one single guess rather than calling it an unknown object or discussing eventualities.",
    'scene_prompt': None,
    'task': None,
    'dict_prompt': "Provide a list in JSON format that contains each object (including furniture, persons, and animals) visible in the image above. "\
                   "Explicitly include each object instance as an individual list element, and never group multiple instances that are clearly distinct from one another. "\
                   "Each list element must be a dictionary with the fields object_name and description. "\
                   "The object_name of all humans must be person."
                   "The description must be a single short sentence (max. 10 words, starting with 'a' or 'an'), "\
                   "that differs from the other descriptions and summarizes the most important information about the type, color, and appearance of the object, "\
                   "allowing for a visual identification of the object without knowing any of the descriptions generated for the other objects.",
    'direct_task_relevancy_prompt': " In addition, examine for each object whether it is relevant for accomplishing the following task: '{task}'"\
                                    "For each list element, add a boolean field task_relevant accordingly. "\
                                    "Do not consider objects that are similar or loosely related to a task-relevant object to be task-relevant. "\
                                    "If you are unsure, it is most helpful to consider it not task-relevant.",
    'decoupled_task_relevancy_prompt': "Examine for each listed object whether it is relevant for accomplishing the following task: '{task}'"\
                                       "Respond with a list encoded as JSON featuring the 'object_name' value of all task-relevant objects from the dictionary you provided above.",
    'decoupled_task_relevancy': True,
    'decoupled_task_relevancy_remove_image': True,
    'levenshtein_threshold': 0, # floats between zero and one dynamically set threshold relative to work length (ceiled)
    'levenshtein_normalization': True,
    'dict_items': [ # first two elements must be object name and description (required); 'valid' and 'invalid' can be deactivated if set to None, 'invalid' is checked first; if 'type' is 'str' Levenshtein matching is used for value while 'key' is always Levenshtein matched
        {
            'key': 'object_name',
            'type': "str",
            'valid': None,
            'invalid': ["", "none", "not applicable", "n/a"],
            'requires': None,
            'required': True
        },
        {
            'key': 'description',
            'type': "str",
            'valid': None,
            'invalid': ["", "none", "not applicable", "n/a"],
            'requires': None,
            'required': True
        }
    ],
    'dict_direct_task_relevancy_item': {
        'key': 'task_relevant',
        'type': "bool",
        'valid': None,
        'invalid': None,
        'requires': None,
        'required': True
    },
    'dict_value_normalization': {
        'remove_underscores': True,
        'remove_punctuation': False,
        'remove_common_specials': True,
        'remove_whitespaces': False,
        'lowercase': False
    },
    'use_json_mode': True,
    'retry': 3, # retry API requests after failure instead of skipping the current image. Use -1 to retry forever.
    'timeout': None, # set None to deactivate or set number of seconds as int or float in which get() must finish
    'leave_images_in_place': False, # attempt to not copy images passed as string (local path or web)
    'keep_image_name': False, # keep original image name when copying instead of using a timestamp based name
    'png_compression_level': 2, # 0 (off) to 9 (max)
    'png_max_pixels': 1920 * 1080, # set None to deactivate re-scaling
    'usage_skip': False, # set True to not obtain usage information (tokens & dollars); Only skipped if entire batch wants to skip
    'usage_delay': 1.0, # time in seconds to sleep before requesting usage to ensure was captured; Maximum of batch is used
    'data_path': os.path.join(get_package_path("vlm_gist"), "data", "descriptions"),
    'image_folder': "description_edits", # folder within data_path to store images; set None to use data_path directly
    'metadata_file': "descriptions.json", # set None to deactivate
    'metadata_write_relative_paths': False, # write relative path between metadata and image instead of absolute image paths to metadata
    'metadata_write_no_paths': True # only write image names to metadata; if True this overwrites 'metadata_write_relative_paths'
}

class StructuredDescription:

    def __init__(self, node, parallel_completions=None, metadata_timer=True, settings=None, logger_name=None, logger_severity=20):
        assert isinstance(node, Node), f"Provided argument 'node' is of invalid type '{type(node).__name__}'. Supported type is 'rclpy.node.Node'."
        assert parallel_completions is None or (isinstance(parallel_completions, int) and parallel_completions > 0), f"Provided argument 'parallel_completions' is of invalid type '{type(parallel_completions).__name__}'. Supported types are 'None' and 'int > 0'."
        assert isinstance(metadata_timer, bool), f"Provided argument 'metadata_timer' is of unsupported type '{type(metadata_timer).__name__}'. Supported type is 'bool'."
        assert logger_name is None or isinstance(logger_name, str), f"Provided argument 'logger_name' is of unsupported type '{type(logger_name).__name__}'. Supported types are 'None' and 'str'."

        self._node = node

        # logger
        if logger_name is None:
            logger_name = (f"{self._node.get_namespace()}.{self._node.get_name()}.structured_description").replace("/", "")
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

    def get(self, images, settings=None):
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
            return False, message, None, None, None
        elif not len(settings) == num_images:
            message = f"Expected number of settings '{len(settings)}' to match the number of images '{num_images}'."
            self._logger.error(message)
            return False, message, None, None, None
        for i in range(len(settings)):
            if settings[i] is None:
                settings[i] = copy.deepcopy(self._settings)
            else:
                success, message, settings[i] = self.complete_settings(settings[i])
                if not success:
                    return False, message, None, None, None
        timeout_idx = [i for i in range(num_images) if settings[i]['timeout'] is not None]
        self._logger.debug(f"Image IDs that require timeout: {timeout_idx}")
        retries = [0 for _ in range(num_images)]

        for settings_image in settings:
            if settings_image.get('scene_prompt') is None and settings_image.get('dict_prompt') is None:
                message = "Setting 'scene_prompt' and setting 'dict_prompt' cannot both be None."
                self._logger.error(message)
                return False, message, None, None, None

        # read images
        images = copy.deepcopy(images)
        image_paths_metadata = []
        no_batch = False
        if not isinstance(images, list):
            no_batch = True
            images = [images]
        for i in range(len(images)):
            image_folder = settings[i]['data_path'] if settings[i]['image_folder'] is None else os.path.join(settings[i]['data_path'], settings[i]['image_folder'])
            success, message, _, path = read_image(
                self,
                image=images[i],
                data_path=image_folder,
                leave_in_place=settings[i]['leave_images_in_place'],
                keep_name=settings[i]['keep_image_name'],
                png_compression_level=settings[i]['png_compression_level'],
                png_max_pixels=settings[i]['png_max_pixels']
            )
            if success:
                images[i] = path
                if settings[i]['metadata_write_no_paths']:
                    image_paths_metadata.append(os.path.basename(path))
                elif settings[i]['metadata_write_relative_paths']:
                    image_paths_metadata.append(os.path.relpath(path, settings[i]['data_path']))
            else:
                return False, message, None, None, None
        self._logger.debug(f"Absolute image paths: {images}")
        self._logger.debug(f"Metadata image paths: {image_paths_metadata}")
        image_paths_metadata_counter = dict(collections.Counter(image_paths_metadata))
        for key in copy.deepcopy(image_paths_metadata_counter):
            if image_paths_metadata_counter[key] == 1:
                del image_paths_metadata_counter[key]
        if len(image_paths_metadata_counter) > 0:
            message = f"The image paths/names supposed to be written to metadata according to settings are not unique: {image_paths_metadata_counter}"
            self._logger.error(message)
            return False, message, None, None, None

        # prepare uuids
        uuids = []
        for i in range(len(images)):
            if settings[i]['usage_skip']:
                uuids.append([None, None, None])
            else:
                uuid_job = []
                for i in range(3):
                    uuid_job.append(uuid.uuid4().hex)
                uuids.append(uuid_job)
        self._logger.debug(f"UUIDs: {uuids}")

        # generate structured descriptions

        self._logger.info(f"Generating '{len(images)}' structured description{'s' if len(images) != 1 else ''}")

        scene_descriptions = [None] * len(images)
        structured_descriptions = [None] * len(images)
        metadata = [None] * len(images)
        for i in range(len(images)):
            metadata[i] = {
                'stamp_batch_start': stamp_start.isoformat(),
                'image_path': image_paths_metadata[i],
                'settings': settings[i],
                'uuid_scene_description': uuids[i][0],
                'uuid_structured_description': uuids[i][1],
                'uuid_decoupled_attribution': uuids[i][2],
                'parallel_completions': len(self._completions),
                'batch_size': len(images)
            }
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_batch_start', metadata[i]['stamp_batch_start']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'image_path', metadata[i]['image_path']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'settings', metadata[i]['settings']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'uuid_scene_description', metadata[i]['uuid_scene_description']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'uuid_structured_description', metadata[i]['uuid_structured_description']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'uuid_decoupled_attribution', metadata[i]['uuid_decoupled_attribution']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'parallel_completions', metadata[i]['parallel_completions']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'batch_size', metadata[i]['batch_size']))
            if len(images) > 1:
                metadata[i]['batch_images'] = images
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'batch_images', image_paths_metadata))

        status_order = ['UNPROCESSED', 'GENERATING_SCENE_DESCRIPTION', 'GENERATING_STRUCTURED_DESCRIPTION', 'GENERATING_DECOUPLED_ATTRIBUTE', 'ERROR', 'FINISHED']
        progress = [{'status': "UNPROCESSED"} for _ in images]
        while True:
            # status and termination

            status = [item['status'] for item in progress]
            info = dict(collections.Counter(status))
            info = {key: info[key] for key in status_order if key in info}
            info['TOTAL'] = len(images)
            done = [item in ["FINISHED", "ERROR"] for item in status]
            num_done = sum(done)

            stamp_now = datetime.datetime.now()
            time_running = (stamp_now - stamp_start).total_seconds()
            time_left = f" (~{format_seconds((time_running / num_done) * (len(images) - num_done))} remaining)" if num_done > 0 else ""

            if num_done == len(images):
                self._logger.info(f"Progress: {info} after {format_seconds(time_running)}")
                break

            self._logger.info(f"Progress: {info} after {format_seconds(time_running)}{time_left}", throttle_duration_sec=1.0)

            # timeouts
            if len(timeout_idx) > 0:
                for i in timeout_idx:
                    if time_running > settings[i]['timeout']:
                        message = f"Timeout while processing '{images[i]}' after '{settings[i]['timeout']}s'."
                        self._logger.error(message)
                        progress[i]['status'] = {'status': "ERROR"}
                        metadata[i]['stamp_error'] = stamp_now.isoformat()
                        metadata[i]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_error', metadata[i]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_error', message))

            # collect finished jobs
            forward = False
            for i in range(len(status)):
                if status[i] == "GENERATING_SCENE_DESCRIPTION":
                    success, _, completion_result = self._node.api_director.async_get(async_id=progress[i]['async_id'], mute_timeout_logging=True, timeout=0.0)
                    if success:
                        forward = True
                        del progress[i]['async_id']
                        self._completions[progress[i]['completions_ID']]['status'] = "IDLE"
                        success, message, completion = completion_result
                        if success:
                            scene_description = completion['text']
                            scene_descriptions[i] = scene_description
                            metadata[i]['scene_description_completion'] = completion
                            metadata[i]['stamp_scene_description_end'] = datetime.datetime.now().isoformat()
                            metadata[i]['duration_scene_description'] = (datetime.datetime.fromisoformat(metadata[i]['stamp_scene_description_end']) - datetime.datetime.fromisoformat(metadata[i]['stamp_scene_description_start'])).total_seconds()
                            metadata[i]['scene_description'] = scene_description
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'scene_description_completion', metadata[i]['scene_description_completion']))
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_scene_description_end', metadata[i]['stamp_scene_description_end']))
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'duration_scene_description', metadata[i]['duration_scene_description']))
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'scene_description', scene_description))
                            self._logger.info(f"Scene description of '{images[i]}' after '{(datetime.datetime.fromisoformat(metadata[i]['stamp_scene_description_end']) - stamp_start).total_seconds():.1f}s':\n{scene_description}")
                            if settings[i].get('dict_prompt') is None:
                                progress[i] = {'status': "FINISHED"}
                            else:
                                progress[i] = {'status': "GENERATED_SCENE_DESCRIPTION"}
                        elif settings[i]['retry'] == -1 or settings[i]['retry'] > retries[i]:
                            if settings[i]['retry'] == -1:
                                self._logger.info(f"Retrying to generate scene description of image '{i}' after failure")
                            else:
                                retries[i] += 1
                                self._logger.info(f"Starting retry attempt '{retries[i]}' of '{settings[i]['retry']}' to generate scene description of image '{i}' after failure")
                            metadata[i]['failed_scene_description_completions'] = metadata[i].get('failed_scene_description_completions', []) + [message if completion is None else completion]
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_scene_description_completions', metadata[i]['failed_scene_description_completions']))
                            progress[i] = {'status': "UNPROCESSED"}
                        else:
                            metadata[i]['stamp_error'] = datetime.datetime.now().isoformat()
                            metadata[i]['report_error'] = message
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_error', metadata[i]['stamp_error']))
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_error', message))
                            progress[i] = {'status': "ERROR"}
                elif status[i] == "GENERATING_STRUCTURED_DESCRIPTION":
                    success, _, completion_result = self._node.api_director.async_get(async_id=progress[i]['async_id'], mute_timeout_logging=True, timeout=0.0)
                    if success:
                        forward = True
                        del progress[i]['async_id']
                        self._completions[progress[i]['completions_ID']]['status'] = "IDLE"
                        success, message, completion = completion_result
                        if success:
                            structured_description_raw = completion['text']
                            success, message, structured_description, report = self._process_structured_description(structured_description_raw, images[i], settings[i])
                            if len(report) > 0:
                                metadata[i]['report_structured_description'] = report
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_structured_description', report))
                            if success:
                                structured_descriptions[i] = structured_description
                                metadata[i]['structured_description_completion'] = completion
                                metadata[i]['stamp_structured_description_end'] = datetime.datetime.now().isoformat()
                                metadata[i]['duration_structured_description'] = (datetime.datetime.fromisoformat(metadata[i]['stamp_structured_description_end']) - datetime.datetime.fromisoformat(metadata[i]['stamp_structured_description_start'])).total_seconds()
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'structured_description_completion', metadata[i]['structured_description_completion']))
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_structured_description_end', metadata[i]['stamp_structured_description_end']))
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'duration_structured_description', metadata[i]['duration_structured_description']))
                                if structured_description_raw != structured_description:
                                    metadata[i]['structured_description_raw'] = structured_description_raw
                                    self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'structured_description_raw', structured_description_raw))
                                if settings[i].get('task') is None:
                                    metadata[i]['structured_description'] = structured_description
                                    self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'structured_description', structured_description))
                                    self._logger.info(f"Structured description of '{images[i]}' after '{(datetime.datetime.fromisoformat(metadata[i]['stamp_structured_description_end']) - stamp_start).total_seconds():.1f}s':\n{json.dumps(structured_description, indent=4)}")
                                    progress[i] = {'status': "FINISHED"}
                                else:
                                    metadata[i]['structured_description_processed'] = copy.deepcopy(structured_description)
                                    self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'structured_description_processed', metadata[i]['structured_description_processed']))
                                    progress[i] = {'status': "GENERATED_STRUCTURED_DESCRIPTION"}
                            elif settings[i]['retry'] == -1 or settings[i]['retry'] > retries[i]:
                                if settings[i]['retry'] == -1:
                                    self._logger.info(f"Retrying to generate structured description of image '{i}' after failure")
                                else:
                                    retries[i] += 1
                                    self._logger.info(f"Starting retry attempt '{retries[i]}' of '{settings[i]['retry']}' to generate structured description of image '{i}' after failure")
                                metadata[i]['failed_structured_description_completions'] = metadata[i].get('failed_structured_description_completions', []) + [message if completion is None else completion]
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_structured_description_completions', metadata[i]['failed_structured_description_completions']))
                                if settings[i].get('scene_prompt') is None:
                                    progress[i] = {'status': "UNPROCESSED"}
                                else:
                                    progress[i] = {'status': "GENERATED_SCENE_DESCRIPTION"}
                            else:
                                metadata[i]['stamp_error'] = datetime.datetime.now().isoformat()
                                metadata[i]['report_error'] = message
                                metadata[i]['structured_description_raw'] = structured_description_raw
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_error', metadata[i]['stamp_error']))
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_error', message))
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'structured_description_raw', structured_description_raw))
                                progress[i] = {'status': "ERROR"}
                        elif settings[i]['retry'] == -1 or settings[i]['retry'] > retries[i]:
                            if settings[i]['retry'] == -1:
                                self._logger.info(f"Retrying to generate structured description of image '{i}' after failure")
                            else:
                                retries[i] += 1
                                self._logger.info(f"Starting retry attempt '{retries[i]}' of '{settings[i]['retry']}' to generate structured description of image '{i}' after failure")
                            metadata[i]['failed_structured_description_completions'] = metadata[i].get('failed_structured_description_completions', []) + [message if completion is None else completion]
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_structured_description_completions', metadata[i]['failed_structured_description_completions']))
                            if settings[i].get('scene_prompt') is None:
                                progress[i] = {'status': "UNPROCESSED"}
                            else:
                                progress[i] = {'status': "GENERATED_SCENE_DESCRIPTION"}
                        else:
                            metadata[i]['stamp_error'] = datetime.datetime.now().isoformat()
                            metadata[i]['report_error'] = message
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_error', metadata[i]['stamp_error']))
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_error', message))
                            progress[i] = {'status': "ERROR"}
                elif status[i] == "GENERATING_DECOUPLED_ATTRIBUTE":
                    success, _, completion_result = self._node.api_director.async_get(async_id=progress[i]['async_id'], mute_timeout_logging=True, timeout=0.0)
                    if success:
                        forward = True
                        del progress[i]['async_id']
                        self._completions[progress[i]['completions_ID']]['status'] = "IDLE"
                        success, message, completion = completion_result
                        if success:
                            decoupled_attribution = completion['text']
                            metadata[i]['decoupled_attribution_raw'] = decoupled_attribution
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'decoupled_attribution_raw', decoupled_attribution))
                            success, message, structured_description_with_task_relevancy, report = self._process_task_relevancy(decoupled_attribution, images[i], settings[i], structured_descriptions[i])
                            if len(report) > 0:
                                metadata[i]['report_decoupled_attribution'] = report
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_decoupled_attribution', report))
                            if success:
                                structured_descriptions[i] = structured_description_with_task_relevancy
                                metadata[i]['decoupled_attribution_completion'] = completion
                                metadata[i]['stamp_decoupled_attribution_end'] = datetime.datetime.now().isoformat()
                                metadata[i]['duration_decoupled_attribution'] = (datetime.datetime.fromisoformat(metadata[i]['stamp_decoupled_attribution_end']) - datetime.datetime.fromisoformat(metadata[i]['stamp_decoupled_attribution_start'])).total_seconds()
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'decoupled_attribution_completion', metadata[i]['decoupled_attribution_completion']))
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_decoupled_attribution_end', metadata[i]['stamp_decoupled_attribution_end']))
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'duration_decoupled_attribution', metadata[i]['duration_decoupled_attribution']))
                                metadata[i]['structured_description'] = structured_descriptions[i]
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'structured_description', structured_descriptions[i]))
                                self._logger.info(f"Structured description of '{images[i]}' after '{(datetime.datetime.fromisoformat(metadata[i]['stamp_decoupled_attribution_end']) - stamp_start).total_seconds():.1f}s':\n{json.dumps(structured_description, indent=4)}")
                                progress[i] = {'status': "FINISHED"}
                            elif settings[i]['retry'] == -1 or settings[i]['retry'] > retries[i]:
                                if settings[i]['retry'] == -1:
                                    self._logger.info(f"Retrying to generate decoupled attribution of image '{i}' after failure")
                                else:
                                    retries[i] += 1
                                    self._logger.info(f"Starting retry attempt '{retries[i]}' of '{settings[i]['retry']}' to generate decoupled attribution of image '{i}' after failure")
                                metadata[i]['failed_decoupled_attribution_completions'] = metadata[i].get('failed_decoupled_attribution_completions', []) + [message if completion is None else completion]
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_decoupled_attribution_completions', metadata[i]['failed_decoupled_attribution_completions']))
                                progress[i] = {'status': "GENERATED_STRUCTURED_DESCRIPTION"}
                            else:
                                metadata[i]['stamp_error'] = datetime.datetime.now().isoformat()
                                metadata[i]['report_error'] = message
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_error', metadata[i]['stamp_error']))
                                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_error', message))
                                progress[i] = {'status': "ERROR"}
                        elif settings[i]['retry'] == -1 or settings[i]['retry'] > retries[i]:
                            if settings[i]['retry'] == -1:
                                self._logger.info(f"Retrying to generate decoupled attribution of image '{i}' after failure")
                            else:
                                retries[i] += 1
                                self._logger.info(f"Starting retry attempt '{retries[i]}' of '{settings[i]['retry']}' to generate decoupled attribution of image '{i}' after failure")
                            metadata[i]['failed_decoupled_attribution_completions'] = metadata[i].get('failed_decoupled_attribution_completions', []) + [message if completion is None else completion]
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_decoupled_attribution_completions', metadata[i]['failed_decoupled_attribution_completions']))
                            progress[i] = {'status': "GENERATED_STRUCTURED_DESCRIPTION"}
                        else:
                            structured_descriptions[i] = None
                            metadata[i]['stamp_error'] = datetime.datetime.now().isoformat()
                            metadata[i]['report_error'] = message
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_error', metadata[i]['stamp_error']))
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_error', message))
                            progress[i] = {'status': "ERROR"}
            if forward:
                continue

            # find next job
            for i in range(len(status)):
                if status[i] == "UNPROCESSED":
                    if settings[i].get('scene_prompt') is not None:
                        job = {'image_ID': i, 'task': "SCENE_DESCRIPTION", "UUID": uuids[i][0]}
                        break
                    else:
                        job = {'image_ID': i, 'task': "STRUCTURED_DESCRIPTION", "UUID": uuids[i][1]}
                        break
                elif status[i] == "GENERATED_SCENE_DESCRIPTION":
                    job = {'image_ID': i, 'task': "STRUCTURED_DESCRIPTION", "UUID": uuids[i][1]}
                    break
                elif status[i] == "GENERATED_STRUCTURED_DESCRIPTION":
                    job = {'image_ID': i, 'task': "DECOUPLED_ATTRIBUTION", "UUID": uuids[i][2]}
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
            self._logger.debug(f"Generating '{job['task']}' for '{images[job['image_ID']]}' using completions '{self._completions[job['completions_ID']]['ID']}'")
            if job['task'] == "SCENE_DESCRIPTION":
                metadata[job['image_ID']]['stamp_scene_description_start'] = datetime.datetime.now().isoformat()
                self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_scene_description_start', metadata[job['image_ID']]['stamp_scene_description_start']))
                # set parameters
                success, message, async_id = self._node.api_director.async_set_parameters(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    parameter_names=list(settings[job['image_ID']]['vlm_scene_description'].keys()),
                    parameter_values=list(settings[job['image_ID']]['vlm_scene_description'].values()),
                    retry=False,
                    succeed_async_id=None
                )
                if not success:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying to generate scene description of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate scene description of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_scene_description_completions'] = metadata[i].get('failed_scene_description_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_scene_description_completions', metadata[i]['failed_scene_description_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add system prompt
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text=settings[job['image_ID']]['system_prompt_scene_description'],
                    role=settings[job['image_ID']]['system_prompt_scene_description_role'],
                    reset_context=True,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to scene description of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate scene description of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_scene_description_completions'] = metadata[i].get('failed_scene_description_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_scene_description_completions', metadata[i]['failed_scene_description_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add image prompt
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text={"role": "user", "content": [{'type': "image_url", 'image_url': {'detail': settings[job['image_ID']]['image_resolution_scene_description'], 'url': images[job['image_ID']]}}]},
                    role="json",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to scene description of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate scene description of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_scene_description_completions'] = metadata[i].get('failed_scene_description_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_scene_description_completions', metadata[i]['failed_scene_description_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # generate scene description
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text=settings[job['image_ID']]['scene_prompt'],
                    role="user",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="text",
                    identifier=job['UUID'],
                    retry=False,
                    succeed_async_id=async_id
                )
                if success:
                    progress[job['image_ID']] = {'status': "GENERATING_SCENE_DESCRIPTION", 'async_id': async_id, 'completions_ID': job['completions_ID']}
                else:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to scene description of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate scene description of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_scene_description_completions'] = metadata[i].get('failed_scene_description_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_scene_description_completions', metadata[i]['failed_scene_description_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
            elif job['task'] == "STRUCTURED_DESCRIPTION":
                metadata[job['image_ID']]['stamp_structured_description_start'] = datetime.datetime.now().isoformat()
                self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_structured_description_start', metadata[job['image_ID']]['stamp_structured_description_start']))
                # set parameters
                success, message, async_id = self._node.api_director.async_set_parameters(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    parameter_names=list(settings[job['image_ID']]['vlm_structured_description'].keys()),
                    parameter_values=list(settings[job['image_ID']]['vlm_structured_description'].values()),
                    retry=False,
                    succeed_async_id=None
                )
                if not success:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to structured description of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate structured description of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_structured_description_completions'] = metadata[i].get('failed_structured_description_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_structured_description_completions', metadata[i]['failed_structured_description_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add system prompt
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text=settings[job['image_ID']]['system_prompt_structured_description'],
                    role=settings[job['image_ID']]['system_prompt_structured_description_role'],
                    reset_context=True,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to structured description of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate structured description of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_structured_description_completions'] = metadata[i].get('failed_structured_description_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_structured_description_completions', metadata[i]['failed_structured_description_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add image prompt
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text={"role": "user", "content": [{'type': "image_url", 'image_url': {'detail': settings[job['image_ID']]['image_resolution_structured_description'], 'url': images[job['image_ID']]}}]},
                    role="json",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to structured description of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate structured description of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_structured_description_completions'] = metadata[i].get('failed_structured_description_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_structured_description_completions', metadata[i]['failed_structured_description_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # generate structured description
                prompt = settings[job['image_ID']]['dict_prompt']
                if settings[job['image_ID']].get('task') is not None and not settings[job['image_ID']]['decoupled_task_relevancy']:
                    prompt += settings[job['image_ID']]['direct_task_relevancy_prompt'].format(task=settings[job['image_ID']]['task'])
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text=prompt,
                    role="user",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="json" if settings[job['image_ID']]['use_json_mode'] else "text",
                    identifier=job['UUID'],
                    retry=False,
                    succeed_async_id=async_id
                )
                if success:
                    progress[job['image_ID']] = {'status': "GENERATING_STRUCTURED_DESCRIPTION", 'async_id': async_id, 'completions_ID': job['completions_ID']}
                else:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to structured description of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate structured description of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_structured_description_completions'] = metadata[i].get('failed_structured_description_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_structured_description_completions', metadata[i]['failed_structured_description_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
            elif job['task'] == "DECOUPLED_ATTRIBUTION":
                metadata[job['image_ID']]['stamp_decoupled_attribution_start'] = datetime.datetime.now().isoformat()
                self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_decoupled_attribution_start', metadata[job['image_ID']]['stamp_decoupled_attribution_start']))
                # set parameters
                success, message, async_id = self._node.api_director.async_set_parameters(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    parameter_names=list(settings[job['image_ID']]['vlm_decoupled_attribution'].keys()),
                    parameter_values=list(settings[job['image_ID']]['vlm_decoupled_attribution'].values()),
                    retry=False,
                    succeed_async_id=None
                )
                if not success:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to decoupled attribution of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate decoupled attribution of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_decoupled_attribution_completions'] = metadata[i].get('failed_decoupled_attribution_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_decoupled_attribution_completions', metadata[i]['failed_decoupled_attribution_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add system prompt
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text=settings[job['image_ID']]['system_prompt_decoupled_attribution'],
                    role=settings[job['image_ID']]['system_prompt_decoupled_attribution_role'],
                    reset_context=True,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to decoupled attribution of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate decoupled attribution of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_decoupled_attribution_completions'] = metadata[i].get('failed_decoupled_attribution_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_decoupled_attribution_completions', metadata[i]['failed_decoupled_attribution_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add image prompt
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text={"role": "user", "content": [{'type': "image_url", 'image_url': {'detail': settings[job['image_ID']]['image_resolution_decoupled_attribution'], 'url': images[job['image_ID']]}}]},
                    role="json",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to decoupled attribution of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate decoupled attribution of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_decoupled_attribution_completions'] = metadata[i].get('failed_decoupled_attribution_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_decoupled_attribution_completions', metadata[i]['failed_decoupled_attribution_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # add structured description
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text=json.dumps(structured_descriptions[job['image_ID']], indent=4),
                    role="assistant",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="none",
                    retry=False,
                    succeed_async_id=async_id
                )
                if not success:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to decoupled attribution of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate decoupled attribution of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_decoupled_attribution_completions'] = metadata[i].get('failed_decoupled_attribution_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_decoupled_attribution_completions', metadata[i]['failed_decoupled_attribution_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue
                # generate decoupled attribute
                prompt = settings[job['image_ID']]['dict_prompt']
                if settings[job['image_ID']].get('task') is not None and not settings[job['image_ID']]['decoupled_task_relevancy']:
                    prompt += settings[job['image_ID']]['direct_task_relevancy_prompt'].format(task=settings[job['image_ID']]['task'])
                success, message, async_id = self._node.api_director.async_prompt(
                    completions_id=self._completions[job['completions_ID']]['ID'],
                    text=settings[job['image_ID']]['decoupled_task_relevancy_prompt'].format(task=settings[job['image_ID']]['task']),
                    role="user",
                    reset_context=False,
                    tool_response_id=None,
                    response_type="json" if settings[job['image_ID']]['use_json_mode'] else "text",
                    identifier=job['UUID'],
                    retry=False,
                    succeed_async_id=async_id
                )
                if success:
                    progress[job['image_ID']] = {'status': "GENERATING_DECOUPLED_ATTRIBUTE", 'async_id': async_id, 'completions_ID': job['completions_ID']}
                else:
                    if settings[job['image_ID']]['retry'] == -1 or settings[job['image_ID']]['retry'] > retries[job['image_ID']]:
                        if settings[job['image_ID']]['retry'] == -1:
                            self._logger.info(f"Retrying generate to decoupled attribution of image '{job['image_ID']}' after failure")
                        else:
                            retries[job['image_ID']] += 1
                            self._logger.info(f"Starting retry attempt '{retries[job['image_ID']]}' of '{settings[job['image_ID']]['retry']}' to generate decoupled attribution of image '{job['image_ID']}' after failure")
                        metadata[i]['failed_decoupled_attribution_completions'] = metadata[i].get('failed_decoupled_attribution_completions', []) + [message]
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'failed_decoupled_attribution_completions', metadata[i]['failed_decoupled_attribution_completions']))
                    else:
                        metadata[job['image_ID']]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[job['image_ID']]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'stamp_error', metadata[job['image_ID']]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[job['image_ID']]['data_path'], settings[job['image_ID']]['metadata_file'], image_paths_metadata[job['image_ID']], 'report_error', message))
                        progress[job['image_ID']]['status'] = "ERROR"
                    self._completions[job['completions_ID']]['status'] = "IDLE"
                    continue

        # retrieve usage
        skip = all(settings[i]['usage_skip'] for i in range(len(settings)))
        if not skip:
            max_wait = max([settings[i]['usage_delay'] for i in range(len(settings))])
            if max_wait > 0.0:
                self._logger.debug(f"Sleeping '{max_wait}s' before requesting usage")
                time.sleep(max_wait)
            success, message, total_usage = self._node.api_director.get_usage(stamp_start=stamp_start)
            if success:
                # warnings = [[] for _ in range(len(settings))]
                tasks = ["scene_description", "structured_description", "decoupled_attribution"]
                history = total_usage.get('completions', {}).get('history', [])
                for i in range(len(settings)):
                    warnings = []
                    usage_image = {}
                    for j in range(3):
                        if j == 0 and settings[i].get('scene_prompt') is not None:
                            target_uuid = uuids[i][0]
                        elif j == 1 and settings[i].get('dict_prompt') is not None:
                            target_uuid = uuids[i][1]
                        elif j == 2 and settings[i].get('task') is not None and settings[i]['decoupled_task_relevancy']:
                            target_uuid = uuids[i][2]
                        else:
                            continue
                        usage_task = []
                        for item in history:
                            if item.get('identifier') == target_uuid:
                                usage_task.append(item)
                        usage_formatted = {
                            'tokens_input_cached': 0,
                            'tokens_input_uncached': 0,
                            'tokens_input': 0,
                            'tokens_output': 0,
                            'dollars_input': 0.0,
                            'dollars_output': 0.0,
                            'dollars_total': 0.0
                        }
                        if len(usage_task) == 0:
                            message = f"Cannot find usage for task '{tasks[j]}' of image '{i}'."
                            warnings.append(message)
                            self._logger.warn(message)
                        else:
                            for usage in usage_task:
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
                        if usage_formatted != {}:
                            usage_image[tasks[j]] = usage_formatted
                        if len(warnings) == 0:
                            if 'tokens_input_cached' not in usage_formatted and 'tokens_input_uncached' not in usage_formatted:
                                message = f"Cannot find input tokens for task '{tasks[j]}' of image '{i}'."
                                warnings.append(message)
                                self._logger.warn(message)
                            if 'tokens_output' not in usage_formatted:
                                message = f"Cannot find output tokens for task '{tasks[j]}' of image '{i}'."
                                warnings.append(message)
                                self._logger.warn(message)

                    if usage_image != {}:
                        usage_total = {
                            'tokens_input_cached': 0,
                            'tokens_input_uncached': 0,
                            'tokens_input': 0,
                            'tokens_output': 0,
                            'dollars_input': 0.0,
                            'dollars_output': 0.0,
                            'dollars_total': 0.0
                        }
                        for usage in usage_image:
                            for key in usage_image[usage]:
                                usage_total[key] += usage_image[usage][key]
                        for key in copy.deepcopy(usage_total):
                            if usage_total[key] == 0 or usage_total[key] == 0.0:
                                del usage_total[key]
                        usage_image['total'] = usage_total
                    if len(warnings) > 0:
                        usage_image['warnings'] = warnings
                    if usage_image != {}:
                        metadata[i]['usage'] = usage_image
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'usage', metadata[i]['usage']))

        # log
        status = [item['status'] for item in progress]
        assert set(status) <= {"FINISHED", "ERROR"}, status
        if "ERROR" in status:
            failures = sum([item['status'] == "ERROR" for item in progress])
            success = failures < len(images)
            if len(images) > 1:
                message = f"Failed to generate '{failures}' out of '{len(images)}' structured descriptions."
            else:
                message = "Failed to generate structured description."
            self._logger.error(message)
        else:
            success = True
            if len(images) > 1:
                message = f"Successfully generated all '{len(images)}' structured descriptions."
            else:
                message = "Successfully generated structured description."
            self._logger.info(message)
        stamp_end = datetime.datetime.now()
        for i in range(len(images)):
            metadata[i]['stamp_batch_end'] = stamp_end.isoformat()
            metadata[i]['duration_batch'] = (stamp_end - stamp_start).total_seconds()
            metadata[i]['success'] = status[i] == "FINISHED"
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_batch_end', metadata[i]['stamp_batch_end']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'duration_batch', metadata[i]['duration_batch']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'success', metadata[i]['success']))

        if no_batch:
            structured_descriptions = structured_descriptions[0]
            scene_descriptions = scene_descriptions[0]
            metadata = metadata[0]

        return success, message, structured_descriptions, scene_descriptions, metadata

    def _process_structured_description(self, response, image, settings):
        class ValidationError(Exception):
            pass

        report = []

        try:
            if not (isinstance(response, dict) or isinstance(response, list)):
                raise ValidationError(f"Expected response to be a list or a dictionary, but it is of type '{type(response).__name__}'.")

            if isinstance(response, list):
                objects_raw = response
            else:
                # must have at least one key
                if len(response.keys()) != 1:
                    message = f"Response is a dictionary with '{len(response.keys())}' keys: {list(response.keys())}"
                    if len(response.keys()) > 0:
                        message = f"{message}: {list(response.keys())}"
                    else:
                        raise ValidationError(message)
                    self._logger.warn(message)
                    report.append(message)

                # take first
                top_key = list(response.keys())[0]

                if isinstance(response[top_key], list):
                    objects_raw = response[top_key]
                elif isinstance(response[top_key], dict):
                    objects_raw = [response[top_key]]
                    message = "Interpreting dictionary under first key as single list element."
                    self._logger.warn(message)
                    report.append(message)
                else:
                    raise ValidationError("Response is a dictionary that cannot be interpreted.")

            # validation objects

            objects__processed = []

            dict_items = copy.deepcopy(settings['dict_items'])
            if settings.get('task') is not None and not settings['decoupled_task_relevancy']:
                dict_items.append(settings['dict_direct_task_relevancy_item'])

            required_keys = [item['key'] for item in dict_items if item['required']]

            for j, obj in enumerate(objects_raw):
                if isinstance(obj, dict):
                    obj_processed = {}
                    requirements = {}

                    for item in dict_items:
                        matched_key = levenshtein_match(item['key'], list(obj.keys()), threshold=settings['levenshtein_threshold'], normalization=settings['levenshtein_normalization'])
                        if matched_key is None:
                            if item['required']:
                                message = f"Object {j + 1}/{len(objects_raw)} does not contain required key '{item['key']}'."
                                self._logger.warn(message)
                            else:
                                message = f"Object {j + 1}/{len(objects_raw)} does not contain optional key '{item['key']}'."
                                self._logger.debug(message)
                            report.append(message)
                        else:
                            if matched_key != item['key']:
                                message = f"Object {j + 1}/{len(objects_raw)} key '{matched_key}' matched to '{item['key']}'."
                                self._logger.warn(message)
                                report.append(message)

                            if item['type'] == 'str':
                                target_type = str
                            elif item['type'] == 'bool':
                                target_type = bool
                            elif item['type'] == 'int':
                                target_type = int
                            elif item['type'] == 'float':
                                target_type = float

                            if target_type == str and not isinstance(obj[matched_key], target_type):
                                message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' is of type '{type(obj[matched_key]).__name__}' instead of 'str', casting it to 'str'."
                                self._logger.warn(message)
                                report.append(message)
                                try:
                                    obj[matched_key] = str(obj[matched_key])
                                except ValidationError as e:
                                    message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' failed casting to type 'str': {repr(e)}"
                                    self._logger.warn(message)
                                    report.append(message)

                            if isinstance(obj[matched_key], target_type):
                                if target_type == str: # do string matching for invalid and valid values
                                    if item['invalid'] is None:
                                        if item['valid'] is None:
                                            obj_processed[item['key']] = normalize_string(
                                                obj[matched_key],
                                                remove_underscores=settings['dict_value_normalization']['remove_underscores'],
                                                remove_punctuation=settings['dict_value_normalization']['remove_punctuation'],
                                                remove_common_specials=settings['dict_value_normalization']['remove_common_specials'],
                                                reduce_whitespaces=True,
                                                remove_whitespaces=settings['dict_value_normalization']['remove_whitespaces'],
                                                lowercase=settings['dict_value_normalization']['lowercase']
                                            )
                                            requirements[item['key']] = item['requires']
                                        else:
                                            matched_value = levenshtein_match(obj[matched_key], item['valid'], threshold=settings['levenshtein_threshold'], normalization=settings['levenshtein_normalization'])
                                            if matched_value is None:
                                                message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' value '{obj[matched_key]}' is not valid."
                                                self._logger.warn(message)
                                                report.append(message)
                                            else:
                                                if matched_value != obj[matched_key]:
                                                    message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' matched value '{obj[matched_key]}' to valid value '{matched_value}'."
                                                    self._logger.warn(message)
                                                    report.append(message)
                                                obj_processed[item['key']] = normalize_string(
                                                    obj[matched_key],
                                                    remove_underscores=settings['dict_value_normalization']['remove_underscores'],
                                                    remove_punctuation=settings['dict_value_normalization']['remove_punctuation'],
                                                    remove_common_specials=settings['dict_value_normalization']['remove_common_specials'],
                                                    reduce_whitespaces=True,
                                                    remove_whitespaces=settings['dict_value_normalization']['remove_whitespaces'],
                                                    lowercase=settings['dict_value_normalization']['lowercase']
                                                )
                                                requirements[item['key']] = item['requires']
                                    else:
                                        matched_value = levenshtein_match(obj[matched_key], item['invalid'], threshold=settings['levenshtein_threshold'], normalization=settings['levenshtein_normalization'])
                                        if matched_value is None:
                                            if item['valid'] is None:
                                                normalized_value = normalize_string(
                                                    obj[matched_key],
                                                    remove_underscores=settings['dict_value_normalization']['remove_underscores'],
                                                    remove_punctuation=settings['dict_value_normalization']['remove_punctuation'],
                                                    remove_common_specials=settings['dict_value_normalization']['remove_common_specials'],
                                                    reduce_whitespaces=True,
                                                    remove_whitespaces=settings['dict_value_normalization']['remove_whitespaces'],
                                                    lowercase=settings['dict_value_normalization']['lowercase']
                                                )
                                                matched_value = levenshtein_match(normalized_value, item['invalid'], threshold=settings['levenshtein_threshold'], normalization=settings['levenshtein_normalization'])
                                                if matched_value is None:
                                                    obj_processed[item['key']] = normalized_value
                                                    requirements[item['key']] = item['requires']
                                                else:
                                                    message = f"Object {j + 1}/{len(objects_raw)} normalized key '{item['key']}' value '{obj[matched_key]}' is invalid."
                                                    self._logger.warn(message)
                                                    report.append(message)
                                            else:
                                                matched_value = levenshtein_match(obj[matched_key], item['valid'], threshold=settings['levenshtein_threshold'], normalization=settings['levenshtein_normalization'])
                                                if matched_value is None:
                                                    message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' value '{obj[matched_key]}' is not valid."
                                                    self._logger.warn(message)
                                                    report.append(message)
                                                else:
                                                    if matched_value != obj[matched_key]:
                                                        message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' matched value '{obj[matched_key]}' to valid value '{matched_value}'."
                                                        self._logger.warn(message)
                                                        report.append(message)
                                                    obj_processed[item['key']] = matched_value
                                                    requirements[item['key']] = item['requires']
                                        else:
                                            if matched_value == obj[matched_key]:
                                                message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' value '{obj[matched_key]}' is invalid."
                                                self._logger.debug(message)
                                            else:
                                                message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' matched value '{obj[matched_key]}' to invalid value '{matched_value}'."
                                                self._logger.warn(message)
                                            report.append(message)
                                else: # do equality matching for invalid and valid values
                                    if item['invalid'] is None:
                                        if item['valid'] is None:
                                            obj_processed[item['key']] = obj[matched_key]
                                            requirements[item['key']] = item['requires']
                                        else:
                                            if obj[matched_key] in item['valid']:
                                                obj_processed[item['key']] = obj[matched_key]
                                                requirements[item['key']] = item['requires']
                                            else:
                                                message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' value '{obj[matched_key]}' is not valid."
                                                self._logger.warn(message)
                                                report.append(message)
                                    else:
                                        if obj[matched_key] in item['invalid']:
                                            message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' value '{obj[matched_key]}' is invalid."
                                            self._logger.debug(message)
                                            report.append(message)
                                        else:
                                            if item['valid'] is None:
                                                obj_processed[item['key']] = obj[matched_key]
                                                requirements[item['key']] = item['requires']
                                            else:
                                                if obj[matched_key] in item['valid']:
                                                    obj_processed[item['key']] = obj[matched_key]
                                                    requirements[item['key']] = item['requires']
                                                else:
                                                    message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' value '{obj[matched_key]}' is not valid."
                                                    self._logger.warn(message)
                                                    report.append(message)
                            else:
                                message = f"Object {j + 1}/{len(objects_raw)} key '{item['key']}' is of type '{type(obj[matched_key]).__name__}' instead of '{target_type.__name__}'."
                                self._logger.warn(message)
                                report.append(message)

                    # establish 'required' fields

                    reject_object = False
                    for k in required_keys:
                        if k not in obj_processed:
                            message = f"Object {j + 1}/{len(objects_raw)} misses required key '{k}'."
                            self._logger.warn(message)
                            report.append(message)
                            reject_object = True
                    if reject_object:
                        continue

                    # establish 'requires' fields

                    key_deletions = []
                    while True:
                        for k in obj_processed:
                            if k in key_deletions:
                                continue
                            elif requirements[k] is not None:
                                if requirements[k] in key_deletions or requirements[k] not in obj_processed:
                                    message = f"Object {j + 1}/{len(objects_raw)} key '{k}' requirement '{requirements[k]}' is not met."
                                    self._logger.warn(message)
                                    report.append(message)
                                    key_deletions.append(k)
                                    break
                        else:
                            break

                    for k in key_deletions:
                        del obj_processed[k]

                    if len(obj_processed) == 0:
                        message = f"Object {j + 1}/{len(objects_raw)} does not contain any valid key."
                        self._logger.warn(message)
                        report.append(message)
                    else:
                        objects__processed.append(obj_processed)

                else:
                    message = f"Object {j + 1}/{len(objects_raw)} is of type '{type(obj)}' instead of 'dict'."
                    self._logger.warn(message)
                    report.append(message)

            if len(objects__processed) == 0:
                raise ValidationError("Found zero valid object dictionaries.")

        except ValidationError as message:
            message = str(message)
            self._logger.error(message)
            report.append(message)
            return False, "Failed to parse valid structured description.", None, report
        else:
            objects_names = [objects__processed[i][settings['dict_items'][0]['key']] for i in range(len(objects__processed))]

            # make object names unique
            counts = collections.Counter(objects_names)
            identifiers = {}
            result = []
            for s in objects_names:
                if counts[s] > 1:
                    identifiers[s] = identifiers.get(s, 0) + 1
                    result.append(f"{s} {identifiers[s]}")
                else:
                    result.append(s)
            parsed_objects_names = result

            for i, name in enumerate(parsed_objects_names):
                objects__processed[i][settings['dict_items'][0]['key']] = name

            return True, "Successfully parsed valid structured description.", objects__processed, report

    def _process_task_relevancy(self, response, image, settings, structured_description):
        class ValidationError(Exception):
            pass

        report = []

        try:
            if isinstance(response, dict):
                if len(response) == 1:
                    key = list(response.keys())[0]
                    if isinstance(response[key], list):
                        response = response[key]
                        message = f"Response is a dict with one key '{key}' containing a list."
                        self._logger.debug(message)
                        report.append(message)

            if not isinstance(response, list):
                raise ValidationError(f"Failed to parse task-relevancy of unsupported type '{type(response).__name__}'.")

            task_relevancy_processed = []
            described_objects = [item[settings['dict_items'][0]['key']] for item in structured_description]

            for item in response:
                matched_key = levenshtein_match( # TODO do same thing as in validation before this
                    str(item),
                    described_objects,
                    threshold=settings['levenshtein_threshold'],
                    normalization=settings['levenshtein_normalization']
                )
                if matched_key is None:
                    message = f"Failed to match task-relevancy object '{item}'."
                    self._logger.warn(message)
                    report.append(message)
                else:
                    if matched_key != item:
                        message = f"Matched task-relevancy object '{item}' to described object '{matched_key}'."
                        self._logger.warn(message)
                        report.append(message)
                    if matched_key in task_relevancy_processed:
                        message = f"Matched task-relevancy object '{item}' was referenced before."
                        self._logger.warn(message)
                        report.append(message)
                    else:
                        task_relevancy_processed.append(matched_key)

        except ValidationError as message:
            self._logger.error(message)
            report.append(message)
            return False, "Failed to parse valid task-relevancy.", None, report
        else:
            self._logger.info(f"Task-Relevancy of '{image}': {task_relevancy_processed}")

            # integrate task-relevancy
            for i, item in enumerate(structured_description):
                if item[settings['dict_items'][0]['key']] in task_relevancy_processed:
                    structured_description[i][settings['dict_direct_task_relevancy_item']['key']] = True
                else:
                    structured_description[i][settings['dict_direct_task_relevancy_item']['key']] = False

            return True, "Successfully parsed task-relevancy.", structured_description, report
