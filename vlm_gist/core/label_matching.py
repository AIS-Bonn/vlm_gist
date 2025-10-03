#!/usr/bin/env python3

# STANDARD

import os
import re
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
from scipy.optimize import linear_sum_assignment

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.core.common import set_logger, set_settings, complete_settings, release_completions, save_metadata

from nimbro_api import ApiDirector
from nimbro_utils.lazy import read_json, write_json, normalize_string, levenshtein_match, get_package_path

default_settings = {
    'definitions_file': os.path.join(get_package_path("vlm_gist"), "data", "definitions.json"), # set None to deactivate; try to read definitions from cache before generating them
    'definitions_write': True, # write newly generated definitions to cache
    'define_labels': True, # define each item (label or label + description) to obtain an additional source embedding
    'define_targets': True, # define each target to obtain an additional target embedding
    'definition_item_indent': 4,
    'definition_item_key_name': "object_name",
    'definition_item_key_description': "description",
    'definition_prompt_system': "You are a helpful assistant. Follow all the instructions you are given as carefully as you can.",
    'definition_prompt_item': "Please define and describe the characteristic properties of the item category of the item described below in 5 sentences:\n{item}\n\nYour description must not refer to the given task or include the words 'item', 'category' and 'context'.",
    'definition_prompt_label': "Please define and describe the characteristic properties of the following term in 5 sentences: {label}",
    'llm': {
        'probe_api_connection': "False",
        'api_endpoint': "OpenAI",
        'model_name': "gpt-4o-2024-11-20",
        'model_temperature': "1.0",
        'model_top_p': "1.0",
        'model_max_tokens': "2000",
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
    'levenshtein_threshold': 0,
    'allow_duplicate_labels': True, # allows assigning multiple labels within a set to the same target
    'use_per_set_targets': False, # pass dataset_targets and augmented_targets per label set to get() instead of using the same for all sets from settings; if True both settings must be empty lists
    'dataset_targets': ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra'],
    'augmented_targets': ['air conditioner', 'ambulance', 'american football', 'artwork', 'ashtray', 'baby crib', 'baby stroller', 'backgammon set', 'baking tray', 'ball', 'balloon', 'barbecue grill', 'barrel', 'baseball', 'basket', 'basketball', 'bathrobe', 'bathtub', 'bean bag', 'blender', 'blinds', 'boot', 'bracelet', 'bread', 'bridge', 'broom', 'bucket', 'building', 'bulletin board', 'button', 'cabinet', 'cable', 'camera', 'can', 'can opener', 'candle', 'candy', 'cardboard box', 'care product', 'carpet', 'chandelier', 'chess board', 'chicken', 'chocolate', 'chopping board', 'clothes hanger', 'clothes iron', 'clothing', 'cloud', 'cocktail glass', 'cocktail shaker', 'coffee machine', 'coffee table', 'colander', 'computer', 'cooking pan', 'cooking pot', 'crane', 'crosswalk', 'curb', 'curtains', 'dartboard', 'decanter', 'decoration', 'deer', 'dirt', 'dish rack', 'dishwasher', 'document', 'door', 'door mat', 'doorbell', 'drawer', 'drone', 'duck', 'dustpan', 'electronic device', 'eye glasses', 'fan', 'faucet', 'fence', 'figurine', 'fireplace', 'fish', 'fish tank', 'flag', 'flashlight', 'floor', 'food', 'footstool', 'fruit', 'futon', 'garage', 'garlic press', 'globe', 'glove', 'golf ball', 'grass', 'ground', 'guitar', 'hairbrush', 'hamper', 'handball', 'handle', 'hat', 'hat stand', 'headphones', 'heater', 'helmet', 'hose', 'ice cream', 'ice cube tray', 'incense holder', 'insect', 'iron', 'ironing board', 'jewelry', 'kettle', 'key', 'kitchen scale', 'knob', 'ladder', 'lake', 'lamp', 'lamp post', 'lantern', 'lawn mower', 'lemon squeezer', 'letter', 'license plate', 'light', 'light bulb', 'loudspeaker', 'magazine', 'magnifying glass', 'mailbox', 'meal', 'mechanic tool', 'medicine', 'medicine cabinet', 'microphone', 'microscope', 'mirror', 'monitor', 'mountain', 'musical instrument', 'napkin', 'net', 'nightstand', 'notepad', 'nut', 'oar', 'ocean', 'office desk', 'ottoman', 'paintbrush', 'painting', 'pantry', 'pencil holder', 'pepper grinder', 'photo', 'photo album', 'picture frame', 'pig', 'pillow', 'placemat', 'plastic wrap', 'plate', 'plunger', 'pole', 'police', 'pond', 'pool table', 'poster', 'power outlet', 'price tag', 'printer', 'projector', 'puzzle', 'rabbit', 'radiator', 'rag', 'rail', 'rake', 'ramp', 'river', 'road sign', 'robot', 'rolling pin', 'ruler', 'safe', 'safety pin', 'salt shaker', 'sand', 'scale', 'scarf', 'sewing machine', 'shampoo', 'shaving kit', 'shelf', 'shoe', 'shoe rack', 'shovel', 'shower', 'sidewalk', 'sieve', 'sign', 'sky', 'smoke detector', 'soap', 'soap dispenser', 'socket', 'spatula', 'spice rack', 'spider', 'stapler', 'stick', 'sticker', 'stone', 'stool', 'storage box', 'strap', 'street', 'streetlight', 'suit', 'sun', 'surface', 'sweatband', 'switch', 'table tennis ball', 'tape dispenser', 'tarp', 'taxi', 'teapot', 'teeth', 'tennis ball', 'tent', 'thermometer', 'thermos', 'tissue box', 'toilet brush', 'toilet paper', 'toolbox', 'towel', 'towel rack', 'toy', 'traffic cone', 'trash bin', 'tray', 'tree', 'tripod', 'trophy', 'tupperware', 'turntable', 'tv stand', 'vacuum cleaner', 'vegetable', 'vent', 'video game console', 'volleyball', 'wall', 'wallet', 'wardrobe', 'washing machine', 'water glass', 'wave', 'weapon', 'wetsuit', 'wheel', 'whipped cream', 'whisk', 'whiteboard', 'window', 'wine rack', 'workout machine', 'writing', 'yoga mat'],
    'mapping_rules': {
        'monitor': 'tv',
        'soap dispenser': 'bottle',
        'ball': 'sports ball',
        'handball': 'sports ball',
        'basketball': 'sports ball',
        'baseball': 'sports ball',
        'volleyball': 'sports ball',
        'tennis ball': 'sports ball',
        'american football': 'sports ball',
        'golf ball': 'sports ball',
        'table tennis ball': 'sports ball',
    },
    'usage_skip': False,
    'usage_delay': 1.0,
    'data_path': os.path.join(get_package_path("vlm_gist"), "data", "label_matches"),
    'metadata_file': "label_matches.json" # set None to deactivate
}

class LabelMatching:

    def __init__(self, node, parallel_completions=None, metadata_timer=True, settings=None, logger_name=None, logger_severity=20):
        assert isinstance(node, Node), f"Provided argument 'node' is of invalid type '{type(node).__name__}'. Supported type is 'rclpy.node.Node'."
        assert parallel_completions is None or (isinstance(parallel_completions, int) and parallel_completions > 0), f"Provided argument 'parallel_completions' is of invalid type '{type(parallel_completions).__name__}'. Supported types are 'None' and 'int > 0'."
        assert isinstance(metadata_timer, bool), f"Provided argument 'metadata_timer' is of unsupported type '{type(metadata_timer).__name__}'. Supported type is 'bool'."
        assert logger_name is None or isinstance(logger_name, str), f"Provided argument 'logger_name' is of unsupported type '{type(logger_name).__name__}'. Supported types are 'None' and 'str'."

        self._node = node

        # logger
        if logger_name is None:
            logger_name = (f"{self._node.get_namespace()}.{self._node.get_name()}.label_matching").replace("/", "")
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

    def get(self, labels, descriptions=None, identifiers=None, dataset_targets=None, augmented_targets=None, settings=None):
        stamp_start = datetime.datetime.now()

        # read labels
        labels = copy.deepcopy(labels)
        some_input_is_batch = False
        if not isinstance(labels, list):
            message = f"Provided argument 'labels' is of unsupported type '{type(settings).__name__}'. Supported type is 'list'."
            self._logger.error(message)
            return False, message, None, None
        if len(labels) == 0:
            message = "Provided argument 'labels' must not be an empty list."
            self._logger.error(message)
            return False, message, None, None
        is_batch = all(isinstance(label, list) for label in labels)
        no_batch = all(isinstance(label, str) for label in labels)
        if not (is_batch or no_batch):
            message = f"Provided argument 'labels' is list that contains invalid type {[type(label).__name__ for label in labels]}. Supported types are either only 'str' or only 'list'."
            self._logger.error(message)
            return False, message, None, None
        if is_batch:
            some_input_is_batch = True
            if not all(all(isinstance(label, str) for label in labels_image) for labels_image in labels):
                message = f"Provided argument 'labels' is list of lists that contains invalid type {[[type(label).__name__ for label in labels_image] for labels_image in labels]}. Supported type is 'str'."
                self._logger.error(message)
                return False, message, None, None
            if any(len(labels_image) == 0 for labels_image in labels):
                message = f"Provided argument 'labels' is list of lists {[len(labels_image) for labels_image in labels]} where one of them is empty."
                self._logger.error(message)
                return False, message, None, None
            if len(labels) == 0:
                message = "Provided argument 'labels' is empty list."
                self._logger.error(message)
                return False, message, None, None
        else:
            labels = [labels]
        num_total_labels = sum(len(image_labels) for image_labels in labels)

        # read descriptions
        descriptions = copy.deepcopy(descriptions)
        if not isinstance(descriptions, list) and descriptions is not None:
            message = f"Provided argument 'descriptions' is of unsupported type '{type(settings).__name__}'. Supported type are 'list' and 'None'."
            self._logger.error(message)
            return False, message, None, None
        if descriptions is None:
            descriptions = []
            for i in range(len(labels)):
                descriptions_image = []
                for _ in range(len(labels[i])):
                    descriptions_image.append(None)
                descriptions.append(descriptions_image)
        else:
            if len(descriptions) == 0:
                message = "Provided argument 'descriptions' must not be an empty list."
                self._logger.error(message)
                return False, message, None, None
            is_batch = all(isinstance(description, list) for description in descriptions)
            no_batch = all(isinstance(description, str) or description is None for description in descriptions)
            if not (is_batch or no_batch):
                message = f"Provided argument 'descriptions' is list that contains invalid type {[type(description).__name__ for description in descriptions]}. Supported types are either only 'str' or 'None', or only 'list'."
                self._logger.error(message)
                return False, message, None, None
            if is_batch:
                some_input_is_batch = True
                if not all(all(isinstance(description, str) or description is None for description in descriptions_image) for descriptions_image in descriptions):
                    message = f"Provided argument 'descriptions' is list of lists that contains invalid type {[[type(description).__name__ for description in descriptions_image] for descriptions_image in descriptions]}. Supported type is 'str' or 'None'."
                    self._logger.error(message)
                    return False, message, None, None
                if any(len(descriptions_image) == 0 for descriptions_image in descriptions):
                    message = f"Provided argument 'descriptions' is list of lists {[len(descriptions_image) for descriptions_image in descriptions]} where one of them is empty."
                    self._logger.error(message)
                    return False, message, None, None
                if len(descriptions) == 0:
                    message = "Provided argument 'descriptions' is empty list."
                    self._logger.error(message)
                    return False, message, None, None
            else:
                descriptions = [descriptions]
            if len(labels) != len(descriptions):
                message = f"Expected number of description sets '{len(descriptions)}' to match the number of label sets '{len(labels)}'."
                self._logger.error(message)
                return False, message, None, None
            for image_labels, image_descriptions in zip(labels, descriptions):
                if len(image_descriptions) != len(image_labels):
                    message = f"Expected number of image descriptions '{len(image_descriptions)}' to match the number of image labels '{len(image_labels)}'."
                    self._logger.error(message)
                    return False, message, None, None

        # read identifiers
        identifiers = copy.deepcopy(identifiers)
        if identifiers is None:
            identifiers = []
            prefix = f"{stamp_start.strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]}"
            if len(labels) == 1:
                identifiers.append(f"{prefix}")
            else:
                for i in range(len(labels)):
                    identifiers.append(f"{prefix}_{i}")
        elif isinstance(identifiers, str):
            identifiers = [identifiers]
        for identifier in identifiers:
            if not isinstance(identifier, str):
                message = f"Provided argument 'identifiers' contains element of unsupported type '{type(identifier).__name__}'. Supported type is 'str'."
                self._logger.error(message)
                return False, message, None, None
        duplicates_identifier = dict(collections.Counter(identifiers))
        for key in copy.deepcopy(duplicates_identifier):
            if duplicates_identifier[key] == 1:
                del duplicates_identifier[key]
        if len(duplicates_identifier) > 0:
            message = f"Provided argument 'identifiers' contains duplicate elements: {duplicates_identifier}"
            self._logger.error(message)
            return False, message, None, None
        if len(identifiers) != len(labels):
            message = f"Expected number of identifiers '{len(identifiers)}' to match the number of images '{len(labels)}'."
            self._logger.error(message)
            return False, message, None, None
        self._logger.debug(f"Identifiers: {identifiers}")

        # read settings
        settings = copy.deepcopy(settings)
        if settings is not None and not isinstance(settings, dict):
            message = f"Provided argument 'settings' is of unsupported type '{type(settings).__name__}'. Supported types are 'None' and 'dict'."
            self._logger.error(message)
            return False, message, None, None
        if settings is None:
            settings = self._settings
        else:
            success, message, settings = self.complete_settings(settings)
            if not success:
                return False, message, None, None

        # read dataset_targets and augmented_targets
        if settings['use_per_set_targets']:
            if len(settings['dataset_targets']) > 0:
                message = "The number of dataset targets must be zero when setting 'use_per_set_targets' is 'True'."
                self._logger.error(message)
                return False, message, None, None
            if len(settings['augmented_targets']) > 0:
                message = "The number of augmented targets must be zero when setting 'use_per_set_targets' is 'True'."
                self._logger.error(message)
                return False, message, None, None

            dataset_targets = copy.deepcopy(dataset_targets)
            if not isinstance(dataset_targets, list):
                message = f"Provided argument 'dataset_targets' is of unsupported type '{type(dataset_targets).__name__}'. Supported type is 'list' when setting 'use_per_set_targets' is 'True'."
                self._logger.error(message)
                return False, message, None, None
            if len(dataset_targets) == 0:
                message = "Provided argument 'dataset_targets' must not be an empty list."
                self._logger.error(message)
                return False, message, None, None
            is_batch = all(isinstance(label, list) for label in dataset_targets)
            no_batch = all(isinstance(label, str) for label in dataset_targets)
            if not (is_batch or no_batch):
                message = f"Provided argument 'dataset_targets' is list that contains invalid type {[type(label).__name__ for label in dataset_targets]}. Supported types are either only 'str' or only 'list'."
                self._logger.error(message)
                return False, message, None, None
            if is_batch:
                some_input_is_batch = True
                if not all(all(isinstance(label, str) for label in dataset_targets_image) for dataset_targets_image in dataset_targets):
                    message = f"Provided argument 'dataset_targets' is list of lists that contains invalid type {[[type(label).__name__ for label in dataset_targets_image] for dataset_targets_image in dataset_targets]}. Supported type is 'str'."
                    self._logger.error(message)
                    return False, message, None, None
                if any(len(dataset_targets_image) == 0 for dataset_targets_image in dataset_targets):
                    message = f"Provided argument 'dataset_targets' is list of lists {[len(dataset_targets_image) for dataset_targets_image in dataset_targets]} where one of them is empty."
                    self._logger.error(message)
                    return False, message, None, None
            else:
                dataset_targets = [dataset_targets]
            if len(labels) != len(dataset_targets):
                message = f"Expected number of dataset target sets '{len(dataset_targets)}' to match the number of label sets '{len(labels)}'."
                self._logger.error(message)
                return False, message, None, None

            matching_targets = []
            for i in range(len(dataset_targets)):
                for j in range(len(dataset_targets[i])):
                    matching_targets.append(dataset_targets[i][j])

            if augmented_targets is None:
                augmented_targets = []
                for i in range(len(labels)):
                    augmented_targets.append([])
            else:
                augmented_targets = copy.deepcopy(augmented_targets)
                if not isinstance(augmented_targets, list):
                    message = f"Provided argument 'augmented_targets' is of unsupported type '{type(augmented_targets).__name__}'. Supported types are 'None' and 'list' when setting 'use_per_set_targets' is 'True'."
                    self._logger.error(message)
                    return False, message, None, None
                if len(augmented_targets) == 0:
                    message = "Provided argument 'augmented_targets' must not be an empty list."
                    self._logger.error(message)
                    return False, message, None, None
                is_batch = all(isinstance(label, list) for label in augmented_targets)
                no_batch = all(isinstance(label, str) for label in augmented_targets)
                if not (is_batch or no_batch):
                    message = f"Provided argument 'augmented_targets' is list that contains invalid type {[type(label).__name__ for label in augmented_targets]}. Supported types are either only 'str' or only 'list'."
                    self._logger.error(message)
                    return False, message, None, None
                if is_batch:
                    some_input_is_batch = True
                    if not all(all(isinstance(label, str) for label in augmented_targets_image) for augmented_targets_image in augmented_targets):
                        message = f"Provided argument 'augmented_targets' is list of lists that contains invalid type {[[type(label).__name__ for label in augmented_targets_image] for augmented_targets_image in augmented_targets]}. Supported type is 'str'."
                        self._logger.error(message)
                        return False, message, None, None
                else:
                    augmented_targets = [augmented_targets]
                if len(labels) != len(augmented_targets):
                    message = f"Expected number of augmented target sets '{len(augmented_targets)}' to match the number of label sets '{len(labels)}'."
                    self._logger.error(message)
                    return False, message, None, None

                for i in range(len(augmented_targets)):
                    for j in range(len(augmented_targets[i])):
                        matching_targets.append(augmented_targets[i][j])

        else:
            if dataset_targets is not None:
                message = f"Provided argument 'dataset_targets' is of unsupported type '{type(dataset_targets).__name__}'. Supported type is 'None' when setting 'use_per_set_targets' is 'False'."
                self._logger.error(message)
                return False, message, None, None
            if augmented_targets is not None:
                message = f"Provided argument 'augmented_targets' is of unsupported type '{type(augmented_targets).__name__}'. Supported type is 'None' when setting 'use_per_set_targets' is 'False'."
                self._logger.error(message)
                return False, message, None, None
            if len(settings['dataset_targets']) == 0:
                message = "The number of dataset targets must be greater zero when setting 'use_per_set_targets' is 'False'."
                self._logger.error(message)
                return False, message, None, None
            matching_targets = settings['dataset_targets'] + settings['augmented_targets']

        matching_targets = list(set(matching_targets))
        self._logger.debug(f"Matching targets: {matching_targets} ({len(matching_targets)})")

        self._logger.info(f"Matching '{len(labels)}' set{'s' if len(labels) != 1 else ''} with a total of '{num_total_labels}' label{'s' if num_total_labels != 1 else ''} to '{len(matching_targets)}' matching target{'s' if len(matching_targets) != 1 else ''}")

        usage_uuid = uuid.uuid4().hex

        # initialize metadata
        warnings = [[] for _ in labels]
        metadata = [None] * len(labels)
        for i in range(len(labels)):
            metadata[i] = {
                'stamp_batch_start': stamp_start.isoformat(),
                'settings': settings,
                'labels': labels[i],
                'num_labels': len(labels[i]),
                'parallel_completions': len(self._completions),
                'batch_usage_uuid': usage_uuid,
                'batch_size_sets': len(labels),
                'batch_size_labels': num_total_labels
            }
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'stamp_batch_start', metadata[i]['stamp_batch_start']))
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'settings', metadata[i]['settings']))
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'labels', metadata[i]['labels']))
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'num_labels', metadata[i]['num_labels']))
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'parallel_completions', metadata[i]['parallel_completions']))
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'batch_usage_uuid', metadata[i]['batch_usage_uuid']))
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'batch_size_sets', metadata[i]['batch_size_sets']))
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'batch_size_labels', metadata[i]['batch_size_labels']))
            if len(labels) > 1:
                metadata[i]['batch_identifiers'] = identifiers
                self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'batch_identifiers', metadata[i]['batch_identifiers']))

        # retrieve definitions
        if settings['define_labels'] or settings['define_targets']:
            def_texts = []
            def_texts_key = []
            def_texts_is_item = []
            if settings['define_labels']:
                labels_def_keys = []
                for i in range(len(labels)):
                    labels_def_keys_image = []
                    for j in range(len(labels[i])):
                        label_norm = re.sub(r'[_]?\d+$', '', labels[i][j]).rstrip() # label_1 -> label; label1 -> label
                        label_norm = re.sub(r'\s+', ' ', label_norm).strip() # reduces consecutive whitespaces (including linebreaks and tabs) to a single whitespace
                        if descriptions[i][j] is None:
                            def_texts_is_item.append(False)
                            item = label_norm
                        else:
                            def_texts_is_item.append(True)
                            description_norm = re.sub(r'[_]?\d+$', '', descriptions[i][j]).rstrip() # label_1 -> label; label1 -> label
                            description_norm = re.sub(r'\s+', ' ', description_norm).strip() # reduces consecutive whitespaces (including linebreaks and tabs) to a single whitespace
                            if label_norm == "":
                                message = f"Using original label '{labels[i][j]}' to generate definition after normalization yielded an empty string."
                                warnings[i].append(message)
                            if description_norm == "":
                                message = f"Using original description '{descriptions[i][j]}' to generate definition after normalization yielded an empty string."
                                warnings[i].append(message)
                            item = {
                                settings['definition_item_key_name']: labels[i][j] if label_norm == "" else label_norm,
                                settings['definition_item_key_description']: descriptions[i][j] if description_norm == "" else description_norm
                            }
                            try:
                                item = json.dumps(item, indent=settings['definition_item_indent'])
                            except Exception as e:
                                message = f"Failed to parse label '{labels[i][j]}' and description '{descriptions[i][j]}' ({i}.{j}) as JSON: {repr(e)}"
                                self._logger.error(message)
                                return False, message, None, metadata
                        def_texts.append(item)
                        def_texts_key.append(normalize_string(
                            item,
                            remove_underscores=True,
                            remove_punctuation=True,
                            remove_common_specials=True,
                            remove_whitespaces=True,
                            lowercase=True
                        ))
                        labels_def_keys_image.append(def_texts_key[-1])
                    labels_def_keys.append(labels_def_keys_image)

            if settings['define_targets']:
                matching_targets_key = [normalize_string(
                    matching_target,
                    remove_underscores=True,
                    remove_punctuation=True,
                    remove_common_specials=True,
                    remove_whitespaces=True,
                    lowercase=True
                ) for matching_target in matching_targets]
                def_texts += matching_targets
                def_texts_key += matching_targets_key
                def_texts_is_item += [False] * len(matching_targets)
                matching_targets_to_def_key = dict(zip(matching_targets, matching_targets_key))

            # remove duplicate def keys
            unique_dict = {}
            [unique_dict.setdefault(x, (y, z)) for x, y, z in zip(def_texts_key, def_texts, def_texts_is_item)]
            def_texts_key_unique = list(unique_dict.keys())
            def_texts_unique, def_texts_is_item_unique = zip(*unique_dict.values())

            # self._logger.debug(f"def_texts_key_unique: {def_texts_key_unique}")
            # self._logger.debug(f"def_texts_unique: {def_texts_unique}")
            # self._logger.debug(f"def_texts_is_item_unique: {def_texts_is_item_unique}")

            # read definitions
            if settings['definitions_file'] is None or not os.path.exists(settings['definitions_file']):
                if settings['definitions_file'] is not None:
                    self._logger.warn(f"Definitions file '{settings['definitions_file']}' does not exist")
                definitions_cache = {}
                def_texts_unique_undefined = def_texts_unique
                def_texts_key_unique_undefined = def_texts_key_unique
                def_texts_is_item_unique_undefined = def_texts_is_item_unique
            else:
                self._logger.info(f"Reading definitions file '{settings['definitions_file']}'")
                success, message, definitions_cache = read_json(file_path=settings['definitions_file'], logger=self._logger)
                if not success:
                    return False, message, None, metadata
                if not isinstance(definitions_cache, dict):
                    message = f"Content of definitions file is of type '{type(definitions_cache).__name__}. Supported type is 'dict'."
                    self._logger.error(message)
                    return False, message, None, metadata
                # identify missing definitions
                undefined_idx = [i for i in range(len(def_texts_key_unique)) if def_texts_key_unique[i] not in definitions_cache]
                def_texts_unique_undefined = [def_texts_unique[i] for i in undefined_idx]
                def_texts_key_unique_undefined = [def_texts_key_unique[i] for i in undefined_idx]
                def_texts_is_item_unique_undefined = [def_texts_is_item_unique[i] for i in undefined_idx]

            # generate missing definitions
            if len(def_texts_unique_undefined) == 0:
                self._logger.info("Found all required definitions in cache")
            else:
                write_definitions = False
                self._logger.info(f"Generating '{len(def_texts_unique_undefined)}' missing definition{'s' if len(def_texts_unique_undefined) != 1 else ''}")

                status_order = ['UNPROCESSED', 'GENERATING_DEFINITION', 'GENERATED_DEFINITION']
                progress = [{'status': "UNPROCESSED"} for _ in def_texts_unique_undefined]
                while True:
                    # termination
                    status = [item['status'] for item in progress]
                    info = dict(collections.Counter(status))
                    info = {key: info[key] for key in status_order if key in info}
                    done = [item in ["GENERATED_DEFINITION"] for item in status]
                    if all(done):
                        self._logger.info(f"Progress: {info} ({len(def_texts_unique_undefined)})")
                        break
                    self._logger.info(f"Progress: {info} ({len(def_texts_unique_undefined)})", throttle_duration_sec=1.0)
                    # self._logger.debug(f"{progress}", throttle_duration_sec=1.0)

                    # collect finished jobs
                    forward = False
                    for i in range(len(status)):
                        if status[i] == "GENERATING_DEFINITION":
                            success, _, completion_result = self._node.api_director.async_get(async_id=progress[i]['async_id'], mute_timeout_logging=True, timeout=0.0)
                            if success:
                                forward = True
                                del progress[i]['async_id']
                                self._completions[progress[i]['completions_ID']]['status'] = "IDLE"
                                success, message, completion = completion_result
                                if success:
                                    definition = completion['text']
                                    self._logger.debug(f"Adding definition '{definition}' to key '{def_texts_key_unique_undefined[i]}'")
                                    assert def_texts_key_unique_undefined[i] not in definitions_cache
                                    definitions_cache[def_texts_key_unique_undefined[i]] = definition
                                    if settings['definitions_write']:
                                        write_definitions = True
                                    progress[i] = {'status': "GENERATED_DEFINITION"}
                                else:
                                    self._logger.info(f"Retrying definition of label '{i}' after failure")
                                    progress[i] = {'status': "UNPROCESSED"}
                    if forward:
                        continue

                    # find next job
                    for i in range(len(status)):
                        if status[i] == "UNPROCESSED":
                            job = {'label_ID': i, 'task': "DEFINITION", "UUID": usage_uuid}
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
                    self._logger.debug(f"Generating '{job['task']}' for '{def_texts_unique_undefined[job['label_ID']]}' using completions '{self._completions[job['completions_ID']]['ID']}'")
                    if job['task'] == "DEFINITION":
                        # set parameters
                        success, message, async_id = self._node.api_director.async_set_parameters(
                            completions_id=self._completions[job['completions_ID']]['ID'],
                            parameter_names=list(settings['llm'].keys()),
                            parameter_values=list(settings['llm'].values()),
                            retry=False,
                            succeed_async_id=None
                        )
                        if not success:
                            self._completions[job['completions_ID']]['status'] = "IDLE"
                            continue
                        # add system prompt
                        success, message, async_id = self._node.api_director.async_prompt(
                            completions_id=self._completions[job['completions_ID']]['ID'],
                            text=settings['definition_prompt_system'],
                            role="system",
                            reset_context=True,
                            tool_response_id=None,
                            response_type="none",
                            retry=False,
                            succeed_async_id=async_id
                        )
                        if not success:
                            continue
                        # generate definition
                        if def_texts_is_item_unique_undefined[job['label_ID']]:
                            prompt = settings['definition_prompt_label'].format(label=def_texts_unique_undefined[job['label_ID']])
                        else:
                            prompt = settings['definition_prompt_item'].format(item=def_texts_unique_undefined[job['label_ID']])

                        success, message, async_id = self._node.api_director.async_prompt(
                            completions_id=self._completions[job['completions_ID']]['ID'],
                            text=prompt,
                            role="user",
                            reset_context=False,
                            tool_response_id=None,
                            response_type="text",
                            identifier=job['UUID'],
                            retry=False,
                            succeed_async_id=async_id
                        )
                        if success:
                            progress[job['label_ID']] = {'status': "GENERATING_DEFINITION", 'async_id': async_id, 'completions_ID': job['completions_ID']}
                        else:
                            self._completions[job['completions_ID']]['status'] = "IDLE"
                            continue

                if write_definitions and settings['definitions_file'] is not None:
                    self._logger.info(f"Saving new definition{'s' if len(def_texts_unique_undefined) != 1 else ''} to '{settings['definitions_file']}'")
                    write_json(file_path=settings['definitions_file'], json_object=definitions_cache, indent=True, logger=self._logger)

        definitions = []
        for i in range(len(labels)):
            definitions_image = []
            for j in range(len(labels[i])):
                if settings['define_labels']:
                    definition = definitions_cache[labels_def_keys[i][j]]
                else:
                    definition = None
                definitions_image.append(definition)
            definitions.append(definitions_image)

        matching_targets_definition = []
        for matching_target in matching_targets:
            if settings['define_targets']:
                def_key = matching_targets_to_def_key[matching_target]
                definition = definitions_cache[def_key]
            else:
                definition = None
            matching_targets_definition.append(definition)

        # retrieve embeddings

        labels_norm = []
        labels_norm_flat = []
        for i in range(len(labels)):
            labels_norm_image = []
            for j in range(len(labels[i])):
                label_norm = re.sub(r'[_]?\d+$', '', labels[i][j]).rstrip() # label_1 -> label; label1 -> label
                label_norm = re.sub(r'\s+', ' ', label_norm).strip() # reduces consecutive whitespaces (including linebreaks and tabs) to a single whitespace
                if label_norm == "":
                    label_norm = labels[i][j]
                    warnings[i].append(f"Using original label '{labels[i][j]}' to obtain embedding after normalization yielded an empty string.")
                labels_norm_flat.append(label_norm)
                labels_norm_image.append(label_norm)
            labels_norm.append(labels_norm_image)
        labels_norm_flat_unique = list(set(labels_norm_flat))

        descriptions_norm = []
        descriptions_norm_flat = []
        for i in range(len(descriptions)):
            descriptions_norm_image = []
            for j in range(len(descriptions[i])):
                if descriptions[i][j] is None:
                    description_norm = None
                else:
                    description_norm = re.sub(r'\s+', ' ', descriptions[i][j]).strip() # reduces consecutive whitespaces (including linebreaks and tabs) to a single whitespace
                    if description_norm == "":
                        description_norm = descriptions[i][j]
                        warnings[i].append(f"Using original description '{descriptions[i][j]}' to obtain embedding after normalization yielded an empty string.")
                    descriptions_norm_flat.append(description_norm)
                descriptions_norm_image.append(description_norm)
            descriptions_norm.append(descriptions_norm_image)
        descriptions_norm_flat_unique = list(set(descriptions_norm_flat))

        definitions_norm = []
        definitions_norm_flat = []
        for i in range(len(definitions)):
            definitions_norm_image = []
            for j in range(len(definitions[i])):
                if definitions[i][j] is None:
                    definition_norm = None
                else:
                    definition_norm = re.sub(r'\s+', ' ', definitions[i][j]).strip() # reduces consecutive whitespaces (including linebreaks and tabs) to a single whitespace
                    if definition_norm == "":
                        definition_norm = definitions[i][j]
                        warnings[i].append(f"Using original definition '{definitions[i][j]}' to obtain embedding after normalization yielded an empty string.")
                    definitions_norm_flat.append(definition_norm)
                definitions_norm_image.append(definition_norm)
            definitions_norm.append(definitions_norm_image)
        definitions_norm_flat_unique = list(set(definitions_norm_flat))

        if settings['define_targets']:
            matching_targets_definition_unique = list(set(matching_targets_definition))
        else:
            matching_targets_definition_unique = []

        texts = labels_norm_flat_unique + descriptions_norm_flat_unique + definitions_norm_flat_unique + matching_targets + matching_targets_definition_unique
        lengths = [len(labels_norm_flat_unique), len(descriptions_norm_flat_unique), len(definitions_norm_flat_unique), len(matching_targets), len(matching_targets_definition_unique)]
        self._logger.info(f"Obtaining '{len(texts)}' text embedding{'s' if len(texts) != 1 else ''}")
        self._logger.debug(f"Unique labels, label descriptions, item definitions, targets, target definitions: {lengths}")
        success, message, embeddings = self._node.api_director.get_embeddings(
            text=texts,
            identifier=usage_uuid,
            retry=True
        )
        if not success:
            return False, message, None, metadata

        lengths = np.cumsum(lengths)

        labels_norm_to_emb = dict(zip(labels_norm_flat_unique, embeddings[:lengths[0]]))
        descriptions_norm_to_emb = dict(zip(descriptions_norm_flat_unique, embeddings[lengths[0]:lengths[1]]))
        definitions_norm_to_emb = dict(zip(definitions_norm_flat_unique, embeddings[lengths[1]:lengths[2]]))

        labels_emb = []
        descriptions_emb = []
        definitions_emb = []
        for i in range(len(labels)):
            labels_emb_image = []
            descriptions_emb_image = []
            definitions_emb_image = []
            for j in range(len(labels[i])):
                labels_emb_image.append(labels_norm_to_emb[labels_norm[i][j]])
                if descriptions_norm[i][j] is None:
                    descriptions_emb_image.append(None)
                else:
                    descriptions_emb_image.append(descriptions_norm_to_emb[descriptions_norm[i][j]])
                if definitions_norm[i][j] is None:
                    definitions_emb_image.append(None)
                else:
                    definitions_emb_image.append(definitions_norm_to_emb[definitions_norm[i][j]])
            labels_emb.append(labels_emb_image)
            descriptions_emb.append(descriptions_emb_image)
            definitions_emb.append(definitions_emb_image)

        matching_targets_emb = embeddings[lengths[2]:lengths[3]]
        if settings['define_targets']:
            matching_targets_definition_to_emb = dict(zip(matching_targets_definition_unique, embeddings[lengths[3]:]))
            matching_targets_definition_emb = [matching_targets_definition_to_emb[definition] for definition in matching_targets_definition]

        # assert len(matching_targets) == len(matching_targets_emb)
        # if settings['define_targets']:
        #     assert len(matching_targets_definition) == len(matching_targets_definition_emb)
        # self._logger.debug(f"matching_targets: {matching_targets} ({len(matching_targets)})")
        # self._logger.debug(f"matching_targets_definition: {[item[:10] for item in matching_targets_definition]} ({len(matching_targets_definition)})")
        # if settings['define_targets']:
        #     self._logger.debug(f"matching_targets_definition_emb: {[type(item).__name__ for item in matching_targets_definition_emb]} ({len(matching_targets_definition_emb)})")
        # for i in range(len(labels)):
        #     self._logger.debug(f"labels[{i}]: {labels[i]} ({len(labels[i])})")
        #     self._logger.debug(f"labels_norm[{i}]: {labels_norm[i]} ({len(labels_norm[i])})")
        #     self._logger.debug(f"labels_emb[{i}]: {[type(item).__name__ for item in labels_emb[i]]} ({len(labels_emb[i])})")
        #     self._logger.debug(f"descriptions[{i}]: {descriptions[i]} ({len(descriptions[i])})")
        #     self._logger.debug(f"descriptions_norm[{i}]: {descriptions_norm[i]} ({len(descriptions_norm[i])})")
        #     self._logger.debug(f"descriptions_emb[{i}]: {[type(item).__name__ for item in descriptions_emb[i]]} ({len(descriptions_emb[i])})")
        #     self._logger.debug(f"definitions[{i}]: {[item[:10] for item in definitions[i]]} ({len(definitions[i])})")
        #     self._logger.debug(f"definitions_norm[{i}]: {[item[:10] for item in definitions_norm[i]]} ({len(definitions_norm[i])})")
        #     self._logger.debug(f"definitions_emb[{i}]: {[type(item).__name__ for item in definitions_emb[i]]} ({len(definitions_emb[i])})")

        # score embedding distances

        if not settings['use_per_set_targets']:
            current_targets = settings['dataset_targets'] + settings['augmented_targets']
            current_dataset_targets = settings['dataset_targets']
            target_idx = [matching_targets.index(target) for target in current_targets]
            current_targets_emb = np.array([matching_targets_emb[j] for j in target_idx])
            if settings['define_targets']:
                current_targets_definition_emb = np.array([matching_targets_definition_emb[j] for j in target_idx])

        results = []
        for i in range(len(labels)):
            if settings['use_per_set_targets']:
                current_targets = dataset_targets[i] + augmented_targets[i]
                current_dataset_targets = dataset_targets[i]
                target_idx = [matching_targets.index(target) for target in current_targets]
                current_targets_emb = np.array([matching_targets_emb[j] for j in target_idx])
                if settings['define_targets']:
                    current_targets_definition_emb = np.array([matching_targets_definition_emb[j] for j in target_idx])

            self._logger.info(f"Matching label set '{i + 1}' of '{len(labels)}' with '{len(labels[i])}' label{'s' if len(labels[i]) != 1 else ''} to '{len(current_targets)}' target{'s' if len(current_targets) != 1 else ''}")
            self._logger.debug(f"Labels[{i}]: '{labels[i]}' ({len(labels[i])})")
            # self._logger.debug(f"Descriptions[{i}]: '{descriptions[i]}' ({len(descriptions[i])})")
            # self._logger.debug(f"Definitions[{i}]: '{definitions[i]}' ({len(definitions[i])})")
            self._logger.debug(f"Targets: '{current_targets}' ({len(current_targets)})")

            # levenshtein matching
            if settings['allow_duplicate_labels']:
                matches_lev = [levenshtein_match(label, current_targets, threshold=settings['levenshtein_threshold'], normalization=True) for label in labels_norm[i]]
                metadata[i]['matches_levenshtein'] = matches_lev
                self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'matches_levenshtein', metadata[i]['matches_levenshtein']))
                self._logger.debug(f"Matches (lev): {matches_lev} ({len(matches_lev)})")

            # construct embeddings vectors
            embedding_vecs, lengths = [], []
            for label_emb, description_emb, definition_emb in zip(labels_emb[i], descriptions_emb[i], definitions_emb[i]):
                embedding_vecs.append(label_emb)
                length = 1
                if description_emb is not None:
                    embedding_vecs.append(description_emb)
                    length += 1
                if definition_emb is not None:
                    embedding_vecs.append(definition_emb)
                    length += 1
                lengths.append(length)

            # figure out how the embeddings are split since there might be 2 or 3 per detection
            lengths = np.array(lengths)
            split_idx = np.cumsum(lengths)

            # normalize them all at once
            embedding_vecs = np.vstack(embedding_vecs) # num_embeddings x emb_dim
            embedding_vecs /= np.linalg.norm(embedding_vecs, axis=1, keepdims=True)

            # split the normalized embeddings
            detection_embs = np.split(embedding_vecs, split_idx[:-1]) # list of arrays (varied lengths)

            # get shapes
            num_detections = len(detection_embs)
            emb_dim = detection_embs[0].shape[-1]
            max_len = lengths.max()

            # create padded array
            padded = np.zeros((num_detections, max_len, emb_dim), dtype=embedding_vecs.dtype) # num_detections x max_len x emb_dim
            for j, emb in enumerate(detection_embs):
                padded[j, :emb.shape[0], :] = emb

            # compute all relevant semantic distances
            target_idx = np.arange(len(current_targets), dtype=int)
            similarity_matrix_labels = padded @ current_targets_emb[target_idx, :].T[None] # num_detections x max_len x len(labels[i])

            # combine to produce meta score
            padding_mask = np.arange(max_len)[None] < lengths[:, None] # num_detections x max_len
            if settings['define_targets']:
                similarity_matrix_definitions = padded @ current_targets_definition_emb[target_idx, :].T[None] # num_detections x max_len x len(labels[i])
                similarity_matrix = np.concatenate([similarity_matrix_labels, similarity_matrix_definitions], axis=1) # num_detections x 2*max_len x len(labels[i])
                padding_mask = np.concatenate([padding_mask, padding_mask], axis=1) # num_detections x 2*max_len
                meta_score = np.sum(similarity_matrix * padding_mask[:, :, None], axis=1) # num_detections x len(labels[i])
                meta_score /= 2 * lengths[:, None] # num_detections x len(labels[i])
            else:
                meta_score = np.sum(similarity_matrix_labels * padding_mask[:, :, None], axis=1) # num_detections x len(labels[i])
                meta_score /= lengths[:, None] # num_detections x len(labels[i])

            valid = np.logical_and(meta_score >= -1.0, meta_score <= 1.0)
            if not np.all(valid):
                invalids = meta_score[~valid]
                self._logger.error(f"Encountered '{len(invalids)}' similarities outside of expected interval: {invalids}")

            if settings['allow_duplicate_labels']:
                order = np.argsort(-meta_score, axis=1) # num_detections x len(labels[i])
                matches_emb = [current_targets[target_idx[order[j, 0]]] for j in range(num_detections)]
                scores = [float(meta_score[j, order[j, 0]]) for j in range(num_detections)]
            else:
                row_ind, col_ind = linear_sum_assignment(meta_score, maximize=True)
                matches_emb = [None for _ in len(labels[i])] # for when the number of matching targets is smaller then the number of labels
                scores = [0.0] * len(labels[i])
                for j in range(len(row_ind)):
                    a_idx = row_ind[j]
                    b_idx = col_ind[j]
                    matches_emb[a_idx] = current_targets[target_idx[b_idx]]
                    scores[a_idx] = float(meta_score[a_idx, b_idx])

            metadata[i]['matches_embeddings'] = matches_emb
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'matches_embeddings', metadata[i]['matches_embeddings']))
            metadata[i]['matches_embeddings_score'] = scores
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'matches_embeddings_score', metadata[i]['matches_embeddings_score']))
            self._logger.debug(f"Matches (emb): {matches_emb} ({len(matches_emb)})")
            self._logger.debug(f"Scores: {scores} ({len(scores)})")

            # combine levenshtein and embedding matches
            if settings['allow_duplicate_labels']:
                matches = matches_emb
            else:
                matches = []
                for match_lev, match_emb in zip(matches_lev, matches_emb):
                    if match_lev is not None:
                        matches.append(match_lev)
                    elif match_emb is not None:
                        matches.append(match_emb)
                    else:
                        matches.append(None)

            # apply mapping rules
            mapping_rules_applied = 0
            if settings['mapping_rules'] is not None:
                for j, match in enumerate(matches):
                    if match in settings['mapping_rules'] and settings['mapping_rules'][match] in current_targets:
                        matches[j] = settings['mapping_rules'][match]
                        mapping_rules_applied += 1

            metadata[i]['mapping_rules_applied'] = mapping_rules_applied
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'mapping_rules_applied', metadata[i]['mapping_rules_applied']))

            metadata[i]['matches'] = matches
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'matches', metadata[i]['matches']))

            # distinguish dataset and augmented matches
            matches_dataset = []
            for match in matches:
                if match in current_dataset_targets:
                    matches_dataset.append(match)
                else:
                    matches_dataset.append(None)

            metadata[i]['matches_dataset'] = matches_dataset
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'matches_dataset', metadata[i]['matches_dataset']))

            results.append(matches_dataset)

        # retrieve usage
        if not settings['usage_skip']:
            if settings['usage_delay'] > 0.0:
                self._logger.debug(f"Sleeping '{settings['usage_delay']}s' before requesting usage")
                time.sleep(settings['usage_delay'])
            success, message, usage = self._node.api_director.get_usage(identifier=usage_uuid)
            if success:
                if usage.get('completions', {}).get('history') is not None:
                    del usage['completions']['history']
                for i in range(len(labels)):
                    metadata[i]['batch_usage'] = usage
                    self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'batch_usage', metadata[i]['batch_usage']))

        # log
        stamp_end = datetime.datetime.now()
        warnings_flat = []
        for i in range(len(labels)):
            if len(warnings[i]) > 0:
                warnings_flat += warnings[i]
                metadata[i]['warnings'] = warnings[i]
                self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'warnings', metadata[i]['warnings']))
            metadata[i]['stamp_batch_end'] = stamp_end.isoformat()
            metadata[i]['duration_batch'] = (stamp_end - stamp_start).total_seconds()
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'stamp_batch_end', metadata[i]['stamp_batch_end']))
            self._metadata_queue.put_nowait((settings['data_path'], settings['metadata_file'], identifiers[i], 'duration_batch', metadata[i]['duration_batch']))
        if len(warnings_flat) > 0:
            self._logger.warn(f"Encountered '{len(warnings_flat)}' warning{'s' if len(warnings_flat) != 1 else ''} in batch: {warnings_flat}")
        message = f"Successfully matched '{len(labels)}' set{'s' if len(labels) != 1 else ''} with a total of '{num_total_labels}' label{'s' if num_total_labels != 1 else ''} to '{len(matching_targets)}' matching target{'s' if len(matching_targets) != 1 else ''}"
        self._logger.info(message)
        if not some_input_is_batch:
            results = results[0]
            metadata = metadata[0]

        return True, message, results, metadata
