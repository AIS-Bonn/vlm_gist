#!/usr/bin/env python3

# STANDARD

import os
import copy
import datetime
import collections
from queue import Queue

import numpy as np

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.core.common import read_image, set_logger, set_settings, complete_settings, save_metadata

from nimbro_api import ApiDirector
from nimbro_utils.lazy import decode_mask, encode_mask, get_package_path

default_settings = {
    'model_name': "florence2", # florence2, kosmos2
    'model_id': 0,
    'model_flavor': "large", # florence2: [base, large, base_ft, large_ft; kosmos2: patch14-224
    # 'model_prompt': {'task_prompt': "<OD>", 'prompt_args': None},
    'model_prompt': {'task_prompt': "<DENSE_REGION_CAPTION>", 'prompt_args': None},
    # 'model_name': "kosmos2", # florence2, kosmos2
    # 'model_flavor': "patch14-224", # florence2: [base, large, base_ft, large_ft; kosmos2: patch14-224
    # 'model_prompt': "<grounding> Describe this image in detail:",
    'model_num_beams': 3,
    'model_max_new_tokens': 1024,
    'model_max_batch_size': 6,
    'segmentation_name': "sam2_realtime", # only 'sam2_realtime' is implemented
    'segmentation_model_id': 0,
    'segmentation_model_id_secondary': None, # set None to deactivate
    'segmentation_flavor': "large", # tiny, small, base, large
    'segmentation_track': True, # uses the tracker once to refine masks instead of forwarding masks from track update
    'segmentation_use_bbox': True, # forward bounding boxes around masks as main bounding box output instead of detections
    'retry': True, # retry API requests after failure instead of skipping the current image
    'leave_images_in_place': False, # attempt to not copy images passed as string (local path or web)
    'keep_image_name': False, # keep original image name when copying instead of using a timestamp based name
    'png_compression_level': 2, # 0 (off) to 9 (max)
    'png_max_pixels': 1920 * 1080, # set None to deactivate re-scaling
    'crop_masks': True, # compresses masks by cropping them using their bounding boxes
    'data_path': os.path.join(get_package_path("vlm_gist"), "data", "baselines"),
    'image_folder': "baseline_edits", # folder within data_path to store images; set None to use data_path directly
    'metadata_file': "baselines.json", # set None to deactivate
    'metadata_write_relative_paths': False, # write relative path between metadata and image instead of absolute image paths to metadata
    'metadata_write_no_paths': True # only write image names to metadata; if True this overwrites 'metadata_write_relative_paths'
}

class Baseline:

    def __init__(self, node, metadata_timer=True, settings=None, logger_name=None, logger_severity=20):
        assert isinstance(node, Node), f"Provided argument 'node' is of invalid type '{type(node).__name__}'. Supported type is 'rclpy.node.Node'."
        assert isinstance(metadata_timer, bool), f"Provided argument 'metadata_timer' is of unsupported type '{type(metadata_timer).__name__}'. Supported type is 'bool'."
        assert logger_name is None or isinstance(logger_name, str), f"Provided argument 'logger_name' is of unsupported type '{type(logger_name).__name__}'. Supported types are 'None' and 'str'."

        self._node = node

        # logger
        if logger_name is None:
            logger_name = (f"{self._node.get_namespace()}.{self._node.get_name()}.baseline").replace("/", "")
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

        # ApiDirector
        if hasattr(self._node, 'api_director'):
            assert isinstance(self._node.api_director, ApiDirector), f"Expected existing attribute 'api_director' of parent node to be of type 'nimbro_api.api_director.ApiDirector' but it is of type '{type(self._node.api_director).__name__}'!"
        else:
            self._logger.debug("Adding 'ApiDirector' to parent node")
            self._node.api_director = ApiDirector(self._node, {'severity': 30})

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
            return False, message, None, None, None, None, None
        elif not len(settings) == num_images:
            message = f"Expected number of settings '{len(settings)}' to match the number of images '{num_images}'."
            self._logger.error(message)
            return False, message, None, None, None, None, None
        for i in range(len(settings)):
            if settings[i] is None:
                settings[i] = copy.deepcopy(self._settings)
            else:
                success, message, settings[i] = self.complete_settings(settings[i])
                if not success:
                    return False, message, None, None, None, None, None

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
            success, message, image, path = read_image(
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
                return False, message, None, None, None, None, None
        self._logger.debug(f"Absolute image paths: {image_paths}")
        self._logger.debug(f"Metadata image paths: {image_paths_metadata}")
        image_paths_metadata_counter = dict(collections.Counter(image_paths_metadata))
        for key in copy.deepcopy(image_paths_metadata_counter):
            if image_paths_metadata_counter[key] == 1:
                del image_paths_metadata_counter[key]
        if len(image_paths_metadata_counter) > 0:
            message = f"The image paths/names supposed to be written to metadata according to settings are not unique: {image_paths_metadata_counter}"
            self._logger.error(message)
            return False, message, None, None, None, None, None

        metadata = [None] * len(images)
        for i, path in enumerate(image_paths_metadata):
            metadata[i] = {
                'stamp_batch_start': stamp_start.isoformat(),
                'settings': settings[i],
                'batch_size': len(images)
            }
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], path, 'stamp_batch_start', metadata[i]['stamp_batch_start']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], path, 'settings', settings[i]))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], path, 'batch_size', len(images)))
            if len(images) > 1:
                metadata[i]['batch_images'] = image_paths_metadata
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], path, 'batch_images', image_paths_metadata))

        # detect

        if len(images) > 1:
            self._logger.info(f"Processing '{len(images)}' image{'s' if len(images) != 1 else ''}")

        labels, bboxes, bboxes_detection, bboxes_mask, masks_list, masks_b64, track_ids = [], [], [], [], [], [], []
        for i in range(len(images)):
            metadata[i]['stamp_image_start'] = datetime.datetime.now().isoformat()
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_image_start', metadata[i]['stamp_image_start']))
            self._logger.info(f"Processing image '{i + 1}' of '{len(images)}' ({image_paths[i]})")

            # detection

            metadata[i]['stamp_inference_detection_start'] = datetime.datetime.now().isoformat()
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_inference_detection_start', metadata[i]['stamp_inference_detection_start']))

            if settings[i]['model_name'] == "florence2":
                success, message, detections, _ = self._node.api_director.florence2(
                    image=image_paths[i],
                    prompt=settings[i]['model_prompt'],
                    model_flavor=settings[i]['model_flavor'],
                    num_beams=settings[i]['model_num_beams'],
                    max_new_tokens=settings[i]['model_max_new_tokens'],
                    max_batch_size=settings[i]['model_max_batch_size'],
                    retry=settings[i]['retry']
                )
            elif settings[i]['model_name'] == "kosmos2":
                success, message, detections, _ = self._node.api_director.kosmos2(
                    image=image_paths[i],
                    prompt=settings[i]['model_prompt'],
                    model_id=settings[i]['model_id'],
                    model_flavor=settings[i]['model_flavor'],
                    num_beams=settings[i]['model_num_beams'],
                    max_new_tokens=settings[i]['model_max_new_tokens'],
                    max_batch_size=settings[i]['model_max_batch_size'],
                    retry=settings[i]['retry']
                )
            else:
                raise NotImplementedError(f"Model '{settings[i]['model_name']}' is not implemented.")
            if success:
                self._logger.info(f"Detected '{len(detections)}' object{'s' if len(detections) else ''}")
            else:
                metadata[i]['stamp_error'] = datetime.datetime.now().isoformat()
                metadata[i]['report_error'] = message
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_error', metadata[i]['stamp_error']))
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_error', message))
                continue

            metadata[i]['stamp_inference_detection_end'] = datetime.datetime.now().isoformat()
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_inference_detection_end', metadata[i]['stamp_inference_detection_end']))
            metadata[i]['duration_inference_detection'] = (datetime.datetime.fromisoformat(metadata[i]['stamp_inference_detection_end']) - datetime.datetime.fromisoformat(metadata[i]['stamp_inference_detection_start'])).total_seconds()
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'duration_inference_detection', metadata[i]['duration_inference_detection']))

            # segmentation
            segmentations = None
            if len(detections) > 0 and settings[i]['segmentation_name'] is not None:
                metadata[i]['stamp_inference_segmentation_start'] = datetime.datetime.now().isoformat()
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_inference_segmentation_start', metadata[i]['stamp_inference_segmentation_start']))

                sam_prompts = [{'object_id': j, 'bbox': det['box_xyxy']} for j, det in enumerate(detections)]

                if settings[i]['segmentation_name'] == "sam2_realtime":
                    success, message, segmentations = self._node.api_director.sam2_realtime_update(
                        image=image_paths[i],
                        prompts=sam_prompts,
                        model_id=settings[i]['segmentation_model_id'],
                        model_flavor=settings[i]['segmentation_flavor'],
                        retry=settings[i]['retry']
                    )
                    if success:
                        self._logger.info(f"Segmented '{len(segmentations)}' detection{'s' if len(segmentations) != 1 else ''}")
                    else:
                        metadata[i]['stamp_error'] = datetime.datetime.now().isoformat()
                        metadata[i]['report_error'] = message
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_error', metadata[i]['stamp_error']))
                        self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_error', message))
                        continue

                    if settings[i]['segmentation_track']:
                        success, message, segmentations = self._node.api_director.sam2_realtime_track(
                            image=image_paths[i],
                            model_id=settings[i]['segmentation_model_id'],
                            retry=settings[i]['retry']
                        )
                        if success:
                            self._logger.info(f"Tracked '{len(segmentations)}' detection{'s' if len(segmentations) != 1 else ''}")
                        else:
                            metadata[i]['stamp_error'] = datetime.datetime.now().isoformat()
                            metadata[i]['report_error'] = message
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_error', metadata[i]['stamp_error']))
                            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'report_error', message))
                            continue
                else:
                    raise NotImplementedError(f"Segmentation model '{settings[i]['segmentation_name']}' is not implemented.")

                metadata[i]['stamp_inference_segmentation_end'] = datetime.datetime.now().isoformat()
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_inference_segmentation_end', metadata[i]['stamp_inference_segmentation_end']))
                metadata[i]['duration_inference_segmentation'] = (datetime.datetime.fromisoformat(metadata[i]['stamp_inference_segmentation_end']) - datetime.datetime.fromisoformat(metadata[i]['stamp_inference_segmentation_start'])).total_seconds()
                self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'duration_inference_segmentation', metadata[i]['duration_inference_segmentation']))

            # secondary tracker
            if settings[i]['segmentation_model_id_secondary'] is not None:
                if settings[i]['segmentation_name'] == "sam2_realtime":
                    self._node.api_director.sam2_realtime_2_update(
                        image=image_paths[i],
                        prompts=sam_prompts,
                        model_id=settings[i]['segmentation_model_id_secondary'],
                        model_flavor=settings[i]['segmentation_flavor'],
                        retry=settings[i]['retry']
                    )
                    if settings[i]['segmentation_track']:
                        self._node.api_director.sam2_realtime_2_track(
                            image=image_paths[i],
                            model_id=settings[i]['segmentation_model_id_secondary'],
                            retry=settings[i]['retry']
                        )

            # construct outputs

            labels_image, bboxes_detection_image, bboxes_mask_image, masks_image_list, masks_image_b64, track_ids_image = [], [], [], [], [], []
            for j, detection in enumerate(detections):
                labels_image.append(detection['label'])
                bboxes_detection_image.append(detection['box_xyxy'])
                if segmentations is not None:
                    bboxes_mask_image.append(segmentations[j]['box_xyxy'])
                    masks_image_list.append(decode_mask(segmentations[j]['mask']).tolist())
                    masks_image_b64.append(segmentations[j]['mask'])
                    track_ids_image.append(segmentations[j]['track_id'])

            labels.append(labels_image)
            bboxes_detection.append(bboxes_detection_image)
            if segmentations is None:
                bboxes_mask.append(None)
                masks_list.append(None)
                masks_b64.append(None)
                track_ids.append(None)
            else:
                bboxes_mask.append(bboxes_mask_image)
                masks_list.append(masks_image_list)
                masks_b64.append(masks_image_b64)
                track_ids.append(track_ids_image)

            if segmentations is not None and settings[i]['segmentation_use_bbox']:
                bboxes.append(bboxes_mask_image)
            else:
                bboxes.append(bboxes_detection_image)

            metadata[i]['num_detections'] = len(detections)
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'num_detections', metadata[i]['num_detections']))

            metadata[i]['labels'] = labels[-1]
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'labels', metadata[i]['labels']))

            metadata[i]['bboxes_detection'] = bboxes_detection[-1]
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'bboxes_detection', metadata[i]['bboxes_detection']))

            metadata[i]['bboxes_mask'] = bboxes_mask[-1]
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'bboxes_mask', metadata[i]['bboxes_mask']))

            if settings[i]['crop_masks']:
                metadata[i]['masks_cropped'] = True
            else:
                metadata[i]['masks_cropped'] = False
                image_shape = (images[i].shape[0], images[i].shape[1])
                for j, (bbox, mask) in enumerate(zip(bboxes_mask[-1], masks_list[-1])):
                    full_mask = np.zeros(image_shape, dtype=bool)
                    x0, y0, x1, y1 = bbox
                    full_mask[y0:y1, x0:x1] = mask
                    masks_list[-1][j] = full_mask.tolist()
                    masks_b64[-1][j] = encode_mask(full_mask)
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'masks_cropped', metadata[i]['masks_cropped']))

            metadata[i]['masks'] = masks_b64[-1]
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'masks', metadata[i]['masks']))

            metadata[i]['bboxes'] = bboxes[-1]
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'bboxes', metadata[i]['bboxes']))

            metadata[i]['track_ids'] = track_ids[-1]
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'track_ids', metadata[i]['track_ids']))

            metadata[i]['stamp_image_end'] = datetime.datetime.now().isoformat()
            metadata[i]['duration_image'] = (datetime.datetime.fromisoformat(metadata[i]['stamp_image_end']) - datetime.datetime.fromisoformat(metadata[i]['stamp_image_start'])).total_seconds()
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'stamp_image_end', metadata[i]['stamp_image_end']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], image_paths_metadata[i], 'duration_image', metadata[i]['duration_image']))

        failures = sum(['stamp_error' in metadata[i] for i in range(len(images))])
        success = failures < len(images)
        successes = len(images) - failures

        # log
        if failures == 0:
            message = f"Successfully processed '{len(images)}' image{'s' if len(images) != 1 else ''}."
            self._logger.info(message)
        elif not success:
            message = f"Failed to process '{len(images)}' image{'s' if len(images) != 1 else ''}."
            self._logger.error(message)
        else:
            message = f"Successfully processed '{len(successes)}' image{'s' if len(successes) != 1 else ''} and failed to process '{len(failures)}' image{'s' if len(failures) != 1 else ''}."
            self._logger.warn(message)

        stamp_end = datetime.datetime.now()
        for i, path in enumerate(image_paths_metadata):
            metadata[i]['stamp_batch_end'] = stamp_end.isoformat()
            metadata[i]['duration_batch'] = (stamp_end - stamp_start).total_seconds()
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], path, 'stamp_batch_end', metadata[i]['stamp_batch_end']))
            self._metadata_queue.put_nowait((settings[i]['data_path'], settings[i]['metadata_file'], path, 'duration_batch', metadata[i]['duration_batch']))

        if no_batch:
            labels = labels[0]
            bboxes = bboxes[0]
            masks_list = masks_list[0]
            track_ids = track_ids[0]
            metadata = metadata[0]

        return success, message, labels, bboxes, masks_list, track_ids, metadata
