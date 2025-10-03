#!/usr/bin/env python3

# STANDARD

import os
import sys
import copy
import json
import traceback

import numpy as np

# ROS

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# CUSTOM

from vlm_gist.fiftyone.fiftyone_utils import import_fiftyone, load_dataset

from nimbro_utils.lazy import start_and_spin_node, SelfShutdown, Logger, ParameterHandler, read_json, write_json, escape

### <Parameter Defaults>

severity = 20

### </Parameter Defaults>

class FiftyOneEval(Node):

    def __init__(self, name="fiftyone_eval", *, context=None, dataset_paths, **kwargs):
        super().__init__(name, context=context, **kwargs)
        self._logger = Logger(self)

        self.dataset_paths = []
        for i, path in enumerate(dataset_paths):
            path = os.path.normpath(path)
            if os.path.isdir(path):
                self.dataset_paths.append(path)
            else:
                self._logger.error(f"Dataset '{path}' does not exist")
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
        for i, path in enumerate(self.dataset_paths):
            self._logger.info(f"Handling dataset '{i + 1}' of '{len(self.dataset_paths)}': '{path}'")
            try:
                fiftyone = import_fiftyone(logger=self._logger)
                dataset = load_dataset(fiftyone=fiftyone, dataset_path=path, logger=self._logger)

                results = {}
                results = self.eval_metadata(dataset=dataset, results=results, path=path)
                results = self.eval_dataset(dataset=dataset, results=results)

                # print partial results
                print_results = copy.deepcopy(results)
                try:
                    del print_results['images']
                except Exception:
                    pass
                try:
                    del print_results['results']['performance_metrics']['per_class']
                except Exception:
                    pass
                self._logger.info(f"Results (partial):\n{json.dumps(print_results, indent=4)}")

                # save evaluation results
                metadata_path = os.path.join(path, "evaluation.json")
                self._logger.info(f"Saving evaluation results to file '{metadata_path}'")
                write_json( # this will overwrite an existing file
                    file_path=metadata_path,
                    json_object=results,
                    indent=True,
                    logger=self._logger
                )

                self._logger.info(f"Visualize dataset: {escape['cyan']}ros2 run vlm_gist fiftyone_show -- {path}{escape['end']}")

                dataset.delete()
            except SelfShutdown as e:
                self._logger.error(f"{e}")
            except Exception as e:
                self._logger.error(f"{repr(e)}\n{traceback.format_exc()}")

        self._logger.info("Node stopped")

    def eval_metadata(self, dataset, results, path):
        # read metadata files

        self._logger.info("Reading metadata files")

        metadata = {}
        for key in ['descriptions', 'detections', 'validations', 'label_matches']:
            metadata_path = os.path.join(path, f"{key}.json")
            if os.path.isfile(metadata_path):
                success, _, metadata[key] = read_json(file_path=metadata_path, logger=self._logger)
            elif key != 'validations':
                self._logger.warn(f"Evaluation will be incomplete because the metadata file '{metadata_path}' does not exist")
            else:
                self._logger.info(f"Metadata file '{metadata_path}' does not exist")

        # obtain metadata metrics

        def nan_aware_sum(a, b):
            a = np.array(copy.deepcopy(a))
            b = np.array(copy.deepcopy(b))
            if len(a.shape) == 0 or len(b.shape) == 0:
                if len(a.shape) == 0 and len(b.shape) == 0:
                    if np.isnan(a):
                        return b.tolist()
                    elif np.isnan(b):
                        return a.tolist()
                    else:
                        return np.sum((a, b)).tolist()
                else:
                    raise ValueError(f"Shapes don't match: {a.shape} and {b.shape}")
            if len(a) == 0 and len(b) > 0:
                a = np.array([np.nan] * len(b))
            elif len(a) > 0 and len(b) == 0:
                b = np.array([np.nan] * len(a))
            elif len(a) != len(b):
                raise ValueError(f"Shapes don't match: {a.shape} and {b.shape}")
            mask_a_nan = np.isnan(a)
            mask_b_nan = np.isnan(b)
            both_nan = mask_a_nan & mask_b_nan
            none_nan = ~mask_a_nan & ~mask_b_nan
            c = np.where(mask_a_nan, b, a)
            c = np.where(mask_b_nan, a, c)
            c = np.where(both_nan, np.nan, c)
            c = np.where(none_nan, a + b, c)
            return c.tolist()

        def add_results(results, key, values):
            if np.all(np.isnan(values)):
                self._logger.warn(f"All values of key '{key}' are NaN")
            else:
                if 'images' not in results:
                    results['images'] = {}
                results['images'][f'{key}_NaN'] = np.count_nonzero(np.isnan(values)) / (max(len(values), 1))
                results['images'][f'{key}_median'] = float(np.nanmedian(values))
                results['images'][f'{key}_mean'] = float(np.nanmean(values))
                results['images'][f'{key}_std'] = float(np.nanstd(values))
                results['images'][f'{key}_max'] = float(np.nanmax(values))
                results['images'][f'{key}_min'] = float(np.nanmin(values))
            return results

        self._logger.info("Obtaining metadata metrics")

        if 'total' not in results:
            results['total'] = {'num_images': len(dataset)}

        results['total']['time_descriptions'] = np.nan
        for image in metadata.get('descriptions', []):
            results['total']['time_descriptions'] = metadata['descriptions'][image].get('duration_batch', np.nan)
            break
        results['total']['time_detections'] = np.nan
        for image in metadata.get('detections', []):
            results['total']['time_detections'] = metadata['detections'][image].get('duration_batch', np.nan)
            break
        results['total']['time_validations'] = np.nan
        for image in metadata.get('validations', []):
            results['total']['time_validations'] = metadata['validations'][image].get('duration_batch', np.nan)
            break
        results['total']['time_label_matches'] = np.nan
        for image in metadata.get('label_matches', []):
            results['total']['time_label_matches'] = metadata['label_matches'][image].get('duration_batch', np.nan)
            break
        if np.all(np.isnan([results['total']['time_descriptions'], results['total']['time_detections'], results['total']['time_validations'], results['total']['time_label_matches']])):
            results['total']['time_all'] = np.nan
        else:
            results['total']['time_all'] = float(np.nansum([results['total']['time_descriptions'], results['total']['time_detections'], results['total']['time_validations'], results['total']['time_label_matches']]))
        if 'validations' not in metadata:
            del results['total']['time_validations']

        described_instances = []
        failures_scene, failures_struct, failures_attr = [], [], []
        time_scene, time_struct, time_attr = [], [], []
        tokens_in_scene, tokens_out_scene, tokens_in_struct, tokens_out_struct, tokens_in_attr, tokens_out_attr = [], [], [], [], [], []
        dollars_in_scene, dollars_out_scene, dollars_in_struct, dollars_out_struct, dollars_in_attr, dollars_out_attr = [], [], [], [], [], []
        for image in metadata.get('descriptions', []):
            obj_list = metadata['descriptions'][image].get('structured_description')
            if obj_list is None:
                described_instances.append(np.nan)
            else:
                described_instances.append(len(obj_list))

            failures_scene.append(len(metadata['descriptions'][image].get('failed_scene_description_completions', [])))
            failures_struct.append(len(metadata['descriptions'][image].get('failed_structured_description_completions', [])))
            failures_attr.append(len(metadata['descriptions'][image].get('failed_decoupled_attribution_completions', [])))

            time_scene.append(metadata['descriptions'][image].get('duration_scene_description', np.nan))
            time_struct.append(metadata['descriptions'][image].get('duration_structured_description', np.nan))
            time_attr.append(metadata['descriptions'][image].get('duration_decoupled_attribution', np.nan))

            tokens_in_scene.append(metadata['descriptions'][image].get('usage', {}).get('scene_description', {}).get('tokens_input', np.nan))
            tokens_out_scene.append(metadata['descriptions'][image].get('usage', {}).get('scene_description', {}).get('tokens_output', np.nan))
            tokens_in_struct.append(metadata['descriptions'][image].get('usage', {}).get('structured_description', {}).get('tokens_input', np.nan))
            tokens_out_struct.append(metadata['descriptions'][image].get('usage', {}).get('structured_description', {}).get('tokens_output', np.nan))
            tokens_in_attr.append(metadata['descriptions'][image].get('usage', {}).get('decoupled_attribution', {}).get('tokens_input', np.nan))
            tokens_out_attr.append(metadata['descriptions'][image].get('usage', {}).get('decoupled_attribution', {}).get('tokens_output', np.nan))

            dollars_in_scene.append(metadata['descriptions'][image].get('usage', {}).get('scene_description', {}).get('dollars_input', np.nan))
            dollars_out_scene.append(metadata['descriptions'][image].get('usage', {}).get('scene_description', {}).get('dollars_output', np.nan))
            dollars_in_struct.append(metadata['descriptions'][image].get('usage', {}).get('structured_description', {}).get('dollars_input', np.nan))
            dollars_out_struct.append(metadata['descriptions'][image].get('usage', {}).get('structured_description', {}).get('dollars_output', np.nan))
            dollars_in_attr.append(metadata['descriptions'][image].get('usage', {}).get('decoupled_attribution', {}).get('dollars_input', np.nan))
            dollars_out_attr.append(metadata['descriptions'][image].get('usage', {}).get('decoupled_attribution', {}).get('dollars_output', np.nan))

        detections, missing_prompts, over_detections = [], [], []
        time_detection, time_segmentation = [], []
        for image in metadata.get('detections', []):
            detections.append(metadata['detections'][image].get('num_detections', np.nan))
            missing_prompts.append(metadata['detections'][image].get('num_missing_prompts', np.nan))
            over_detections.append(metadata['detections'][image].get('num_over_detections', np.nan))
            time_detection.append(metadata['detections'][image].get('duration_inference_detection', np.nan))
            time_segmentation.append(metadata['detections'][image].get('duration_inference_segmentation', np.nan))

        validation_keeps, validation_updates, validation_rejects, validation_errors = [], [], [], []
        time_validation_mean, time_validation_max, time_validation_min = [], [], []
        tokens_in_validation, tokens_out_validation = [], []
        dollars_in_validation, dollars_out_validation = [], []
        for image in metadata.get('validations', []):
            validations_total = metadata['validations'][image].get('summary', {}).get('labels')
            if validations_total is not None:
                validations_total = len(validations_total)
                if validations_total is None or metadata['validations'][image].get('summary', {}).get('keep') is None:
                    validation_keeps.append(np.nan)
                else:
                    validation_keeps.append(len(metadata['validations'][image]['summary']['keep']) / validations_total)
                if validations_total is None or metadata['validations'][image].get('summary', {}).get('update') is None:
                    validation_updates.append(np.nan)
                else:
                    validation_updates.append(len(metadata['validations'][image]['summary']['update']) / validations_total)
                if validations_total is None or metadata['validations'][image].get('summary', {}).get('reject') is None:
                    validation_rejects.append(np.nan)
                else:
                    validation_rejects.append(len(metadata['validations'][image]['summary']['reject']) / validations_total)
                if validations_total is None or metadata['validations'][image].get('summary', {}).get('error') is None:
                    validation_errors.append(np.nan)
                else:
                    validation_errors.append(len(metadata['validations'][image]['summary']['error']) / validations_total)

            time_validation_mean.append(metadata['validations'][image].get('summary', {}).get('duration_validation_mean', np.nan))
            time_validation_max.append(metadata['validations'][image].get('summary', {}).get('duration_validation_max', np.nan))
            time_validation_min.append(metadata['validations'][image].get('summary', {}).get('duration_validation_min', np.nan))

            tokens_in_validation.append(metadata['validations'][image].get('summary', {}).get('usage', {}).get('tokens_input', np.nan))
            tokens_out_validation.append(metadata['validations'][image].get('summary', {}).get('usage', {}).get('tokens_output', np.nan))

            dollars_in_validation.append(metadata['validations'][image].get('summary', {}).get('usage', {}).get('dollars_input', np.nan))
            dollars_out_validation.append(metadata['validations'][image].get('summary', {}).get('usage', {}).get('dollars_output', np.nan))

        matches_accept, matches_reject, matches_conf = [], [], []
        for image in metadata.get('label_matches', []):
            if 'matches_dataset' in metadata['label_matches'][image]:
                matches_accept.append(len([item for item in metadata['label_matches'][image]['matches_dataset'] if item is not None]))
                matches_reject.append(len([item for item in metadata['label_matches'][image]['matches_dataset'] if item is None]))
                if 'matches_embeddings_score' in metadata['label_matches'][image]:
                    matches_conf.append(np.mean(metadata['label_matches'][image]['matches_embeddings_score']))
                else:
                    matches_conf.append(np.nan)
            else:
                matches_accept.append(np.nan)
                matches_reject.append(np.nan)
                matches_conf.append(np.nan)

        validation_keeps_and_updates = nan_aware_sum(validation_keeps, validation_updates)
        validation_rejects_and_errors = nan_aware_sum(validation_rejects, validation_errors)
        time_description = nan_aware_sum(time_scene, nan_aware_sum(time_struct, time_attr))
        time_detection_and_segmentation = nan_aware_sum(time_detection, time_segmentation)
        time_all = nan_aware_sum(nan_aware_sum(time_description, time_detection_and_segmentation), time_validation_mean)
        tokens_in_description = nan_aware_sum(nan_aware_sum(tokens_in_scene, tokens_in_struct), tokens_in_attr)
        tokens_out_description = nan_aware_sum(nan_aware_sum(tokens_out_scene, tokens_out_struct), tokens_out_attr)
        tokens_in_all = nan_aware_sum(tokens_in_description, tokens_in_validation)
        tokens_out_all = nan_aware_sum(tokens_out_description, tokens_out_validation)
        dollars_scene_description = nan_aware_sum(dollars_in_scene, dollars_out_scene)
        dollars_structured_description = nan_aware_sum(dollars_in_struct, dollars_out_struct)
        dollars_decoupled_attribution = nan_aware_sum(dollars_in_attr, dollars_out_attr)
        dollars_in_description = nan_aware_sum(nan_aware_sum(dollars_in_scene, dollars_in_struct), dollars_in_attr)
        dollars_out_description = nan_aware_sum(nan_aware_sum(dollars_out_scene, dollars_out_struct), dollars_out_attr)
        dollars_description = nan_aware_sum(dollars_in_description, dollars_out_description)
        dollars_validation = nan_aware_sum(dollars_in_validation, dollars_out_validation)
        dollars_in_all = nan_aware_sum(dollars_in_description, dollars_in_validation)
        dollars_out_all = nan_aware_sum(dollars_out_description, dollars_out_validation)
        dollars_all = nan_aware_sum(dollars_in_all, dollars_out_all)

        if np.all(np.isnan(tokens_in_description)):
            results['total']['tokens_in_descriptions'] = np.nan
        else:
            results['total']['tokens_in_descriptions'] = int(np.nansum(tokens_in_description))
        if np.all(np.isnan(tokens_out_description)):
            results['total']['tokens_out_descriptions'] = np.nan
        else:
            results['total']['tokens_out_descriptions'] = int(np.nansum(tokens_out_description))
        if 'validations' in metadata:
            if np.all(np.isnan(tokens_in_validation)):
                results['total']['tokens_in_validations'] = np.nan
            else:
                results['total']['tokens_in_validations'] = int(np.nansum(tokens_in_validation))
            if np.all(np.isnan(tokens_out_validation)):
                results['total']['tokens_out_validations'] = np.nan
            else:
                results['total']['tokens_out_validations'] = int(np.nansum(tokens_out_validation))
        if np.all(np.isnan(tokens_in_all)):
            results['total']['tokens_in_all'] = np.nan
        else:
            results['total']['tokens_in_all'] = int(np.nansum(tokens_in_all))
        if np.all(np.isnan(tokens_out_all)):
            results['total']['tokens_out_all'] = np.nan
        else:
            results['total']['tokens_out_all'] = int(np.nansum(tokens_out_all))
        if np.all(np.isnan(dollars_description)):
            results['total']['dollars_descriptions'] = np.nan
        else:
            results['total']['dollars_descriptions'] = float(np.nansum(dollars_description))
        if 'validations' in metadata:
            if np.all(np.isnan(dollars_validation)):
                results['total']['dollars_validations'] = np.nan
            else:
                results['total']['dollars_validations'] = float(np.nansum(dollars_validation))
        if np.all(np.isnan(dollars_all)):
            results['total']['dollars_all'] = np.nan
        else:
            results['total']['dollars_all'] = float(np.nansum(dollars_all))
        results['total']['dollars_label_matching'] = np.nan
        for image in metadata.get('label_matches', []):
            if 'batch_usage' in metadata['label_matches'][image]:
                results['total']['dollars_label_matching'] = metadata['label_matches'][image]['batch_usage'].get('dollars_total', np.nan)
            break
        results['total']['dollars_all_and_label_matching'] = nan_aware_sum(results['total']['dollars_label_matching'], results['total']['dollars_all'])

        results = add_results(results, 'failures_scene_description', failures_scene)
        results = add_results(results, 'failures_structured_description', failures_struct)
        results = add_results(results, 'failures_decoupled_attribution', failures_attr)

        results = add_results(results, 'described_instances', described_instances)
        results = add_results(results, 'detected_instances', detections)
        results = add_results(results, 'missed_prompts', missing_prompts)
        results = add_results(results, 'over_detections', over_detections)

        results = add_results(results, 'validation_keeps', validation_keeps)
        results = add_results(results, 'validation_updates', validation_updates)
        results = add_results(results, 'validation_keeps_and_updates', validation_keeps_and_updates)
        results = add_results(results, 'validation_rejects', validation_rejects)
        results = add_results(results, 'validation_errors', validation_errors)
        results = add_results(results, 'validation_rejects_and_errors', validation_rejects_and_errors)

        results = add_results(results, 'label_matching_accept', matches_accept)
        results = add_results(results, 'label_matching_reject', matches_reject)
        results = add_results(results, 'label_matching_confidence', matches_conf)

        results = add_results(results, 'time_scene_description', time_scene)
        results = add_results(results, 'time_structured_description', time_struct)
        results = add_results(results, 'time_decoupled_attribution', time_attr)
        results = add_results(results, 'time_description', time_description)
        results = add_results(results, 'time_detection', time_detection)
        results = add_results(results, 'time_segmentation', time_segmentation)
        results = add_results(results, 'time_detection_and_segmentation', time_detection_and_segmentation)
        results = add_results(results, 'time_validation_mean', time_validation_mean)
        results = add_results(results, 'time_validation_max', time_validation_max)
        results = add_results(results, 'time_validation_min', time_validation_min)
        results = add_results(results, 'time_all', time_all)

        results = add_results(results, 'tokens_in_scene_description', tokens_in_scene)
        results = add_results(results, 'tokens_out_scene_description', tokens_out_scene)
        results = add_results(results, 'tokens_in_structured_description', tokens_in_struct)
        results = add_results(results, 'tokens_out_structured_description', tokens_out_struct)
        results = add_results(results, 'tokens_in_decoupled_attribution', tokens_in_attr)
        results = add_results(results, 'tokens_out_decoupled_attribution', tokens_out_attr)
        results = add_results(results, 'tokens_in_description', tokens_in_description)
        results = add_results(results, 'tokens_out_description', tokens_out_description)
        results = add_results(results, 'tokens_in_validation', tokens_in_validation)
        results = add_results(results, 'tokens_out_validation', tokens_out_validation)
        results = add_results(results, 'tokens_in_all', tokens_in_all)
        results = add_results(results, 'tokens_out_all', tokens_out_all)

        results = add_results(results, 'dollars_in_scene_description', dollars_in_scene)
        results = add_results(results, 'dollars_out_scene_description', dollars_out_scene)
        results = add_results(results, 'dollars_scene_description', dollars_scene_description)
        results = add_results(results, 'dollars_in_structured_description', dollars_in_struct)
        results = add_results(results, 'dollars_out_structured_description', dollars_out_struct)
        results = add_results(results, 'dollars_structured_description', dollars_structured_description)
        results = add_results(results, 'dollars_in_decoupled_attribution', dollars_in_attr)
        results = add_results(results, 'dollars_out_decoupled_attribution', dollars_out_attr)
        results = add_results(results, 'dollars_decoupled_attribution', dollars_decoupled_attribution)
        results = add_results(results, 'dollars_in_description', dollars_in_description)
        results = add_results(results, 'dollars_out_description', dollars_out_description)
        results = add_results(results, 'dollars_description', dollars_description)
        results = add_results(results, 'dollars_in_validation', dollars_in_validation)
        results = add_results(results, 'dollars_out_validation', dollars_out_validation)
        results = add_results(results, 'dollars_validation', dollars_validation)
        results = add_results(results, 'dollars_in_all', dollars_in_all)
        results = add_results(results, 'dollars_out_all', dollars_out_all)
        results = add_results(results, 'dollars_all', dollars_all)

        return results

    def eval_dataset(self, dataset, results):
        schema = dataset.get_field_schema()

        # obtain groundtruth and detection metrics

        if 'results' not in results:
            results['results'] = {}

        self._logger.info("Obtaining groundtruth and detection metrics")

        for key in ['ground_truth', 'detections']:
            if key not in schema:
                self._logger.warn(f"The dataset does not contain the field '{key}'")
                continue

            if key not in results['results']:
                results['results'][key] = {}

            results['results'][key]['annotations'] = dataset.count(f"{key}.detections")
            results['results'][key]['annotations_unique'] = len(dataset.distinct(f"{key}.detections.label"))

            counts = [len(dets) if dets else 0 for dets in dataset.values(f"{key}.detections")]
            results['results'][key]['annotations_mean'] = float(np.mean(counts))
            results['results'][key]['annotations_std'] = float(np.std(counts))
            results['results'][key]['annotations_max'] = int(np.max(counts))
            results['results'][key]['annotations_min'] = int(np.min(counts))

            areas = np.array([det.bounding_box[2] * det.bounding_box[3] for dets in dataset.values(f"{key}.detections") if dets for det in dets])
            results['results'][key]['area_mean'] = float(np.mean(areas))
            results['results'][key]['area_std'] = float(np.std(areas))
            results['results'][key]['area_max'] = float(np.max(areas))
            results['results'][key]['area_min'] = float(np.min(areas))

            areas = [[det.bounding_box[2] * det.bounding_box[3] for det in dets] if dets else [] for dets in dataset.values(f"{key}.detections")]
            areas_accumulated = np.array([sum(image_areas) for image_areas in areas])
            results['results'][key]['area_accumulated_mean'] = float(np.mean(areas_accumulated))
            results['results'][key]['area_accumulated_std'] = float(np.std(areas_accumulated))
            results['results'][key]['area_accumulated_max'] = float(np.max(areas_accumulated))
            results['results'][key]['area_accumulated_min'] = float(np.min(areas_accumulated))

        # obtain performance metrics

        self._logger.info("Checking dataset validity")
        for sample in dataset.iter_samples(progress=True):
            for field in ['ground_truth', 'detections']:
                if field in sample:
                    if not sample[field] is None:
                        for detection in sample[field].detections:
                            assert detection.bounding_box[2] > 0, f"{os.path.basename(sample.filepath)} - {field}: {detection.bounding_box}"
                            assert detection.bounding_box[3] > 0, f"{os.path.basename(sample.filepath)} - {field}: {detection.bounding_box}"

        if 'ground_truth' not in schema or 'detections' not in schema:
            self._logger.warn("Can't obtain performance metrics without both fields fields 'ground_truth' and 'detections'")
        else:
            self._logger.info("Obtaining performance metrics")
            try:
                eval_result = dataset.evaluate_detections(
                    pred_field='detections',
                    gt_field='ground_truth',
                    use_masks=False,
                    # use_boxes=True,
                    # classes=dataset.distinct("ground_truth.detections.label"),
                    classes=None,
                    compute_mAP=True,
                    # method="coco",
                    # iou=0.5,
                    eval_key="eval",
                )
                map_computed = True
            except Exception as e:
                self._logger.warn(f"Failed to comptue mAP, falling back too not computing mAP: {repr(e)}")
                eval_result = dataset.evaluate_detections(
                    pred_field='detections',
                    gt_field='ground_truth',
                    use_masks=False,
                    # use_boxes=True,
                    # classes=dataset.distinct("ground_truth.detections.label"),
                    classes=None,
                    compute_mAP=False,
                    # method="coco",
                    # iou=0.5,
                    eval_key="eval",
                )
                map_computed = False

            eval_patches = dataset.to_evaluation_patches("eval")
            counts = eval_patches.count_values("type")
            metrics = eval_result.metrics()

            true_positives = counts.get("tp", 0)
            false_positives = counts.get("fp", 0)
            false_negatives = counts.get("fn", 0)
            mAP = float(eval_result.mAP()) if map_computed else None
            mAR = float(eval_result.mAR()) if map_computed else None
            accuracy = float(metrics['accuracy'])
            precision = float(metrics['precision'])
            recall = float(metrics['recall'])
            fscore = float(metrics['fscore'])
            support = metrics['support']
            per_class_metrics = eval_result.report()

            results['results']['performance_metrics'] = {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'mAP': mAP,
                'mAR': mAR,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'fscore': fscore,
                'support': support,
                'per_class': per_class_metrics
            }

        return results

def main():
    if len(sys.argv) < 2:
        print("Usage: ros2 run vlm_gist fiftyone_eval -- <dataset_path_1> [dataset_path_2 ...]")
    else:
        dataset_paths = sys.argv[1:]
        start_and_spin_node(FiftyOneEval, node_args={'dataset_paths': dataset_paths})

if __name__ == '__main__':
    main()
