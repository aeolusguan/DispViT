# Copyright (c) Facebook, Inc. and its affiliates.

from collections import OrderedDict, abc
import functools
import itertools
import logging

import numpy as np
import torch
from hydra.utils import instantiate
from torch import distributed as dist

from .data.worker_fn import get_rank, get_world_size


def parse_validation_datasets(dataset_configs):
    """Normalize bare names and {dataset_name: max_disp} entries into evaluator specs."""
    if dataset_configs is None:
        return []
    if not isinstance(dataset_configs, abc.Sequence) or isinstance(dataset_configs, (str, bytes)):
        raise ValueError(
            "data.val_datasets must be a list of dataset names or single-key mappings, or null."
        )

    specs = []
    for index, entry in enumerate(dataset_configs):
        if isinstance(entry, str):
            dataset_name, overrides = entry, {}
        elif isinstance(entry, abc.Mapping):
            if "name" in entry or "max_disp" in entry:
                raise ValueError(
                    f"data.val_datasets[{index}] uses the removed name/max_disp record format; "
                    "use a single-key mapping such as {things: 192}, or a bare dataset name."
                )
            if len(entry) != 1:
                raise ValueError(f"data.val_datasets[{index}] must contain exactly one dataset name.")
            dataset_name, max_disp = next(iter(entry.items()))
            if max_disp is not None and (
                isinstance(max_disp, bool) or not isinstance(max_disp, (int, float))
            ):
                raise ValueError(f"data.val_datasets[{index}] max_disp must be a number or null.")
            overrides = {"max_disp": max_disp}
        else:
            raise ValueError(
                f"data.val_datasets[{index}] must be a dataset name or a single-key mapping."
            )
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError(f"data.val_datasets[{index}] must have a non-empty string dataset name.")
        specs.append((dataset_name, overrides))
    return specs


def validate_evaluation_config(eval_conf, dataset_configs):
    """Validate and normalize evaluation settings before model or distributed setup."""
    if eval_conf is not None and "max_disp" in eval_conf:
        raise ValueError(
            "eval.max_disp is no longer supported; use single-key entries such as "
            "{things: 192} under data.val_datasets."
        )
    return parse_validation_datasets(dataset_configs)


def build_evaluators(eval_conf, dataset_configs):
    """Build evaluators from compact validation entries, preserving omitted overrides."""
    specs = validate_evaluation_config(eval_conf, dataset_configs)
    evaluators = []
    for dataset_name, overrides in specs:
        evaluator = instantiate(eval_conf, _recursive_=False, **overrides)
        if isinstance(evaluator, DispEvaluator):
            logging.getLogger(__name__).info(
                "Evaluation max_disp for %s: %s", dataset_name, evaluator._max_disp
            )
        evaluators.append(evaluator)
    return evaluators


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron2,
    so that they are easy to copypaste into a spreadsheet.


    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    assert isinstance(results, abc.Mapping) or not len(results), results
    logger = logging.getLogger(__name__)
    for task, res in results.items():
        if isinstance(res, abc.Mapping):
            important_res = [(k, v) for k, v in res.items()]
            logger.info("copypaste: Task: {}".format(task))
            logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
        else:
            logger.info(f"copypaste: {task}={res}")


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary pickable data (not necessarily tensors).

    Args:
        data: any pickable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks.
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend='gloo')
    else:
        return dist.group.WORLD
    

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    This class will accumulate information of the inputs/outputs (by:meth:`process`),
    add produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that' used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our trainer.py, We expect the following format:

                * key: the name of the task (e.g., disp)
                * value: a dict of {metric name: score}, e.g., : {"EPE": 0.5}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if get_rank() == 0 and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results
    

class DispEvaluator(DatasetEvaluator):
    """Evaluate disparity accuracy using metrics."""

    def __init__(self, thres, only_valid, max_disp=None):
        """
        Args:
            thres (list[str] or None): threshold for outlier
            only_valid (bool): whether invalid pixels are excluded from evaluation
            max_disp (int or None): If None, maximum disparity will be regarded as infinity
        """
        # If true, will collect results from all ranks and return evaluation
        # in the main process. Otherwise, will evaluate the results in the current
        # process.
        self._distributed = get_world_size() > 1
        self._max_disp = np.inf if max_disp is None else max_disp
        self._thres = thres
        self._only_valid = only_valid

    def reset(self):
        self._epe = []
        self._thres_metric = OrderedDict()
        self._d1 = []
        self._val = []

        if self._thres is not None:
            for t in self._thres:
                self._thres_metric[t] = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to the model. It is a dict, which contain keys like "img1",
                "img2", "disp".
            outputs: the outputs of the model. It is a dict.
        """
        inputs = [dict(zip(inputs, t)) for t in zip(*inputs.values())]
        outputs = [dict(zip(outputs, t)) for t in zip(*outputs.values())]
        for input, output in zip(inputs, outputs):
            disp_pr = output["disp"]
            disp_gt = input["disp"].to(disp_pr.device)
            valid_gt = input["valid"].to(disp_pr.device)
            if self._only_valid:
                valid = valid_gt & (disp_gt < self._max_disp)
            else:
                valid = disp_gt < self._max_disp
            assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)

            epe = (disp_pr - disp_gt).abs()
            epe = epe.flatten()
            val = valid.flatten()

            if not val.any():
                continue
            if not torch.isfinite(epe[val]).all():
                raise ValueError("Non-finite disparity errors at valid ground-truth pixels.")

            self._epe.append(epe[val].mean().item())
            self._d1.append(((epe[val] > 3) & (epe[val] / disp_gt.flatten()[val] > 0.05)).float().sum().item())
            self._val.append(val.sum().item())

            if len(self._thres_metric) > 0:
                for t in self._thres:
                    tf = float(t)
                    out = (epe > tf)
                    self._thres_metric[t].append(out[val].float().mean().item())

    def evaluate(self):
        if self._distributed:
            synchronize()
            epe = list(itertools.chain(*gather(self._epe, dst=0)))
            d1 = list(itertools.chain(*gather(self._d1, dst=0)))
            val = list(itertools.chain(*gather(self._val, dst=0)))
            thres_metric = OrderedDict()
            for k, v in self._thres_metric.items():
                thres_metric[k] = list(itertools.chain(*gather(v, dst=0)))
            if get_rank() != 0:
                return {}
        else:
            epe = self._epe
            d1 = self._d1
            val = self._val
            thres_metric = self._thres_metric

        if not val or sum(val) == 0:
            raise ValueError("No valid ground-truth pixels were available for disparity evaluation.")

        epe = torch.tensor(epe).mean().item()    
        d1 = torch.tensor(d1).sum().item() / torch.tensor(val).sum().item() * 100
        res = {'epe': epe, 'd1': d1}
        for k, v in thres_metric.items():
            res[f'BP-{k}'] = torch.tensor(v).mean().item() * 100
        
        results = {'disp': res}
        return results