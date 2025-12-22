import os
import copy
import torch
import logging
from collections import defaultdict
from detectron2.engine import HookBase
from .timer import Timer, get_time_str
from detectron2.engine.defaults import create_ddp_model  # 如有需要
from detectron2.utils.comm import (
    is_main_process,
    get_world_size,
    all_gather,
    synchronize,
)
from .pseudo_tta import _accumulate_predictions_from_multiple_gpus, compute_on_dataset
from .evaluate import evaluate

class TTAHook(HookBase):
    """
    A hook that implements Test-Time Adaptation (TTA) via iterative self-supervised adaptation.
    This hook resets the model to the source checkpoint for each test image, then performs a number
    of adaptation iterations using pseudo-labels, and finally evaluates the adapted model at several
    TTA breakpoints.

    It integrates the logic of tta_inference into the hook so that after training, TTA evaluation is run.
    """
    def __init__(
        self,
        source_model,
        data_loader,
        dataset_name,
        voc_evaluation,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        tta_breakpoints=(),
        cfg=None,
    ):
        """
        Args:
            source_model: the original model checkpoint (used to reset the model).
            model: the current model (should be a GeneralizedRCNN) on which to run TTA.
            data_loader: a test data loader (batch size must be 1).
            dataset_name: the name of the dataset.
            voc_evaluation: flag for VOC-style evaluation.
            iou_types: types of IoU metrics to evaluate.
            box_only: if True, only evaluate boxes.
            device: device for computation.
            expected_results: expected results (optional).
            expected_results_sigma_tol: tolerance.
            output_folder: folder to save intermediate predictions.
            tta_breakpoints: a tuple of iterations at which to record predictions.
            cfg: configuration object.
        """
        self.source_model = source_model
        self.data_loader = data_loader
        self.dataset_name = dataset_name
        self.voc_evaluation = voc_evaluation
        self.iou_types = iou_types
        self.box_only = box_only
        self.device = torch.device(device)
        self.expected_results = expected_results
        self.expected_results_sigma_tol = expected_results_sigma_tol
        self.output_folder = output_folder
        self.tta_breakpoints = tta_breakpoints
        self.cfg = cfg

    def after_train(self):
        """
        在训练结束后执行 TTA 推理。该函数会：
         1. 重置模型参数为 source_model（原始 checkpoint）；
         2. 对每个测试图像（要求 batch_size=1）进行 TTA 自监督适应，
            记录不同 TTA 迭代下的预测结果；
         3. 汇总多个 GPU 的预测结果（若有分布式）；
         4. 保存预测结果（如果指定了 output_folder）；
         5. 调用 evaluate 对每个 TTA 结果进行评估，并打印输出。
        """
        logging.info("Start TTA inference on {}...".format(self.dataset_name))
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()

        # 运行 TTA 过程（函数 compute_on_dataset 内部实现了自监督适应迭代）
        predictions = compute_on_dataset(
            self.source_model,
            self.data_loader,
            self.device,
            self.tta_breakpoints,
            inference_timer,
            self.cfg,
        )
        # 等待所有 GPU 完成
        synchronize()
        total_time = total_timer.toc()
        num_devices = get_world_size()
        logging.info(
            "Total TTA run time: {} ({} s / img per device, on {} devices)".format(
                get_time_str(total_time),
                total_time * num_devices / len(self.data_loader.dataset),
                num_devices,
            )
        )
        # 收集各个 GPU 的预测结果
        for i, p in enumerate(predictions):
            predictions[i] = _accumulate_predictions_from_multiple_gpus(p)
        if not is_main_process():
            return

        # 保存预测结果到 output_folder（如果指定）
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            torch.save(predictions[0], os.path.join(self.output_folder, "prediction_0.pth"))
            for i in range(len(predictions) - 1):
                torch.save(
                    predictions[i + 1],
                    os.path.join(self.output_folder, "prediction_%d.pth" % self.tta_breakpoints[i]),
                )

        extra_args = dict(
            box_only=self.box_only,
            iou_types=self.iou_types,
            expected_results=self.expected_results,
            expected_results_sigma_tol=self.expected_results_sigma_tol,
        )

        evaluations = []
        for i, p in enumerate(predictions):
            eval_result = evaluate(
                dataset=self.data_loader.dataset,
                predictions=p,
                output_folder=self.output_folder,
                voc_evaluate=self.voc_evaluation,
                **extra_args,
            )
            evaluations.append(eval_result)
        logging.info("TTA evaluations: {}".format(evaluations))
