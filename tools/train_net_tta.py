# #!/usr/bin/env python
# # Copyright (c) Facebook, Inc. and its affiliates.
# """
# A main training script.

# This scripts reads a given config file and runs the training or evaluation.
# It is an entry point that is made to train standard models in detectron2.

# In order to let one script support training of many models,
# this script contains logic that are specific to these built-in models and therefore
# may not be suitable for your own project.
# For example, your research project perhaps only needs a single "evaluator".

# Therefore, we recommend you to use detectron2 as an library and take
# this file as an example of how to use the library.
# You may want to write your own script with your datasets and other customizations.
# """
# import sys
# import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# import logging
# from collections import OrderedDict
# import torch
# import copy
# import detectron2.utils.comm as comm
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog
# from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
# from detectron2.evaluation import (
#     CityscapesInstanceEvaluator,
#     CityscapesSemSegEvaluator,
#     COCOEvaluator,
#     COCOPanopticEvaluator,
#     DatasetEvaluators,
#     LVISEvaluator,
#     PascalVOCDetectionEvaluator,
#     SemSegEvaluator,
#     verify_results,
# )
# from detectron2.modeling import GeneralizedRCNNWithTTA, build_model

# #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# import torch.multiprocessing
# import torch.nn.functional as F
# torch.multiprocessing.set_sharing_strategy('file_system')


# import lib.data.fewshot
# import lib.data.ovdshot
# from lib.categories import SEEN_CLS_DICT, ALL_CLS_DICT

# from collections import defaultdict

# import numpy as np

# from detectron2.evaluation.evaluator import DatasetEvaluator
# from detectron2.data.datasets.coco_zeroshot_categories import COCO_SEEN_CLS, \
#     COCO_UNSEEN_CLS, COCO_OVD_ALL_CLS
    
# from sklearn.metrics import precision_recall_curve
# from sklearn import metrics as sk_metrics
# from tta_utils.tta_hook import TTAHook
# from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts, DatasetCatalog, MetadataCatalog

# import sys
# sys.path.append("..")
# from tta_utils.pseudo_tta import _accumulate_predictions_from_multiple_gpus, compute_on_dataset
# from tta_utils.evaluate import evaluate
# from tta_utils.timer import Timer, get_time_str
# from detectron2.utils.comm import (
#     is_main_process,
#     get_world_size,
#     all_gather,
#     synchronize,
# )

# class Trainer(DefaultTrainer):
#     """
#     We use the "DefaultTrainer" which contains pre-defined default logic for
#     standard training workflow. They may not work for you, especially if you
#     are working on a new research project. In that case you can write your
#     own training loop. You can use "tools/plain_train_net.py" as an example.
#     """

#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         """
#         Create evaluator(s) for a given dataset.
#         This uses the special metadata "evaluator_type" associated with each builtin dataset.
#         For your own dataset, you can simply create an evaluator manually in your
#         script and do not have to worry about the hacky if-else logic here.
#         """
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         evaluator_list = []

#         if 'OpenSet' in cfg.MODEL.META_ARCHITECTURE:
#             if 'lvis' in dataset_name:
#                 evaluator_list.append(LVISEvaluator(dataset_name, output_dir=output_folder))
#             else:
#                 dtrain_name = cfg.DATASETS.TRAIN[0]
#                 # for coco14 FSOD benchmark
#                 if 'coco' in dataset_name:
#                     seen_cnames = SEEN_CLS_DICT['fs_coco14_base_train']
#                 else:
#                     seen_cnames = SEEN_CLS_DICT[dtrain_name]
#                 all_cnames = ALL_CLS_DICT[dtrain_name]
#                 unseen_cnames = [c for c in all_cnames if c not in seen_cnames]
#                 evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, few_shot_mode=True,
#                                                 seen_cnames=seen_cnames, unseen_cnames=unseen_cnames,
#                                                 all_cnames=all_cnames))
#             return DatasetEvaluators(evaluator_list)
    

#         evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
#         if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
#             evaluator_list.append(
#                 SemSegEvaluator(
#                     dataset_name,
#                     distributed=True,
#                     output_dir=output_folder,
#                 )
#             )
#         if evaluator_type in ["coco", "coco_panoptic_seg"]:
#             evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
#         if evaluator_type == "coco_panoptic_seg":
#             evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
#         if evaluator_type == "cityscapes_instance":
#             assert (
#                 torch.cuda.device_count() >= comm.get_rank()
#             ), "CityscapesEvaluator currently do not work with multiple machines."
#             return CityscapesInstanceEvaluator(dataset_name)
#         if evaluator_type == "cityscapes_sem_seg":
#             assert (
#                 torch.cuda.device_count() >= comm.get_rank()
#             ), "CityscapesEvaluator currently do not work with multiple machines."
#             return CityscapesSemSegEvaluator(dataset_name)
#         elif evaluator_type == "pascal_voc":
#             return PascalVOCDetectionEvaluator(dataset_name)
#         elif evaluator_type == "lvis":
#             return LVISEvaluator(dataset_name, output_dir=output_folder)
        
#         if len(evaluator_list) == 0:
#             raise NotImplementedError(
#                 "no Evaluator for the dataset {} with the type {}".format(
#                     dataset_name, evaluator_type
#                 )
#             )
#         elif len(evaluator_list) == 1:
#             return evaluator_list[0]
#         return DatasetEvaluators(evaluator_list)

#     @classmethod
#     def test_with_TTA(cls, cfg, model):
#         logger = logging.getLogger("detectron2.trainer")
#         # In the end of training, run an evaluation with TTA
#         # Only support some R-CNN models.
#         logger.info("Running inference with test-time augmentation ...")
#         model = GeneralizedRCNNWithTTA(cfg, model)
#         evaluators = [
#             cls.build_evaluator(
#                 cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
#             )
#             for name in cfg.DATASETS.TEST
#         ]
#         res = cls.test(cfg, model, evaluators)
#         res = OrderedDict({k + "_TTA": v for k, v in res.items()})
#         return res


# def setup(args):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.DE.CONTROLLER = args.controller

#     cfg.freeze()
#     default_setup(cfg, args)
#     print(cfg.DATASETS.TEST)
#     return cfg


# def main(args):
#     cfg = setup(args)

#     if args.eval_only:
#         model = Trainer.build_model(cfg)
#         DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#             cfg.MODEL.WEIGHTS, resume=args.resume
#         )
#         res = Trainer.test(cfg, model)
#         if cfg.TEST.AUG.ENABLED:
#             res.update(Trainer.test_with_TTA(cfg, model))
#         if comm.is_main_process():
#             verify_results(cfg, res)
#         return res

#     """
#     If you'd like to do anything fancier than the standard training logic,
#     consider writing your own training loop (see plain_train_net.py) or
#     subclassing the trainer.
#     """
#     trainer = Trainer(cfg)
#     trainer.resume_or_load(resume=args.resume)
#     if cfg.TEST.AUG.ENABLED:
#         offline_config = get_cfg()
#         offline_config.merge_from_file(cfg.DE.OFFLINE_RPN_CONFIG)
#         trainer.register_hooks(
#             [hooks.EvalHook(0, lambda: trainer.test_twih_TTA(cfg, build_model(offline_config)))]
#         )        
#         # trainer.register_hooks(
#         #     [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
#         # )
    
#     tta_breakpoints = (5, 10)

#     dataset_loader = build_detection_test_loader(cfg ,cfg.DATASETS.TEST[0])
#     tta_hook = TTAHook(source_model= trainer.model,
#                        data_loader=dataset_loader,
#                        dataset_name=cfg.DATASETS.TEST[0],
#                        voc_evaluation=False,
#                        iou_types=("bbox",),
#                        box_only=False,
#                        device=cfg.MODEL.DEVICE,
#                        expected_results=(),
#                        expected_results_sigma_tol=4,
#                        output_folder=os.path.join(cfg.OUTPUT_DIR, "tta_inference"),
#                        tta_breakpoints=tta_breakpoints,
#                        cfg=cfg,  
#                        )
#     trainer.register_hooks([tta_hook])
#     return trainer.train()


# if __name__ == "__main__":
#     args = default_argument_parser().parse_args()
#     print("Command Line Args:", args)
#     # launch(
#     #     main,
#     #     args.num_gpus,
#     #     num_machines=args.num_machines,
#     #     machine_rank=args.machine_rank,
#     #     dist_url=args.dist_url,
#     #     args=(args,),
#     # )
#     cfg = setup(args)
#     trainer = Trainer(cfg)

#     tta_breakpoints = (5, 10)
#     dataset_loader = build_detection_test_loader(cfg ,cfg.DATASETS.TEST[0])

#     offline_config = get_cfg()
#     offline_config.merge_from_file(cfg.DE.OFFLINE_RPN_CONFIG)
#     generalized_rcnn = build_model(offline_config)

#     for batch in dataset_loader:
#         x = generalized_rcnn(batch)
#         break

#     source_model= trainer.model
#     data_loader=dataset_loader
#     dataset_name=cfg.DATASETS.TEST[0]
#     voc_evaluation=False
#     iou_types=("bbox",)
#     box_only=False
#     device=cfg.MODEL.DEVICE
#     expected_results=()
#     expected_results_sigma_tol=4
#     output_folder=os.path.join(cfg.OUTPUT_DIR, "tta_inference")
    
    
#     logging.info("Start TTA inference on {}...".format(dataset_name))
#     total_timer = Timer()
#     inference_timer = Timer()
#     total_timer.tic()

#     # 运行 TTA 过程（函数 compute_on_dataset 内部实现了自监督适应迭代）
#     predictions = compute_on_dataset(
#         source_model,
#         data_loader,
#         device,
#         tta_breakpoints,
#         inference_timer,
#         cfg,
#     )
#     # 等待所有 GPU 完成
#     synchronize()
#     total_time = total_timer.toc()
#     num_devices = get_world_size()
#     logging.info(
#         "Total TTA run time: {} ({} s / img per device, on {} devices)".format(
#             get_time_str(total_time),
#             total_time * num_devices / len(data_loader.dataset),
#             num_devices,
#         )
#     )
#     # 收集各个 GPU 的预测结果
#     for i, p in enumerate(predictions):
#         predictions[i] = _accumulate_predictions_from_multiple_gpus(p)
#     # 保存预测结果到 output_folder（如果指定）
#     if output_folder:
#         os.makedirs(output_folder, exist_ok=True)
#         torch.save(predictions[0], os.path.join(output_folder, "prediction_0.pth"))
#         for i in range(len(predictions) - 1):
#             torch.save(
#                 predictions[i + 1],
#                 os.path.join(output_folder, "prediction_%d.pth" % tta_breakpoints[i]),
#             )
#         extra_args = dict(
#         box_only=box_only,
#         iou_types=iou_types,
#         expected_results=expected_results,
#         expected_results_sigma_tol=expected_results_sigma_tol,
#         )

#     evaluations = []
#     for i, p in enumerate(predictions):
#         eval_result = evaluate(
#             dataset=data_loader.dataset,
#             predictions=p,
#             output_folder=output_folder,
#             voc_evaluate=voc_evaluation,
#             **extra_args,
#         )
#         evaluations.append(eval_result)
#     logging.info("TTA evaluations: {}".format(evaluations))
#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import logging
from collections import OrderedDict
import torch
import copy
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA, build_model

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.multiprocessing
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')


import lib.data.fewshot
import lib.data.ovdshot
from lib.categories import SEEN_CLS_DICT, ALL_CLS_DICT

from collections import defaultdict

import numpy as np

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data.datasets.coco_zeroshot_categories import COCO_SEEN_CLS, \
    COCO_UNSEEN_CLS, COCO_OVD_ALL_CLS
    
from sklearn.metrics import precision_recall_curve
from sklearn import metrics as sk_metrics
from tta_utils.tta_hook import TTAHook
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts, DatasetCatalog, MetadataCatalog, build_detection_train_loader

import sys
sys.path.append("..")
from tta_utils.pseudo_tta import _accumulate_predictions_from_multiple_gpus, compute_on_dataset
from tta_utils.evaluate import evaluate
from tta_utils.timer import Timer, get_time_str
from detectron2.utils.comm import (
    is_main_process,
    get_world_size,
    all_gather,
    synchronize,
)

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []

        if 'OpenSet' in cfg.MODEL.META_ARCHITECTURE:
            if 'lvis' in dataset_name:
                evaluator_list.append(LVISEvaluator(dataset_name, output_dir=output_folder))
            else:
                dtrain_name = cfg.DATASETS.TRAIN[0]
                # for coco14 FSOD benchmark
                if 'coco' in dataset_name:
                    seen_cnames = SEEN_CLS_DICT['fs_coco14_base_train']
                else:
                    seen_cnames = SEEN_CLS_DICT[dtrain_name]
                all_cnames = ALL_CLS_DICT[dtrain_name]
                unseen_cnames = [c for c in all_cnames if c not in seen_cnames]
                evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, few_shot_mode=True,
                                                seen_cnames=seen_cnames, unseen_cnames=unseen_cnames,
                                                all_cnames=all_cnames))
            return DatasetEvaluators(evaluator_list)
    

        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DE.CONTROLLER = args.controller

    cfg.freeze()
    default_setup(cfg, args)
    print(cfg.DATASETS.TEST)
    return cfg


def main(args):
    cfg = setup(args)
    source_model = Trainer.build_model(cfg)
    checkpointer = DetectionCheckpointer(source_model)
    # source_model.only_train_mask = True
    source_model.turn_off_box_training(force=True)
    source_model.turn_off_cls_training(force=True)
    #checkpointer.load(f"/home/jack/code/NTIRE2025_CDFSOD-main/output/vitl/uodd_5shot_2/{cfg.DATASETS.TRAIN[0]}/model_final.pth")
    checkpointer.load(f"/home/jack/code/NTIRE2025_CDFSOD-main/output/vitl/dior_1shot_2/model_final.pth")
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    # trainer = Trainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    
    # tta_breakpoints = (5, 10)
    # trainer.train()
    # # 分布式环境中，确保所有进程都已经完成训练
    # if torch.distributed.is_initialized():
    #     torch.distributed.barrier()

    # source_model= trainer.model
    
    # del trainer
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()

    # if torch.cuda.is_available():
    #     with torch.cuda.device("cuda"):
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()
    dataset_loader = build_detection_test_loader(cfg ,cfg.DATASETS.TEST[0])
    
    # tta_hook = TTAHook(source_model= trainer.model,
    #                    data_loader=dataset_loader,
    #                    dataset_name=cfg.DATASETS.TEST[0],
    #                    voc_evaluation=False,
    #                    iou_types=("bbox",),
    #                    box_only=False,
    #                    device=cfg.MODEL.DEVICE,
    #                    expected_results=(),
    #                    expected_results_sigma_tol=4,
    #                    output_folder=os.path.join(cfg.OUTPUT_DIR, "tta_inference"),
    #                    tta_breakpoints=tta_breakpoints,
    #                    cfg=cfg,  
    #                    )
    # trainer.register_hooks([tta_hook])


    tta_breakpoints = (5, 10)
    data_loader=dataset_loader
    dataset_name=cfg.DATASETS.TEST[0]
    device=cfg.MODEL.DEVICE
    output_folder=os.path.join(cfg.OUTPUT_DIR, "tta_inference")
    
    
    logging.info("Start TTA inference on {}...".format(dataset_name))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()



    # 运行 TTA 过程（函数 compute_on_dataset 内部实现了自监督适应迭代）
    predictions = compute_on_dataset(
        source_model,
        data_loader,
        device,
        tta_breakpoints,
        inference_timer,
        cfg,
    )

    # 等待所有 GPU 完成
    dtrain_name = cfg.DATASETS.TRAIN[0]
    seen_cnames = SEEN_CLS_DICT[dtrain_name]
    all_cnames = ALL_CLS_DICT[dtrain_name]
    unseen_cnames = [c for c in all_cnames if c not in seen_cnames]
    evaluater = COCOEvaluator(dataset_name, output_dir=output_folder, few_shot_mode=True,
                                                seen_cnames=seen_cnames, unseen_cnames=unseen_cnames,
                                                all_cnames=all_cnames)
    evaluater.reset()
    for predict_item in predictions:
        evaluater._predictions.append(predict_item)
    evaluater.evaluate()
    return 


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    # torch.multiprocessing.set_start_method("spawn", force=True)
    main(args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
    
    
    
    # cfg = setup(args)
    # trainer = Trainer(cfg)

    # tta_breakpoints = (2, 3)
    # dataset_loader = build_detection_test_loader(cfg ,cfg.DATASETS.TEST[0])

    # # train_dataset_loader = trainer.build_train_loader(cfg)

    # # for batch in train_dataset_loader:
    # #     x = batch
    # #     break

    # offline_config = get_cfg()
    # offline_config.merge_from_file(cfg.DE.OFFLINE_RPN_CONFIG)
    # generalized_rcnn = build_model(offline_config)

    # source_model= trainer.model
    # data_loader=dataset_loader
    # dataset_name=cfg.DATASETS.TEST[0]
    # voc_evaluation=False
    # iou_types=("bbox",)
    # box_only=False
    # device=cfg.MODEL.DEVICE
    # expected_results=()
    # expected_results_sigma_tol=4
    # output_folder=os.path.join(cfg.OUTPUT_DIR, "tta_inference")
    
    
    # logging.info("Start TTA inference on {}...".format(dataset_name))
    # total_timer = Timer()
    # inference_timer = Timer()
    # total_timer.tic()

    # # 运行 TTA 过程（函数 compute_on_dataset 内部实现了自监督适应迭代）
    # predictions = compute_on_dataset(
    #     source_model,
    #     data_loader,
    #     device,
    #     tta_breakpoints,
    #     inference_timer,
    #     cfg,
    # )
    # # 等待所有 GPU 完成


    # evaluator_list = []
    # dtrain_name = cfg.DATASETS.TRAIN[0]
    # seen_cnames = SEEN_CLS_DICT[dtrain_name]
    # all_cnames = ALL_CLS_DICT[dtrain_name]
    # unseen_cnames = [c for c in all_cnames if c not in seen_cnames]
    # evaluater = COCOEvaluator(dataset_name, output_dir=output_folder, few_shot_mode=True,
    #                                             seen_cnames=seen_cnames, unseen_cnames=unseen_cnames,
    #                                             all_cnames=all_cnames)
    # evaluater.reset()
    # for predict_item in predictions:
    #     evaluater._predictions.append(predict_item)
    # evaluater.evaluate()
    # # synchronize()
    # # total_time = total_timer.toc()
    # # num_devices = get_world_size()
    # # logging.info(
    # #     "Total TTA run time: {} ({} s / img per device, on {} devices)".format(
    # #         get_time_str(total_time),
    # #         total_time * num_devices / len(data_loader.dataset),
    # #         num_devices,
    # #     )
    # # )
    # # # 收集各个 GPU 的预测结果
    # # # for i, p in enumerate(predictions):
    # # #     predictions[i] = _accumulate_predictions_from_multiple_gpus(p)
    # # # 保存预测结果到 output_folder（如果指定）
    # # if output_folder:
    # #     os.makedirs(output_folder, exist_ok=True)
    # #     torch.save(predictions[0], os.path.join(output_folder, "prediction_0.pth"))
    # #     for i in range(len(predictions) - 1):
    # #         torch.save(
    # #             predictions[i + 1],
    # #             os.path.join(output_folder, "prediction_%d.pth" % tta_breakpoints[i]),
    # #         )
    # #     extra_args = dict(
    # #     box_only=box_only,
    # #     iou_types=iou_types,
    # #     expected_results=expected_results,
    # #     expected_results_sigma_tol=expected_results_sigma_tol,
    # #     )

    # # evaluations = []
    # # for i, p in enumerate(predictions):
    # #     eval_result = evaluate(
    # #         dataset=data_loader.dataset,
    # #         predictions=p,
    # #         output_folder=output_folder,
    # #         voc_evaluate=voc_evaluation,
    # #         **extra_args,
    # #     )
    # #     evaluations.append(eval_result)
    # # logging.info("TTA evaluations: {}".format(evaluations))