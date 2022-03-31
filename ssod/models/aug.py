import numpy as np
from mmdet.core import multiclass_nms
import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid
import copy
import torch.nn.functional as F

@DETECTORS.register_module()
class SoftTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SoftTeacher, self).__init__(
            dict(teacher=build_detector(model),teacher2=build_detector(model), student=build_detector(model),student2=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.sign=0
        if train_cfg is not None:
            self.freeze("teacher")
            self.freeze("teacher2")
            self.unsup_weight = self.train_cfg.unsup_weight

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            # dochange
            sup_loss2 = self.student2.forward_train(**data_groups["sup"])
            sup_loss2 = {"sup2_" + k:v for k, v in sup_loss2.items()}
            loss.update(**sup_loss)
            loss.update(**sup_loss2)
        if "teacher_weak" in data_groups:
            # teacher supervise sutdent2
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["teacher_weak"],data_groups["teacher2_weak"],data_groups["unsup_student2_strong"],flag=1
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

            
            # teacher2 supervise sutdent
            unsup_loss2 = weighted_loss(
                self.foward_unsup_train(
                    data_groups["teacher_weak"],data_groups["teacher2_weak"],data_groups["unsup_student_strong"],flag=2
                ),
                weight=self.unsup_weight,
            )
            unsup_loss2 = {"unsup2_" + k: v for k, v in unsup_loss2.items()}
            
            loss.update(**unsup_loss2)
            
            
        return loss

    def foward_unsup_train(self, student_weak,student_weak2 ,student_strong,flag):
        # sort the teacher and student input to avoid some bugs
#         tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
#         snames = [meta["filename"] for meta in student_data["img_metas"]]
#         tidx = [tnames.index(name) for name in snames]
#         with torch.no_grad():
#             teacher_info = self.extract_teacher_info(
#                 teacher_data["img"][
#                     torch.Tensor(tidx).to(teacher_data["img"].device).long()
#                 ],
#                 [teacher_data["img_metas"][idx] for idx in tidx],
#                 [teacher_data["proposals"][idx] for idx in tidx]
#                 if ("proposals" in teacher_data)
#                 and (teacher_data["proposals"] is not None)
#                 else None,
#             )
#         student_info = self.extract_student_info(teacher_info,**student_data)

        s_weak_names = [meta["filename"] for meta in student_weak["img_metas"]]
        s_strong_names = [meta["filename"] for meta in student_strong["img_metas"]]
        s_weak_names2 = [meta["filename"] for meta in student_weak2["img_metas"]]
        tidx=[s_weak_names.index(name) for name in s_strong_names] ## weak supervise strong
        tidx2=[s_weak_names2.index(name) for name in s_strong_names] ## weak supervise strong
        

        
        if flag==1:
            with torch.no_grad():
                teacher_info=self.extract_teacher_info(
                    student_weak['img'][
                        torch.Tensor(tidx).to(student_weak["img"].device).long()
                    ],
                    [student_weak['img_metas'][idx] for idx in tidx],
                    [student_weak['proposals'][idx] for idx in tidx]
                    if ("proposals" in student_weak)
                    and (student_weak["proposals"] is not None)
                    else None,
                )
            student_info=self.extract_student2_info(teacher_info,**student_strong)
#             return self.compute_pseudo_label_loss(student_info, teacher_info,flag=1)
        elif flag==2:
            with torch.no_grad():
                teacher_info=self.extract_teacher2_info(
                    student_weak2['img'][
                        torch.Tensor(tidx2).to(student_weak2["img"].device).long()
                    ],
                    [student_weak2['img_metas'][idx] for idx in tidx2],
                    [student_weak2['proposals'][idx] for idx in tidx2]
                    if ("proposals" in student_weak2)
                    and (student_weak2["proposals"] is not None)
                    else None,
                )
            student_info=self.extract_student_info(teacher_info,**student_strong)

    
        return self.compute_pseudo_label_loss(student_info, teacher_info,flag)
        
        

    def compute_pseudo_label_loss(self, student_info, teacher_info,flag):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}
        if flag==1:
            rpn_loss, proposal_list = self.rpn_loss(
                student_info["rpn_out"],
                pseudo_bboxes,
                student_info["img_metas"],
                student_info=student_info,
            )
            loss.update(rpn_loss)
        else:
            # flag==2 flag==3
            rpn_loss, proposal_list = self.rpn_loss2(
                student_info["rpn_out"],
                pseudo_bboxes,
                student_info["img_metas"],
                student_info=student_info,
            )
            loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]
        if flag==1:
            loss.update(
                self.unsup_rcnn_cls_loss(
                    student_info["backbone_feature"],
                    student_info["img_metas"],
                    proposals,
                    pseudo_bboxes,
                    pseudo_labels,
                    teacher_info["transform_matrix"],
                    student_info["transform_matrix"],
                    teacher_info["img_metas"],
                    teacher_info["backbone_feature"],
                    student_info=student_info,
                )
            )
            loss.update(
                self.unsup_rcnn_reg_loss(
                    student_info["backbone_feature"],
                    student_info["img_metas"],
                    proposals,
                    pseudo_bboxes,
                    pseudo_labels,
                    student_info=student_info,
                )
            )
        elif flag==2:
            # flag==2 flag==3
            loss.update(
                self.unsup_rcnn_cls_loss2(
                    student_info["backbone_feature"],
                    student_info["img_metas"],
                    proposals,
                    pseudo_bboxes,
                    pseudo_labels,
                    teacher_info["transform_matrix"],
                    student_info["transform_matrix"],
                    teacher_info["img_metas"],
                    teacher_info["backbone_feature"],
                    student_info=student_info,
                )
            )
            loss.update(
                self.unsup_rcnn_reg_loss2(
                    student_info["backbone_feature"],
                    student_info["img_metas"],
                    proposals,
                    pseudo_bboxes,
                    pseudo_labels,
                    student_info=student_info,
                )
            )

        return loss
    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student2.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student2.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student2.train_cfg.get(
                "rpn_proposal", self.student2.test_cfg.rpn
            )
            proposal_list = self.student2.rpn_head.get_bboxes(
                *rpn_out, img_metas, cfg=proposal_cfg
            )
            log_image_with_boxes(
                "rpn",
                student_info["img"][0],
                pseudo_bboxes[0][:, :4],
                bbox_tag="rpn_pseudo_label",
                scores=pseudo_bboxes[0][:, 4],
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
            return losses, proposal_list
        else:
            return {}, None
    
    def rpn_loss2(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas, cfg=proposal_cfg
            )
            log_image_with_boxes(
                "rpn",
                student_info["img"][0],
                pseudo_bboxes[0][:, :4],
                bbox_tag="rpn_pseudo_label",
                scores=pseudo_bboxes[0][:, 4],
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
            return losses, proposal_list
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        gt_bboxes2, gt_labels2, _ = multi_apply(
            filter_invalid,
            [bbox for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )

        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student2.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student2.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student2.train_cfg.rcnn
        )
        #print(gt_bboxes2[0][0].shape)   
        # change
        ignore_num=0
        student_soft=bbox_results['cls_score'].new_zeros(0,81)
        teacher_soft=bbox_results['cls_score'].new_zeros(0,81)
        all_num=0
        for ind,i in enumerate(sampling_results):
           # print(i.pos_assigned_gt_inds)
           # print(len(gt_bboxes2[ind]))
            for jind,j in enumerate(i.pos_assigned_gt_inds):
                if gt_bboxes2[ind][j][4]<0.9:
                    bbox_targets[1][all_num+jind]=0
                    ignore_num+=1
                    student_soft=torch.cat([student_soft,bbox_results['cls_score'][all_num+jind].unsqueeze(0)],dim=0)    # get the ind th photo's jind th sample
                    teacher_soft=torch.cat([teacher_soft,gt_bboxes2[ind][j][5:81+5].unsqueeze(0)],dim=0)
            # print(bbox_targets[1][ind*512:ind*512+jind])
            all_num+=i.pos_bboxes.size(0)                                                                                        
            all_num+=i.neg_bboxes.size(0) 
      #  print("student")        
      #  print(student_soft)
      #  print("teacher")
      #  print(teacher_soft)

     #   if self.sign==0:
      #      torch.save(student_soft,"student.pth")
      #      torch.save(teacher_soft,"teacher.pth")
            

        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student2.roi_head.bbox_head.num_classes
            #bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student2.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )


        loss_kl = F.kl_div(torch.log_softmax(student_soft,dim=1),teacher_soft,reduction="none")
   #     if self.sign==0:
  #          torch.save(loss_kl,"loss.pth")
       # if self.sign%200==0:
           # print(loss_kl)
           # print(ignore_num)
      #  print(ignore_num)
        loss_soft = loss_kl.mean(1).sum()/max(ignore_num, 1.0)
        self.sign+=1

        loss['loss_soft'] = loss_soft




        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_cls",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return loss
    def unsup_rcnn_cls_loss2(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
#         print(pseudo_labels)
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        gt_bboxes2, gt_labels2, _ = multi_apply(
            filter_invalid,
            [bbox for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )

        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        #print(gt_bboxes2[0][0].shape)   
        # change
        ignore_num=0
        student_soft=bbox_results['cls_score'].new_zeros(0,81)
        teacher_soft=bbox_results['cls_score'].new_zeros(0,81)
        all_num=0
        for ind,i in enumerate(sampling_results):
           # print(i.pos_assigned_gt_inds)
           # print(len(gt_bboxes2[ind]))
            for jind,j in enumerate(i.pos_assigned_gt_inds):
                if gt_bboxes2[ind][j][4]<0.9:
                    bbox_targets[1][all_num+jind]=0
                    ignore_num+=1
                    student_soft=torch.cat([student_soft,bbox_results['cls_score'][all_num+jind].unsqueeze(0)],dim=0)    # get the ind th photo's jind th sample
                    teacher_soft=torch.cat([teacher_soft,gt_bboxes2[ind][j][5:81+5].unsqueeze(0)],dim=0)
            # print(bbox_targets[1][ind*512:ind*512+jind])
            all_num+=i.pos_bboxes.size(0)                                                                                        
            all_num+=i.neg_bboxes.size(0) 
      #  print("student")        
      #  print(student_soft)
      #  print("teacher")
      #  print(teacher_soft)

      #  if self.sign==0:
      #      torch.save(student_soft,"student.pth")
      #      torch.save(teacher_soft,"teacher.pth")
            

        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher2.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
            #bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )


        loss_kl = F.kl_div(torch.log_softmax(student_soft,dim=1),teacher_soft,reduction="none")
  #      if self.sign==0:
  #          torch.save(loss_kl,"loss.pth")
       # if self.sign%200==0:
           # print(loss_kl)
           # print(ignore_num)
      #  print(ignore_num)
        loss_soft = loss_kl.mean(1).sum()/max(ignore_num, 1.0)
        self.sign+=1

        loss['loss_soft'] = loss_soft




        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_cls",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return loss
    def unsup_rcnn_cls_loss3(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
#         print(pseudo_labels)
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result2(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
#         print(gt_bboxes)
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
           # bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_cls",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return loss

    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student2.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return {"loss_bbox": loss_bbox}
    
    def unsup_rcnn_reg_loss2(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return {"loss_bbox": loss_bbox}

    def get_sampling_result2(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results
    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student2.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student2.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results
    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    
    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list
        proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, None, rescale=False
        )

        bbox=[]                                                                                                                  
        label=[]                                                                                                                 
        inds=[]                                                                                                                  
        for img_id in range(len(proposal_list)):                                                                                 
            _bbox,_label,_inds=multiclass_nms(proposal_list[img_id],                                                             
                       proposal_label_list[img_id],self.teacher.test_cfg.rcnn.score_thr,                                          
            self.teacher.test_cfg.rcnn.nms,self.teacher.test_cfg.rcnn.max_per_img,return_inds=True)                              
            bbox.append(_bbox)                                                                                                   
            label.append(_label)                                                                                                 
            inds.append(_inds)                                                                                                   
                                                                                                                                  
        soft=[]                                                                                                                  
        for img_id in range(len(proposal_list)):                                                                                 
            temp=bbox[0].new_zeros(0,81)                                                                                                              
            for ind in inds[img_id]: 
            #    print(proposal_label_list[img_id][ind//80])                                                                                            
                temp=torch.cat([temp,proposal_label_list[img_id][ind//80].unsqueeze(0)],dim=0) 
            temp=temp.to(bbox[0].device)                                                               
            soft.append(temp)

      #  print(bbox)
      #  print(soft)

        proposal_list = [p.to(feat[0].device) for p in bbox]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        # change add soft_label to 5:5+81
        proposal_list=[
            torch.cat([_bbox,_soft],dim=-1) for _bbox,_soft in zip(proposal_list,soft)
        ]

        proposal_label_list = [p.to(feat[0].device) for p in label]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, 4],  # change 
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info
    
    
    def extract_teacher2_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher2.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher2.train_cfg.get(
                "rpn_proposal", self.teacher2.test_cfg.rpn
            )
            rpn_out = list(self.teacher2.rpn_head(feat))
            proposal_list = self.teacher2.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list = self.teacher2.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, None, rescale=False
        )

        bbox=[]                                                                                                                  
        label=[]                                                                                                                 
        inds=[]                                                                                                                  
        for img_id in range(len(proposal_list)):                                                                                 
            _bbox,_label,_inds=multiclass_nms(proposal_list[img_id],                                                             
                       proposal_label_list[img_id],self.teacher2.test_cfg.rcnn.score_thr,                                          
            self.teacher2.test_cfg.rcnn.nms,self.teacher2.test_cfg.rcnn.max_per_img,return_inds=True)                              
            bbox.append(_bbox)                                                                                                   
            label.append(_label)                                                                                                 
            inds.append(_inds)                                                                                                   
                                                                                                                                  
        soft=[]                                                                                                                  
        for img_id in range(len(proposal_list)):                                                                                 
            temp=bbox[0].new_zeros(0,81)                                                                                                              
            for ind in inds[img_id]: 
            #    print(proposal_label_list[img_id][ind//80])                                                                                            
                temp=torch.cat([temp,proposal_label_list[img_id][ind//80].unsqueeze(0)],dim=0) 
            temp=temp.to(bbox[0].device)                                                               
            soft.append(temp)

      #  print(bbox)
      #  print(soft)

        proposal_list = [p.to(feat[0].device) for p in bbox]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        # change add soft_label to 5:5+81
        proposal_list=[
            torch.cat([_bbox,_soft],dim=-1) for _bbox,_soft in zip(proposal_list,soft)
        ]

        proposal_label_list = [p.to(feat[0].device) for p in label]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, 4],  # change 
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        reg_unc = self.compute_uncertainty_with_aug2(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    
    def extract_student_info(self,teacher, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        pseudo_bboxes = copy.deepcopy(teacher['det_bboxes'])
        img2 = copy.deepcopy(img) 
        
        
        # change get up 0.9 bboxes for randerase
        gt_bboxes, _ , _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
         #   pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=0.9,
        )
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(gt_bboxes[0].device)
            for meta in img_metas
        ]
        M = self._get_trans_mat(
            teacher["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            gt_bboxes,
            M,
            [meta["img_shape"] for meta in img_metas],
        )
        
        lower=0.02
        prob=0.5
        higher=0.3
        ratio=4
        
        for ind,i in enumerate(pseudo_bboxes):
            for jind,j in enumerate(i):
                if prob<=np.random.rand():
                    continue
                
                x1,y1,x2,y2 = pseudo_bboxes[ind][jind,0:4]
                w_bbox=x2-x1
                h_bbox=y2-y1
                area = w_bbox*h_bbox
                
                target_area=np.random.uniform(lower,higher)*area
                aspect = np.random.uniform(ratio,1/ratio)
                
                h = int((target_area*aspect)**0.5)
                w = int((target_area/aspect)**0.5)
                
                if w<w_bbox and h<h_bbox:
                    off_y1 = np.random.randint(0,max(int(h_bbox-h),1))
                    off_x1 = np.random.randint(0,max(int(w_bbox-w),1))
                    
                    img[ind,:,int(y1+off_y1):int(y1+off_y1+h),int(x1+off_x1):int(x1+off_x1+w)] = 0

        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info
    
    def extract_student2_info(self,teacher, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        pseudo_bboxes = copy.deepcopy(teacher['det_bboxes'])
        img2 = copy.deepcopy(img) 
        
        
        # change get up 0.9 bboxes for randerase
        gt_bboxes, _ , _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
         #   pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=0.9,
        )
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(gt_bboxes[0].device)
            for meta in img_metas
        ]
        M = self._get_trans_mat(
            teacher["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            gt_bboxes,
            M,
            [meta["img_shape"] for meta in img_metas],
        )
        
        lower=0.02
        prob=0.5
        higher=0.3
        ratio=4
        
        for ind,i in enumerate(pseudo_bboxes):
            for jind,j in enumerate(i):
                if prob<=np.random.rand():
                    continue
                
                x1,y1,x2,y2 = pseudo_bboxes[ind][jind,0:4]
                w_bbox=x2-x1
                h_bbox=y2-y1
                area = w_bbox*h_bbox
                
                target_area=np.random.uniform(lower,higher)*area
                aspect = np.random.uniform(ratio,1/ratio)
                
                h = int((target_area*aspect)**0.5)
                w = int((target_area/aspect)**0.5)
                
                if w<w_bbox and h<h_bbox:
                    off_y1 = np.random.randint(0,max(int(h_bbox-h),1))
                    off_x1 = np.random.randint(0,max(int(w_bbox-w),1))
                    
                    img[ind,:,int(y1+off_y1):int(y1+off_y1+h),int(x1+off_x1):int(x1+off_x1+w)] = 0



        student_info["img"] = img
        feat = self.student2.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student2.with_rpn:
            rpn_out = self.student2.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info


    def compute_uncertainty_with_aug(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc
    def compute_uncertainty_with_aug2(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher2.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc
    
    
    def compute_uncertainty_with_aug3(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"teacher2." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            state_dict.update({"student2." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

