from ..builder import DETECTORS
from .single_stage_sipmask import SingleStageSipMaskDetector


@DETECTORS.register_module
class SipMask(SingleStageSipMaskDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SipMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
