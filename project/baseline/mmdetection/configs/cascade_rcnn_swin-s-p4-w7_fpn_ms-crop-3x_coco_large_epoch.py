_base_ = './cascade-rcnn-swin.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)