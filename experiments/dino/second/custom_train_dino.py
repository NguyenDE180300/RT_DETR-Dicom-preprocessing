_base_ = './mmdetection/configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py'

import os

# ----------------------
# METADATA (21 classes)
# ----------------------
metainfo = dict(
    classes=(
        'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
        'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema',
        'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity',
        'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass',
        'Other lesion', 'Pleural effusion', 'Pleural thickening',
        'Pneumothorax', 'Pulmonary fibrosis', 'Rib fracture'
    )
)

# ----------------------
# DATA ROOT (IMPORTANT)
# ----------------------
data_root = '/home/Ubuntu/DATA/images'
ann_root = '/home/Ubuntu/DATA/annotations'

# ----------------------
# MODEL
# ----------------------
model = dict(
    num_queries=600,
    bbox_head=dict(num_classes=22),
    backbone=dict(
        frozen_stages=-1,
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./mmdetection/checkpoints/swin_large_patch4_window12_384_22k.pth'
        )
    )
)

# ----------------------
# TRAIN DATA
# ----------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file=os.path.join(ann_root, 'instances_train_mmdet.json'),
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

# ----------------------
# VAL DATA
# ----------------------
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file=os.path.join(ann_root, 'instances_val_mmdet.json'),
        data_prefix=dict(img='train/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='PackDetInputs')
        ]
    )
)

# ----------------------
# EVALUATOR
# ----------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(ann_root, 'instances_val_mmdet.json'),
    metric='bbox'
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# ----------------------
# TRAIN LOOP (24e schedule)
# ----------------------
max_epochs = 24

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)

# ----------------------
# OPTIMIZER (AMP FIXED)
# ----------------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
    accumulative_counts=4
)

# ----------------------
# LR SCHEDULER
# ----------------------
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='ReduceOnPlateauParamScheduler',
        param_name='lr',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=3,
        factor=0.1,
        by_epoch=True)
]

# ----------------------
# HOOKS
# ----------------------
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50)
)

# ----------------------
# WORK DIR
# ----------------------
work_dir = './work_dirs/dino_swin_l_vinbigdata'
