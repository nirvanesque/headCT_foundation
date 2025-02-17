import os
import yaml
from yacs.config import CfgNode as CN

# Initialize the configuration node
_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data Settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 64  # Batch size for a single GPU
_C.DATA.BASE_PATH = '<path-to>/datasets'  # Base path for datasets
_C.DATA.TRAIN_CSV_PATH = '<path-to>/datasets/train.csv'  # Path to training CSV
_C.DATA.VAL_CSV_PATH = '<path-to>/datasets/val.csv'  # Path to validation CSV
_C.DATA.TEST_CSV_PATH = '<path-to>/datasets/test.csv'  # Path to test CSV
_C.DATA.PIN_MEMORY = True  # Pin memory for DataLoader
_C.DATA.NUM_WORKERS = 4  # Number of workers for DataLoader
_C.DATA.CACHE_NUM = -1  # Number of cache items
_C.DATA.CACHE_RATE = 1.0  # Cache rate
_C.DATA.CACHE_DIR = '<path-to>/cache_dir'  # Cache directory
_C.DATA.DATASET = 'nyu'  # Dataset name
_C.DATA.FEW_SHOTS = -1  # Number of few shots
_C.DATA.NUM_CLASSES = 2  # Number of classes

# -----------------------------------------------------------------------------
# General Model Settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'mae'  # Model type
_C.MODEL.PRETRAINED = None  # Pretrained model path
_C.MODEL.DIR = '<path-to>/model_saved'  # Model save directory
_C.MODEL.SAVE_NAME = 'debug.pt'  # Model save name
_C.MODEL.ROI = [96, 96, 96]  # Region of Interest
_C.MODEL.IN_CHANS = 3  # Input channels

# -----------------------------------------------------------------------------
# MAE Settings
# -----------------------------------------------------------------------------
_C.MAE = CN()
_C.MAE.INPUT_SIZE = 96  # Input size
_C.MAE.PATCH_SIZE = 16  # Patch size
_C.MAE.MASK_RATIO = 0.75  # Mask ratio
_C.MAE.IN_CHANS = 3  # Input channels
_C.MAE.DROPOUT_RATE = 0.0  # Dropout rate
_C.MAE.PATCH_EMBED = 'conv'  # Patch embedding layer type
_C.MAE.POS_EMBED = 'sincos'  # Position embedding layer type
_C.MAE.NORM_LAYER = 'layernorm'  # Normalization layer
_C.MAE.SPATIAL_DIMS = 3  # Spatial dimension of input
_C.MAE.NORM_PIX_LOSS = False  # Patch pixels normalization
_C.MAE.RETURN_IMAGE = False  # Return image

_C.MAE.ENCODER_EMBED_DIM = 768  # Encoder embedding dimension
_C.MAE.ENCODER_DEPTH = 12  # Encoder depth
_C.MAE.ENCODER_MLP_DIM = 3072  # Encoder MLP layer dimension
_C.MAE.ENCODER_NUM_HEADS = 12  # Encoder number of multi-heads

_C.MAE.DECODER_EMBED_DIM = 768  # Decoder embedding dimension
_C.MAE.DECODER_DEPTH = 8  # Decoder depth
_C.MAE.DECODER_MLP_DIM = 2048  # Decoder MLP layer dimension
_C.MAE.DECODER_NUM_HEADS = 16  # Decoder number of multi-heads

_C.MAE.USE_BIAS = False  # Use bias

# -----------------------------------------------------------------------------
# DINO Settings
# -----------------------------------------------------------------------------
_C.DINO = CN()
_C.DINO.GLOBAL_CROP_SIZE = [112, 112, 112]  # Global crop size
_C.DINO.GLOBAL_CROP_NUM = 2  # Number of global crops
_C.DINO.LOCAL_CROP_SIZE = [64, 64, 64]  # Local crop size
_C.DINO.LOCAL_CROP_NUM = 2  # Number of local crops
_C.DINO.HEAD_N_LAYERS = 3  # Number of layers in DINO head
_C.DINO.HEAD_N_PROTOTYPES = 65536  # Number of prototypes in DINO head
_C.DINO.BOTTLENECK_DIM = 256  # Bottleneck dimension
_C.DINO.HEAD_HIDDEN_DIM = 2048  # Hidden dimension in DINO head
_C.DINO.MOMENTUM_TEACHER = 0.994  # Momentum Start Value for teacher model
_C.DINO.MOMENTUM_TEACHER_END = 1.0  # Momentum End Value for teacher model
_C.DINO.WARMUP_TEACHER_TEMP = 0.04  # Warmup teacher temperature
_C.DINO.TEACHER_TEMP = 0.07  # Teacher temperature
_C.DINO.WARMUP_TEACHER_EPOCHS = 30  # Warmup teacher epochs
_C.DINO.DINO_LOSS_WEIGHT = 1.0  # DINO loss weight
_C.DINO.USE_BN = True # Use BatchNorm in DINO head
_C.DINO.NORM_LAST_LAYER = True # Normalize last layer
_C.DINO.FREEZE_LAST_LAYER = 1 # Freeze last layer

# -----------------------------------------------------------------------------
# VIT Settings
# -----------------------------------------------------------------------------
_C.VIT = CN()
_C.VIT.INPUT_SIZE = 96  # Input size
_C.VIT.PATCH_SIZE = 12  # Patch size
_C.VIT.IN_CHANS = 3  # Input channels
_C.VIT.DROPOUT_RATE = 0.0  # Dropout rate
_C.VIT.PATCH_EMBED = 'conv'  # Patch embedding layer type
_C.VIT.POS_EMBED = 'sincos'  # Position embedding layer type
_C.VIT.NORM_LAYER = 'layernorm'  # Normalization layer
_C.VIT.SPATIAL_DIMS = 3  # Spatial dimension of input

_C.VIT.NUM_LAYERS = 12  # Number of layers in ViT
_C.VIT.NUM_HEADS = 12  # Number of multi-heads in ViT
_C.VIT.HIDDEN_SIZE = 768  # Hidden dimension in ViT
_C.VIT.MLP_DIM = 3072  # MLP layer dimension in ViT
_C.VIT.NUM_REGISTER_TOKENS = 0  # Number of register tokens
_C.VIT.PATCHES_OVERLAP = 0.2  # Patches splitting overlap
_C.VIT.POOLING = 'cls'  # Pooling type

_C.VIT.CLASSIFICATION = False  # Classification flag

_C.VIT.USE_BIAS = False  # Use bias

# -----------------------------------------------------------------------------
# Training Settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.MAX_EPOCHS = 100  # Maximum number of epochs
_C.TRAIN.VAL_EVERY = 10  # Validate every N epochs
_C.TRAIN.BASE_LR = 1.5e-3  # Base learning rate
_C.TRAIN.MIN_LR = 1.5e-7  # Minimum learning rate
_C.TRAIN.WEIGHT_DECAY = 0.04  # Weight decay
_C.TRAIN.WEIGHT_DECAY_END = 0.4  # End weight decay
_C.TRAIN.BETA1 = 0.9  # AdamW beta1
_C.TRAIN.BETA2 = 0.95  # AdamW beta2
_C.TRAIN.MOMENTUM = 0.9  # Momentum
_C.TRAIN.LOSS = 'l1'  # Loss type
_C.TRAIN.TEMPERATURE = 0.5  # Contrastive loss temperature
_C.TRAIN.OPTIMIZER = 'AdamW'  # Optimizer type
_C.TRAIN.SCHEDULER = 'cosine'  # Scheduler type
_C.TRAIN.PER_WARMUP = 0.05  # Percentage of linear warmup
_C.TRAIN.GRAD_CLIP = 1.0  # Gradient clipping
_C.TRAIN.LOCK = False  # Lock backbone
_C.TRAIN.LORA = False  # Train with LoRA
_C.TRAIN.CLASSIFIER = 'linear'  # Downstream classifier layer
_C.TRAIN.LABEL_NAME = 'cancer'  # Downstream label name

# -----------------------------------------------------------------------------
# Logging Settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.OUTPUT_DIR = '<path-to>/headCT_foundation/log'  # Logging directory
_C.LOG.FILENAME = 'headCT_foundation'  # Logging file save name

# -----------------------------------------------------------------------------
# wandb Settings
# -----------------------------------------------------------------------------
_C.WANDB = CN()
_C.WANDB.WANDB_ENABLE = False  # Enable wandb
_C.WANDB.PROJECT = 'headCT_foundation'  # wandb project name

# -----------------------------------------------------------------------------
# Misc Settings
# -----------------------------------------------------------------------------
_C.SEED = 42  # Seed for reproducibility
_C.AMP_ENABLE = False  # Enable Pytorch automatic mixed precision
_C.LOCAL_RANK = 0  # Local rank for distributed training
_C.OUTPUT = ''  # Path to output folder
_C.TAG = 'default'  # Tag of experiment
_C.PREDS_SAVE_NAME = 'None'  # Prediction save name tags

def _update_config_from_file(config, cfg_file):
    """
    Update configuration from a YAML file.

    Args:
        config (CfgNode): Configuration node to update.
        cfg_file (str): Path to the YAML configuration file.
    """
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    print(f'=> merge config from {cfg_file}')
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    """
    Update configuration based on command line arguments.

    Args:
        config (CfgNode): Configuration node to update.
        args (Namespace): Command line arguments.
    """
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        return hasattr(args, name) and eval(f'args.{name}')

    # Merge from specific arguments
    if _check_args('preds_save_name'):
        config.PREDS_SAVE_NAME = args.preds_save_name
    if _check_args('dataset'):
        config.DATA.DATASET = args.dataset
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('few_shots'):
        config.DATA.FEW_SHOTS = args.few_shots
    if _check_args('num_workers'):
        config.DATA.NUM_WORKERS = args.num_workers
    if _check_args('train_csv_path'):
        config.DATA.TRAIN_CSV_PATH = args.train_csv_path
    if _check_args('val_csv_path'):
        config.DATA.VAL_CSV_PATH = args.val_csv_path
    if _check_args('test_csv_path'):
        config.DATA.TEST_CSV_PATH = args.test_csv_path
    if _check_args('optimizer'):
        config.TRAIN.OPTIMIZER = args.optimizer
    if _check_args('scheduler'):
        config.TRAIN.SCHEDULER = args.scheduler
    if _check_args('max_epochs'):
        config.TRAIN.MAX_EPOCHS = args.max_epochs
    if _check_args('grad_clip'):
        config.TRAIN.GRAD_CLIP = args.grad_clip
    if _check_args('base_lr'):
        config.TRAIN.BASE_LR = args.base_lr
    if _check_args('min_lr'):
        config.TRAIN.MIN_LR = args.min_lr
    if _check_args('weight_decay'):
        config.TRAIN.WEIGHT_DECAY = args.weight_decay
    if _check_args('lock'):
        config.TRAIN.LOCK = args.lock
    if _check_args('pooling'):
        config.VIT.POOLING = args.pooling
    if _check_args('seed'):
        config.SEED = args.seed
    if _check_args('use_amp'):
        config.AMP_ENABLE = args.use_amp
    if _check_args('use_wandb'):
        config.WANDB.WANDB_ENABLE = args.use_wandb
    if _check_args('wandb_project'):
        config.WANDB.PROJECT = args.wandb_project
    if _check_args('model_name'):
        config.MODEL.NAME = args.model_name
    if _check_args('model_load_path'):
        config.MODEL.PRETRAINED = args.model_load_path
    if _check_args('label_name'):
        config.TRAIN.LABEL_NAME = args.label_name
    if _check_args('classifier'):
        config.TRAIN.CLASSIFIER = args.classifier
    if _check_args('filename'):
        config.LOG.FILENAME = args.filename

    # Set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # Output folder
    config.OUTPUT = os.path.join(config.OUTPUT)

    config.freeze()

def get_config(args):
    """
    Get a yacs CfgNode object with default values.

    Args:
        args (Namespace): Command line arguments.

    Returns:
        CfgNode: Configuration node with updated values.
    """
    config = _C.clone()
    update_config(config, args)
    return config
