import argparse

# 1D models
from models.model_1d.cnn_2d_to_1d import Network1D as CNN_2D_TO_1D
from models.model_1d.vit_2d_to_1d import DynamicViT as VIT_2D_TO_1D


# 2D models
from models.model_2d.ae import Network2D as AE
from models.model_2d.ae_2d_to_2d import Network2D as AE_2D_TO_2D
from models.model_2d.ae_2d_to_2d_multi_task import Network2D as AE_2D_TO_2D_MULTI_TASK
from models.model_2d.ae_6_2d_to_6_2d import Network2D as AE_6_2D_TO_6_2D

# 2D models - SOTA
from models.sota.model_2d.gap_cnn.main import Network2D as GAP_CNN
from models.sota.model_2d.vgg_ae_demo.main import Network2D as VGG_AE_DEMO


# 3D models
from models.model_3d.ae_6_2d_to_3d import Network3D as AE_6_2D_TO_3D
from models.model_3d.ae_3d_to_3d import Network3D as AE_3D_TO_3D

# 3D models - SOTA
from models.sota.model_3d.unet3d.main import Network3D as UNet3D


def init_model(args: argparse.Namespace):
    print(f"[Model: '{args.model}'] Initializing...")
    model_map = {
        # 1D models
        "cnn_2d_to_1d": CNN_2D_TO_1D,
        "vit_2d_to_1d": VIT_2D_TO_1D,

        # 2D models
        "ae": AE,
        "ae_2d_to_2d": AE_2D_TO_2D,
        "ae_2d_to_2d_multi_task": AE_2D_TO_2D_MULTI_TASK,
        "ae_6_2d_to_6_2d": AE_6_2D_TO_6_2D,

        # 2D sota models
        "gap_cnn": GAP_CNN,
        "vgg_ae_demo": VGG_AE_DEMO,

        # 3D models
        "ae_6_2d_to_3d": AE_6_2D_TO_3D,
        "ae_3d_to_3d": AE_3D_TO_3D,

        # 3D sota models
        "unet3d": UNet3D,
    }
    if args.model in list(model_map.keys()):
        return model_map[args.model](args=args)
    else:
        raise Exception("Model not available in 'Model List'")
