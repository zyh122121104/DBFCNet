import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from dataloader.read_excel import create_dataloader
from dataloader.e import create_dataloader,RandomTransform,Transform
from ceus_easy import Resnet
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(train_df,val_df,test_df,args):
    transform = RandomTransform()
    transform1 = Transform()
    train_dataset = create_dataloader(train_df,transform=transform)
    val_dataset = create_dataloader(val_df, transform=transform1)
    test_dataset = create_dataloader(test_df,transform=transform1)

    return train_dataset,val_dataset,test_dataset

def model_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        in_ch=3,
        base_ch=16,
        num_class=2,
        num_frame=16,
        blocks_layers=[2, 2, 2, 2],
        attention_apply=[False, False, False, False],
        fusion_apply=[],
    )
    return res

def create_model(
    in_ch = 3,
    base_ch = 16,
    num_class = 2,
    num_frame = 16,
    blocks_layers = [2,2,2,2],
    attention_apply = [False,False,False,False],
    fusion_apply = [],
):
    return Resnet(
    in_ch = in_ch,
    base_ch = base_ch,
    num_class = num_class,
    num_frame = num_frame,
    blocks_layers = blocks_layers,
    attention_apply = attention_apply,
    fusion_apply = fusion_apply,
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")