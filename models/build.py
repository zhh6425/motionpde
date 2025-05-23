from fvcore.common.registry import Registry
from transformers import PretrainedConfig
import torch

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""

def build_model(name, args, num_classes=None):
    # Construct the model
    # name = args.model_name
    if args.model_init and "Base" in name:
        # init from pretrained model
        model_config = PretrainedConfig.from_pretrained(args.model_init)
        if not model_config.train_mode == args.train_mode: # load pre-trained encoder
            model_init = args.model_init
            args.model_init = None
            model = MODEL_REGISTRY.get(name)(args, num_classes)
            model_pretrain = MODEL_REGISTRY.get(name).from_pretrained(
                model_init, 
                config=model_config, 
                num_classes=num_classes,
                )
            model.encoder = model_pretrain.encoder
            args.model_init = model_init
        else:
            model = MODEL_REGISTRY.get(name).from_pretrained(
                args.model_init, 
                config=model_config, 
                num_classes=num_classes,
            )
            
    else:
        model = MODEL_REGISTRY.get(name)(args, num_classes)
    return model
