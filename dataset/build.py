from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""

def build_dataset(dataset_name, cfg, split):
    name = dataset_name.upper()
    return DATASET_REGISTRY.get(name)(cfg, split)
