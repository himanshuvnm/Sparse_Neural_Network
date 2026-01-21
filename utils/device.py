import torch

def get_device(cfg_device="auto"):
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return cfg_device
