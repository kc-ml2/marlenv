import os
import torch


class ArgumentParser(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def to_onehot(tensor, size):
    assert len(tensor.shape) == 1
    tensor = tensor.unsqueeze(1)
    onehot = torch.zeros(tensor.size(0), size)
    onehot.scatter_(1, tensor, 1).long()
    return onehot


def to_idx(onehot):
    assert len(onehot.shape) == 2
    idx = onehot.nonzero()[:,1]
    return idx


def save_model(model, path):
    pth = {
        'model': model,
        'checkpoint': model.state_dict()
    }
    if os.path.isfile(path):
        os.remove(path)
    torch.save(pth, path)


def load_model(path):
    pth = torch.load(path, map_location=lambda storage, loc: storage)
    model = pth['model']
    model.load_state_dict(pth['checkpoint'])
    return model
