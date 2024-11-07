import dataclasses
import hashlib
import json
from typing import Collection

from omegaconf import OmegaConf, DictConfig


def dict_drop_empty(pairs):
    return dict(
        (k, v)
        for k, v in pairs
        if not (
            v is None
            or not v and isinstance(v, Collection)
        )
    )


def json_default(thing):
    try:
        return dataclasses.asdict(thing, dict_factory=dict_drop_empty)
    except TypeError:
        pass
    raise TypeError(f"object of type {type(thing).__name__} not serializable")


def get_hash(dictionary=None, *, _root_=None):
    if dictionary is None:
        dictionary = _root_

    if isinstance(dictionary, DictConfig):
        dictionary = {k: OmegaConf.to_container(v, resolve=True) if isinstance(v, DictConfig) else v
                      for k, v in dictionary.items()
                      if k not in ['hydra', 'device', 'core']}

    if 'training_pipeline' in dictionary:
        del dictionary['training_pipeline']['schema']['experiments']

    if 'final_evaluation' in dictionary:
        del dictionary['final_evaluation']

    if 'serialization' in dictionary:
        del dictionary['serialization']

    # https://death.andgravity.com/stable-hashing
    return str(hashlib.md5(json.dumps(dictionary,
                                      default=json_default,
                                      ensure_ascii=False,
                                      sort_keys=True,
                                      indent=None,
                                      separators=(',', ':')).encode('utf-8')).digest().hex())

def process_saving_path(*dicts) -> str:
    d = {f'dict{i}': OmegaConf.to_container(dd) if isinstance(dd, DictConfig) else dd
         for i, dd in enumerate(dicts)}
    return get_hash(d)
