import dataclasses
import hashlib
import json
import os.path
from itertools import chain
from typing import Collection, Sequence

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


def get_hash(*to_remove, dictionary=None, _root_=None):

    def delete_keys_from_dict(dict_del, lst_keys):
        if len(lst_keys) == 1:
            if lst_keys[0] in dict_del:
                del dict_del[lst_keys[0]]
            else:
                return

        for k, v in dict_del.items():
            if isinstance(v, dict) and k == lst_keys[0]:
                delete_keys_from_dict(v, lst_keys[1:])

        return dict_del

    if dictionary is None:
        dictionary = _root_

    if isinstance(dictionary, DictConfig):
        dictionary = {k: OmegaConf.to_container(v, resolve=True) if isinstance(v, DictConfig) else v
                      for k, v in dictionary.items()
                      if k not in ['hydra', 'device', 'core']}
    else:
        for k, v in dictionary.items():
            if k in ['hydra', 'device', 'core']:
                del dictionary[k]

    serialization = dictionary.get('serialization', {})

    for v in serialization.get('values_to_skip', []):
        delete_keys_from_dict(dictionary, v.split('.'))

    # https://death.andgravity.com/stable-hashing
    return str(hashlib.md5(json.dumps(dictionary,
                                      default=json_default,
                                      ensure_ascii=False,
                                      sort_keys=True,
                                      indent=None,
                                      separators=(',', ':')).encode('utf-8')).digest().hex())


def get_path(*to_add, dictionary=None, _root_=None):

    def get_keys_from_dict(dict_del, lst_keys):
        if len(lst_keys) == 1:
            if lst_keys[0] in dict_del:
                return dict_del[lst_keys[0]]
            else:
                return None

        for k, v in dict_del.items():
            if isinstance(v, dict) and k == lst_keys[0]:
                return get_keys_from_dict(v, lst_keys[1:])

        return None

    if dictionary is None:
        dictionary = _root_

    if isinstance(dictionary, DictConfig):
        dictionary = {k: OmegaConf.to_container(v, resolve=True) if isinstance(v, DictConfig) else v
                      for k, v in dictionary.items()
                      if k not in ['hydra', 'device', 'core']}
    else:
        for k, v in dictionary.items():
            if k in ['hydra', 'device', 'core']:
                del dictionary[k]

    serialization = dictionary.get('serialization', {})

    paths = []

    values_to_add = serialization.get('values_to_add', [])
    if isinstance(values_to_add, str):
        values_to_add = [values_to_add]

    for v in values_to_add:
        if isinstance(v, str):
            s = get_keys_from_dict(dictionary, v.split('.'))
            if s is not None:
                paths.append(s.split('.')[-1])
        elif isinstance(v, Sequence):
            s = list(map(str, [get_keys_from_dict(dictionary, z.split('.')) for z in v]))
            if not all([_s is None for _s in s]):
                paths.append('_'.join([_s if _s is not None else 'none' for _s in s]))

    values_to_prepend = serialization.get('values_to_prepend', [])
    if isinstance(values_to_prepend, str):
        values_to_prepend = [values_to_prepend]

    values_to_append = serialization.get('values_to_append', [])
    if isinstance(values_to_append, str):
        values_to_append = [values_to_append]

    paths = values_to_prepend + paths + values_to_append

    if len(paths) == 0:
        return ''

    return os.path.join(*paths)


def process_saving_path(*dicts) -> str:
    d = {f'dict{i}': OmegaConf.to_container(dd) if isinstance(dd, DictConfig) else dd
         for i, dd in enumerate(dicts)}
    return get_hash(d)
