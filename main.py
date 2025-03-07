import collections
import json
import logging
import os
import pickle
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from ast import literal_eval
from typing import Sequence

import hydra

import numpy as np
import tqdm.auto as tqdm
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
import torch
from timm.models import VisionTransformer
from torch import nn

from torch.utils.data import DataLoader

from comm.channel import GaussianNoiseChannel
from comm.evaluation import digital_jpeg, digital_resize, analog_resize, gaussian_snr_activations
from methods import SemanticVit
from methods.base import evaluation
from methods.proposal import AdaptiveBlock, semantic_evaluation
from serialization import get_hash, get_path
from utils import get_pretrained_model, CommunicationPipeline


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


OmegaConf.register_new_resolver("to_hash", get_hash)
OmegaConf.register_new_resolver("get_path", get_path)


@hydra.main(config_path="configs",
            version_base='1.2',
            config_name="default")
def main(cfg: DictConfig):
    # A logger for this file
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg, resolve=True))

    device = cfg.get('device', 'cpu')
    if device == 'cpu':
        warnings.warn("Device set to cpu.")
    elif torch.cuda.is_available():
        torch.cuda.set_device(device)
        device = 'cuda:{}'.format(device)
    else:
        warnings.warn(f"Device not found {device} "
                      f"or CUDA {torch.cuda.is_available()}")

    device = torch.device(device)

    to_download = not os.path.exists(cfg.training_pipeline.dataset.train.root)

    train_dataset = hydra.utils.instantiate(cfg.training_pipeline.dataset.train, download=to_download,
                                            _convert_="partial")
    test_dataset = hydra.utils.instantiate(cfg.training_pipeline.dataset.test, _convert_="partial")

    training_schema = cfg.training_pipeline.schema
    dev_split = training_schema.get('dev_split', None)

    outer_experiment_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    log.info(f'Saving path: {outer_experiment_path}')

    for seed in range(training_schema.experiments):

        log.info(f'Experiment N{seed + 1}')

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        experiment_path = os.path.join(outer_experiment_path, str(seed))

        model_path = os.path.join(experiment_path, 'model.pt')
        wadb_id_path = os.path.join(experiment_path, 'wandbid.pkl')
        plot_path = os.path.join(experiment_path, 'evaluation_results')
        evaluation_results = os.path.join(experiment_path, 'evaluation_results')
        opt_results_path = os.path.join(experiment_path, 'opt_results')

        os.makedirs(experiment_path, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)
        os.makedirs(evaluation_results, exist_ok=True)
        os.makedirs(opt_results_path, exist_ok=True)

        datalaoder_wrapper = hydra.utils.instantiate(
            cfg.training_pipeline.dataloader, _partial_=True)

        # if 'dev_dataloader' in cfg.training_pipeline.dataloader:
        #     test_dataloader = hydra.utils.instantiate(
        #         cfg.training_pipeline.dataloader, dataset=test_dataset)
        # else:

        all_y = [y for x, y in train_dataset]

        classes = len(np.unique(all_y))
        dev_dataloader = None
        dev_dataset = None

        if dev_split is not None and dev_split > 0:
            idx = np.arange(len(train_dataset))
            np.random.shuffle(idx)

            if isinstance(dev_split, int):
                dev_i = dev_split
            else:
                dev_i = int(len(idx) * dev_split)

            dev_idx = idx[:dev_i]
            train_idx = idx[dev_i:]

            dev_dataset = torch.utils.data.Subset(train_dataset, dev_idx)
            train_dataset = torch.utils.data.Subset(train_dataset, train_idx)

        test_dataloader = datalaoder_wrapper(dataset=test_dataset, shuffle=False,
                                             batch_size=cfg.training_pipeline.schema.eval_batch_size)

        train_dataloader = datalaoder_wrapper(dataset=train_dataset, shuffle=True,
                                              batch_size=cfg.training_pipeline.schema.train_batch_size)

        if 'mobilenet' in cfg.model._target_:
            model = hydra.utils.instantiate(cfg.model)
            # if classes > 1000:
            #     pass
        else:
            model = hydra.utils.instantiate(cfg.model, num_classes=classes)

        if 'pretraining_pipeline' in cfg:
            pre_cfg = cfg.pretraining_pipeline
            # pre_cfg = OmegaConf.merge(cfg.pretraining_pipeline, cfg.model)
            OmegaConf.update(pre_cfg, 'model', cfg.model, force_add=True)
            # pre_cfg = OmegaConf.to_container(pre_cfg, resolve=True)

            pre_cfg['model'] = OmegaConf.to_container(cfg.model, resolve=True)

            hash_path = get_hash(dictionary=pre_cfg)

            pre_trained_path = os.path.join(cfg.core.pretrained_root, hash_path)
            os.makedirs(pre_trained_path, exist_ok=True)

            premodel_path = os.path.join(pre_trained_path, f'model_{seed}.pt')

            if os.path.exists(premodel_path):
                model.load_state_dict(torch.load(premodel_path))
                model = model.to(device)
            else:
                model = get_pretrained_model(pre_cfg, model, device)
                torch.save(model.state_dict(), premodel_path)

            model.eval()
            with torch.no_grad():
                t, c = 0, 0
                for x, y in test_dataloader:
                    x, y = x.to(device), y.to(device)

                    pred = model(x)
                    c += (pred.argmax(-1) == y).sum().item()
                    t += len(x)

            log.info(f'Pre trained model score: {c}, {t}, ({c / t})')
            model = model.cpu()

        model = model.to(device)

        if 'method' in cfg and 'model' in cfg.method:
            model = hydra.utils.instantiate(cfg.method.model, model=model)

        if 'training_pipeline' in cfg:
            use_ptmodel = cfg.training_pipeline.schema.get('use_pretrained_model', False)
            if os.path.exists(model_path) or use_ptmodel:
                if not use_ptmodel:
                    model_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(model_dict)
                    log.info(f'Model loaded')
                else:
                    log.info(f'Using the pretrained model')
            else:
                log.info(f'Training of the model')

                gradient_clipping_value = cfg.training_pipeline.schema.get('gradient_clipping_value', None)

                loss_f = hydra.utils.instantiate(cfg.method.loss, model=model)
                optimizer = hydra.utils.instantiate(cfg.training_pipeline.optimizer,
                                                    params=model.parameters())

                scheduler = None
                if 'scheduler' in cfg:
                    scheduler = hydra.utils.instantiate(cfg.training_pipeline.scheduler,
                                                        optimizer=optimizer)

                bar = tqdm.tqdm(range(cfg.training_pipeline.schema.epochs),
                                leave=False,
                                desc='Model training')
                epoch_losses = []

                for epoch in bar:
                    model.train()
                    for x, y in train_dataloader:
                        x, y = x.to(device), y.to(device)

                        pred = model(x)
                        loss = loss_f(pred, y)

                        epoch_losses.append(loss.item())

                        optimizer.zero_grad()
                        loss.backward()

                        if gradient_clipping_value is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_value)

                        optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

                    log.info(f'Training epoch {epoch} ended')

                    with torch.no_grad():
                        model.eval()

                        t, c = 0, 0

                        for x, y in test_dataloader:
                            x, y = x.to(device), y.to(device)

                            pred = model(x)
                            c += (pred.argmax(-1) == y).sum().item()
                            t += len(x)

                        score = c / t

                    bar.set_postfix({'Test acc': score, 'Epoch loss': np.mean(epoch_losses)})

                torch.save(model.state_dict(), model_path)

        final_evaluation = cfg.get('final_evaluation', {})
        if final_evaluation is None:
            final_evaluation = {}

        log.info(f'Final evaluation')

        for key, value in final_evaluation.items():
            overwrite = value.get('overwrite', False)

            if not os.path.exists(os.path.join(evaluation_results, f'{key}.json')) or overwrite:

                with warnings.catch_warnings(action="ignore"):
                    results = hydra.utils.instantiate(value, dataset=test_dataset, model=model)

                if results is not None:
                    with open(os.path.join(evaluation_results, f'{key}.json'), 'w') as f:
                        json.dump(results, f, ensure_ascii=True, indent=4)

        log.info(f'Comm baselines evaluation')

        snr = np.arange(-20, 20 + 1, 2.5)
        kn = np.linspace(0.01, 1., num=20, endpoint=True)

        ##########################
        ##### ANALOG RESIZE #####
        ##########################
        # analog_resize_results = None
        # if os.path.exists(os.path.join(evaluation_results, f'digital_resize.json')):
        #     try:
        #         with open(os.path.join(evaluation_results, f'digital_resize.json'), 'r') as f:
        #             analog_resize_results = json.load(f)
        #     except:
        #         pass
        #
        # analog_resize_results = analog_resize(model=model, dataset=test_dataset, kn=kn, snr=snr, batch_size=256,
        #                                       previous_results=analog_resize_results)
        #
        # with open(os.path.join(evaluation_results, f'analog_resize.json'), 'w') as f:
        #     json.dump(analog_resize_results, f, ensure_ascii=True, indent=4)
        # #
        # log.info(f'analog_resize baselines evaluation ended')

        ########################
        ### DIGITAL RESIZE #####
        ########################
        # digital_resize_results = None
        # if os.path.exists(os.path.join(evaluation_results, f'digital_resize.json')):
        #     try:
        #         with open(os.path.join(evaluation_results, f'digital_resize.json'), 'r') as f:
        #             digital_resize_results = json.load(f)
        #     except Exception as e:
        #         print(e)
        #
        # digital_resize_results = digital_resize(model=model, dataset=test_dataset, kn=kn, snr=snr, batch_size=256,
        #                                         previous_results=digital_resize_results)
        #
        # with open(os.path.join(evaluation_results, f'digital_resize.json'), 'w') as f:
        #     json.dump(digital_resize_results, f, ensure_ascii=True, indent=4)
        #
        # log.info(f'digital_resize baselines evaluation ended')
        #
        # ############################
        # ####### DIGITAL   JPEG #####
        # ############################
        #
        # jpeg_results = None
        # if os.path.exists(os.path.join(evaluation_results, f'digital_jpeg.json')):
        #     try:
        #         with open(os.path.join(evaluation_results, f'digital_jpeg.json'), 'r') as f:
        #             jpeg_results = json.load(f)
        #     except Exception as e:
        #         print(e)
        #
        # jpeg_results = digital_jpeg(model=model, dataset=test_dataset, kn=kn, snr=snr, batch_size=256,
        #                             previous_results=jpeg_results)
        #
        # with open(os.path.join(evaluation_results, f'digital_jpeg.json'), 'w') as f:
        #     json.dump(jpeg_results, f, ensure_ascii=True, indent=4)
        #
        # log.info(f'digital_jpeg baselines evaluation ended')

        if cfg.get('jscc', None) is not None:

            average_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            models = defaultdict(dict)

            for experiment_key, experiment_cfg in cfg['jscc'].items():
                compression = experiment_cfg.encoder.output_size
                use_for_opt_problem = experiment_cfg.get('use_for_opt_problem', False)
                opt_index = experiment_cfg.get('opt_index', None)
                # assert opt_index is not None, experiment_cfg

                # opt_index = opt_index if isinstance(opt_index, Sequence) else [opt_index]

                log.info(f'Comm experiment called {experiment_key}')

                comm_experiment_path = os.path.join(experiment_path, experiment_key)
                os.makedirs(comm_experiment_path, exist_ok=True)

                splitting_point = experiment_cfg.splitting_point

                comm_model = deepcopy(model)

                if isinstance(model, (VisionTransformer, SemanticVit)):

                    if splitting_point > 0:
                        # splitting_point = splitting_point + 1

                        if experiment_cfg.get('unwrap_after', False):
                            for i, b in enumerate(comm_model.blocks[splitting_point:]):
                                if hasattr(b, 'base_block'):
                                    comm_model.blocks[i] = b.base_block

                        if experiment_cfg.get('unwrap_before', False):
                            for i, b in enumerate(comm_model.blocks[:splitting_point]):
                                if hasattr(b, 'base_block'):
                                    comm_model.blocks[i] = b.base_block

                        blocks_before = comm_model.blocks[:splitting_point]
                        blocks_after = comm_model.blocks[splitting_point:]

                    else:
                        blocks_before = comm_model.blocks
                        blocks_after = None

                        if experiment_cfg.get('unwrap_after', False):
                            for i, b in enumerate(comm_model.blocks):
                                if hasattr(b, 'base_block'):
                                    comm_model.blocks[i] = b.base_block

                    comm_model_path = os.path.join(comm_experiment_path, 'comm_model.pt')
                    # os.makedirs(comm_model_path, exist_ok=True)

                    encoder = hydra.utils.instantiate(experiment_cfg.encoder, input_size=comm_model.num_features)
                    decoder = hydra.utils.instantiate(experiment_cfg.decoder, input_size=encoder.output_size,
                                                      output_size=comm_model.num_features)

                    channel = experiment_cfg.get('channel', None)

                    if channel is not None:
                        channel = hydra.utils.instantiate(channel)

                    communication_pipeline = CommunicationPipeline(channel=channel, encoder=encoder,
                                                                   decoder=decoder).to(
                        device)

                    if isinstance(model, SemanticVit):

                        if blocks_after is not None:
                            comm_model._model.blocks = nn.Sequential(*blocks_before, communication_pipeline,
                                                                     *blocks_after)
                        else:
                            comm_model._model.blocks = nn.Sequential(*blocks_before, communication_pipeline)
                    else:
                        if blocks_after is not None:
                            comm_model.blocks = nn.Sequential(*blocks_before, communication_pipeline, *blocks_after)
                        else:
                            comm_model.blocks = nn.Sequential(*blocks_before, communication_pipeline)
                else:

                    use_cnn_ae = experiment_cfg.get('use_cnn_ae', False)
                    blocks_before = comm_model.features[:splitting_point]
                    blocks_after = comm_model.features[splitting_point:]

                    oo = blocks_before(x)

                    if not use_cnn_ae:
                        flatten_oo = torch.flatten(oo, 2)
                        input_size = flatten_oo.shape[-1]
                        flatten = nn.Flatten(start_dim=2)
                        unflatten = nn.Unflatten(dim=-1, unflattened_size=oo.shape[2:])

                    else:
                        input_size = oo.shape[1:]

                    comm_model_path = os.path.join(comm_experiment_path, 'comm_model.pt')
                    # os.makedirs(comm_model_path, exist_ok=True)

                    encoder = hydra.utils.instantiate(experiment_cfg.encoder, input_size=input_size)

                    if not use_cnn_ae:
                        ins = encoder(blocks_before(x).cpu().flatten(2)).shape
                        ins = ins[-1]
                    else:
                        ins = encoder(blocks_before(x).cpu()).shape
                        ins = ins[1:]

                    decoder = hydra.utils.instantiate(experiment_cfg.decoder, input_size=ins,
                                                      output_size=input_size)

                    channel = experiment_cfg.get('channel', None)

                    if channel is not None:
                        channel = hydra.utils.instantiate(channel, _convert_='partial')

                    communication_pipeline = CommunicationPipeline(channel=channel, encoder=encoder,
                                                                   decoder=decoder).to(
                        device)

                    # if blocks_after is not None:
                    if not use_cnn_ae:
                        comm_model.features = nn.Sequential(*blocks_before, flatten, communication_pipeline, unflatten,
                                                            *blocks_after)
                    else:
                        comm_model.features = nn.Sequential(*blocks_before, communication_pipeline, *blocks_after)
                    # else:
                    #     comm_model.features = nn.Sequential(*blocks_before, communication_pipeline)

                # gaussian_snr_activations(comm_model, test_dataset, 10, evaluation)

                overwrite_model = experiment_cfg.get('overwrite_model', False)
                overwrite_evaluation = experiment_cfg.get('overwrite_evaluation', overwrite_model)

                if os.path.exists(comm_model_path) and not overwrite_model:
                    model_dict = torch.load(comm_model_path, map_location=device)
                    comm_model.load_state_dict(model_dict)

                    log.info(f'Comm model loaded')
                else:
                    overwrite_evaluation = True

                    log.info(f'Training the model')

                    gradient_clipping_value = cfg.training_pipeline.schema.get('gradient_clipping_value', None)

                    freeze_model = experiment_cfg.get('freeze_model', True)
                    mse = experiment_cfg.get('mse', False)

                    if mse:
                        loss_f = nn.MSELoss()
                        optimizer = hydra.utils.instantiate(cfg.training_pipeline.optimizer,
                                                            params=communication_pipeline.parameters())
                    else:
                        loss_f = nn.CrossEntropyLoss()

                        if not freeze_model:
                            optimizer = hydra.utils.instantiate(cfg.training_pipeline.optimizer,
                                                                params=comm_model.parameters())
                        else:
                            optimizer = hydra.utils.instantiate(cfg.training_pipeline.optimizer,
                                                                params=chain(communication_pipeline.parameters(),
                                                                             blocks_after.parameters()))

                    scheduler = None
                    if 'scheduler' in cfg:
                        scheduler = hydra.utils.instantiate(cfg.training_pipeline.scheduler,
                                                            optimizer=optimizer)

                    bar = tqdm.tqdm(range(experiment_cfg.epochs),
                                    leave=False,
                                    desc='Comm model training')
                    epoch_losses = []

                    blocks_before.eval()

                    for epoch in bar:
                        comm_model.train()

                        # communication_pipeline.train()
                        if freeze_model or mse:
                            blocks_before.eval()
                        if mse:
                            blocks_after.eval()

                        for x, y in train_dataloader:
                            x, y = x.to(device), y.to(device)

                            pred = comm_model(x)
                            if mse:
                                loss = loss_f(communication_pipeline.input, communication_pipeline.output)
                            else:
                                loss = loss_f(pred, y)

                            epoch_losses.append(loss.item())

                            optimizer.zero_grad()
                            loss.backward()

                            if gradient_clipping_value is not None:
                                torch.nn.utils.clip_grad_norm_(comm_model.parameters(), gradient_clipping_value)

                            optimizer.step()

                        if scheduler is not None:
                            scheduler.step()

                    torch.save(comm_model.state_dict(), comm_model_path)

                # if use_for_opt_problem:
                #     # models[str(compression)] = deepcopy(comm_model).eval().cpu()
                #     models[str(compression)] = deepcopy(comm_model).eval()

                if opt_index is not None:
                # for opt_i in opt_index if isinstance(opt_index, Sequence) else [opt_index]:
                    models[opt_index][str(compression)] = deepcopy(comm_model.cpu()).eval()

                final_evaluation = cfg.get('comm_evaluation', {})
                if final_evaluation is None:
                    final_evaluation = {}

                comm_model.eval()

                keys_to_use = []

                for key, value in final_evaluation.items():
                    pkl = value.get('use_pickle', False)
                    if pkl:
                        current_path = os.path.join(comm_experiment_path, f'{key}.pkl')
                    else:
                        current_path = os.path.join(comm_experiment_path, f'{key}.json')

                    if not os.path.exists(current_path) or overwrite_evaluation:
                        with warnings.catch_warnings(action="ignore"):
                            results = hydra.utils.instantiate(value,
                                                              dataset=test_dataset,
                                                              model=comm_model,
                                                              _convert_="partial")

                        if results is not None:
                            if not pkl:
                                with open(current_path, 'w') as f:
                                    json.dump(results, f, ensure_ascii=True, indent=4, cls=NpEncoder)
                            else:
                                with open(current_path, 'wb') as f:
                                    pickle.dump(results, f)

                        log.info(f'{key} evaluation ended')

                    if opt_index is not None and key == 'semantic':
                        with open(os.path.join(comm_experiment_path, f'{key}.json'), 'r') as f:
                            results = json.load(f)

                        if '0' in results and len(results) <= 5:
                            results = results['0']

                        for snr, vals in results.items():
                            snr = snr
                            accuracy = vals['accuracy']

                            if 'all_sizes' in vals:
                                sizes = vals['all_sizes']
                                for k in accuracy.keys():
                                    acc = accuracy[k]
                                    size = list(sizes[k].values())[-1][0] * (192 * compression)

                                    _results = []
                                    # for opt_i in opt_index:
                                    average_results[opt_index][snr][k][compression] = {'accuracy': acc, 'kn': size / (224 * 224 * 3)}
                            else:
                                # for opt_i in opt_index:
                                average_results[opt_index][snr][compression] = {'accuracy': accuracy, 'kn': vals['symbols'] / (224 * 224 * 3)}

                                    # _results.append({'accuracy': acc, 'kn': size / (224 * 224 * 3)})
                                    #
                                    # average_results[opt_i][snr][compression] = {'accuracy': accuracy,
                                    #                                             'kn': vals['symbols'] / (224 * 224 * 3)}
                            # for opt_i in opt_index:
                            #     average_results[opt_index][snr][compression] = _results

            # MINIMIZATION PROBLEM

            def z_f(t, zetas, gammas, gamma_avg, mu):
                if t == 0:
                    return 0

                return max(0, zetas[t - 1] + mu * (gammas[t - 1] - gamma_avg))

            def get_mapping(snr, results):
                def flatten(dictionary, parent_key='', separator='_'):

                    items = []
                    for key, value in dictionary.items():
                        new_key = str(parent_key) + separator + str(key) if parent_key else key
                        if isinstance(value, collections.abc.MutableMapping):
                            if len(value) == 2 and 'accuracy' in value and 'kn' in value:
                                items.append((new_key, value))
                            else:
                                items.extend(flatten(value, new_key, separator).items())
                        elif isinstance(value, list):
                            for k, v in enumerate(value):
                                items.extend(flatten(v, f'{k}_{new_key}').items())
                        else:
                            items.append((new_key, value))
                    return dict(items)

                closest_snr = min(results.keys(), key=lambda x: abs(float(x) - snr))

                kns = []

                for k, v in flatten(results[closest_snr]).items():
                    kns.append((k, v['kn'], v['accuracy']))

                # for alpha, va in results[closest_snr].items():
                #     for c, vc in va.items():
                #         kns.append((alpha, c, vc['kn'], vc['accuracy']))

                return kns

            opt_cfg = {
                '20_0': {
                    'v': [1, 10, 100, 1000, 10000, 100000],
                    'mu': [1, 10, 100],
                    'kn': [0.05, 0.005, 0.0025, 0.001, 0.02, 0.01, 0.1],
                    'T': 10000,
                    'snr_mu': 20,
                    'snr_sigma': 0
                },
                '10_2.5': {
                    'v': [1, 10, 100, 1000, 10000, 100000],
                    'mu': [1, 10, 100],
                    'kn': [0.005, 0.0025, 0.001, 0.02, 0.05, 0.02, 0.01, 0.1],
                    'T': 10000,
                    'snr_mu': 10,
                    'snr_sigma': 2.5
                },
                '20_2.5': {
                    'v': [1, 10, 100, 1000, 10000, 100000],
                    'mu': [1, 10, 100],
                    'kn': [0.005, 0.0025, 0.001, 0.02, 0.05, 0.02, 0.01, 0.1],
                    'T': 10000,
                    'snr_mu': 20,
                    'snr_sigma': 2.5
                },
                '0_2.5': {
                    'v': [1, 10, 100, 1000, 10000, 100000],
                    'mu': [1, 10, 100],
                    'kn': [0.005, 0.0025, 0.001, 0.02, 0.05, 0.02, 0.01, 0.1],
                    'T': 10000,
                    'snr_mu': 0,
                    'snr_sigma': 2.5
                },
                '10_0': {
                    'v': [1, 10, 100, 1000, 10000, 100000],
                    'mu': [1, 10, 100],
                    'kn': [0.005, 0.0025, 0.001, 0.02, 0.05, 0.02, 0.01, 0.1],
                    'T': 10000,
                    'snr_mu': 10,
                    'snr_sigma': 0
                },
                '0_7.5': {
                    'v': [1, 10, 100, 1000, 10000, 100000],
                    'mu': [1, 10, 100],
                    'kn': [0.005, 0.0025, 0.001, 0.02, 0.05, 0.02, 0.01, 0.1],
                    'T': 10000,
                    'snr_mu': 0,
                    'snr_sigma': 7.5
                },
            }

            for opt_index, collated_results in average_results.items():
                for opt_name, vals in opt_cfg.items():
                    log.info(f'Optimization {opt_name} (index {opt_index}) started.')

                    # opt_index == 1 is for compatibility with past results. TO BE REMOVED WHEN PUBLISHED

                    opt_file_name = f'{opt_name}.json' if opt_index == 1 else f'{opt_name}_{opt_index}.json'
                    opt_offline_file_name = f'offline_{opt_file_name}'

                    opt_path = os.path.join(opt_results_path, opt_file_name)
                    opt_offline_path = os.path.join(opt_results_path, opt_offline_file_name)

                    snr_f = lambda: 1 * np.random.normal(vals['snr_mu'], vals['snr_sigma'], 1)[0]

                    Vs = vals['v']
                    Ms = vals['mu']
                    KNs = vals['kn']
                    T = vals['T']

                    opt_results = {}
                    if os.path.exists(opt_path):
                        with open(opt_path, 'r') as f:
                            try:
                                opt_results = json.load(f)
                            except json.decoder.JSONDecodeError:
                                log.info(f'Error loading file: {opt_path}')

                    for kn in KNs:
                        for V in Vs:
                            for mu in Ms:
                                if str((kn, V, mu)) in opt_results:
                                    continue

                                acc = []

                                gamma_s = []
                                zeta_s = []

                                for i in tqdm.tqdm(range(T), leave=False):
                                    snr = snr_f()
                                    z = z_f(i, mu=mu, gamma_avg=kn, gammas=gamma_s, zetas=zeta_s)

                                    mapping = get_mapping(snr, results=collated_results)

                                    min_ac = np.argmin([-V * accuracy + z * kn for _, kn, accuracy in mapping])
                                    _, kn_s, accuracy = mapping[min_ac]

                                    # print(kn, V, mu, alpha, c, kn_s)
                                    acc.append(accuracy)

                                    gamma_s.append(kn_s)
                                    zeta_s.append(z)

                                opt_results[str((kn, V, mu))] = {'gamma': gamma_s, 'accuracy': acc, 'zeta': zeta_s}

                                with open(opt_path, 'w') as f:
                                    json.dump(opt_results, f, cls=NpEncoder)

                                # print(opt_index, opt_name, 'Gamma', np.mean(gamma_s[-1000:]), kn, V, mu)
                                # print('Accuracy', np.mean(acc[-1000:]), (
                                #         (np.asarray(gamma_s)[-100 + 1:] - np.asarray(gamma_s)[-100:-1]) / np.asarray(
                                #     gamma_s)[-100 + 1:]).mean())

                    best_values = defaultdict(list)

                    for key, value in opt_results.items():
                        kn, V, mu = literal_eval(key)

                        gammas = value['gamma']
                        zetas = value['zeta']

                        if np.mean(gammas[-1000:]) < kn:
                            best_values[kn].append((np.mean(value['accuracy'][-1000:]), key))

                    keys = sorted(best_values.keys())

                    offline_results = {}
                    if os.path.exists(opt_offline_path):
                        with open(opt_offline_path, 'r') as f:
                            offline_results = json.load(f)

                    for k in keys:
                        mx = max(best_values[k], key=lambda x: literal_eval(x[1])[1])
                        # mx = max(best_values[k], key=lambda x: x[0])
                        if mx[1] in offline_results:
                            continue

                        kn, V, mu = literal_eval(mx[1])

                        gammas = opt_results[mx[1]]['gamma']
                        zetas = opt_results[mx[1]]['zeta']

                        with torch.no_grad():
                            ccs, gs, dcs = [], [], []

                            for n in range(3 if vals['snr_sigma'] > 0 else 1):
                                gamma_s = deepcopy(gammas)
                                zeta_s = deepcopy(zetas)

                                c, t, dc = 0, 0, 0
                                g = []

                                for x, y in tqdm.tqdm(DataLoader(test_dataset, shuffle=True, batch_size=1),
                                                      leave=False):
                                    alpha = None
                                    snr = snr_f()
                                    z = z_f(len(gamma_s), mu=mu, gamma_avg=kn, gammas=gamma_s, zetas=zeta_s)

                                    mapping = get_mapping(snr, results=collated_results)

                                    min_ac = np.argmin([-V * accuracy + z * _kn for _, _kn, accuracy in mapping])
                                    kkk, _, _ = mapping[min_ac]

                                    if isinstance(kkk, str):
                                        kkk = tuple(map(float, kkk.split('_')))
                                        if len(kkk) == 1:
                                            compression = kkk[0]
                                        elif len(kkk) == 2:
                                            alpha, compression = kkk
                                        else:
                                            assert False
                                    else:
                                        compression = kkk

                                    sample_model = models[opt_index][str(compression)].eval()
                                    x = x.to(next(sample_model.parameters()).device)
                                    y = y.to(next(sample_model.parameters()).device)

                                    channel = [m for m in sample_model.modules()
                                               if isinstance(m, GaussianNoiseChannel)][0]
                                    channel.test_snr = snr

                                    if alpha is None:
                                        pred = sample_model(x)
                                    else:
                                        pred = sample_model(x, alpha=alpha)

                                    _g = np.prod(channel.symbols) / (224 * 224 * 3)
                                    g.append(_g)

                                    classification = (pred.argmax(-1) == y).sum().item()

                                    c += classification
                                    dc += classification * (_g <= kn)

                                    t += len(x)

                                    gamma_s.append(_g)
                                    zeta_s.append(z)

                                ccs.append(c / t)

                                dcs.append(dc / t)
                                gs.append(np.mean(g))

                            print(opt_name, k, np.mean(gs), np.mean(ccs))

                        offline_results[mx[1]] = {'accuracy': ccs,
                                                  'constrained_accuracy': dcs,
                                                  'gamma': gs}

                    with open(opt_offline_path, 'w') as f:
                        json.dump(offline_results, f, cls=NpEncoder)


if __name__ == "__main__":
    main()
