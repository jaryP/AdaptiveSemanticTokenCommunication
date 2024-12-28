import json
import logging
import os
import random
import warnings
from collections import defaultdict
from copy import deepcopy

import hydra

import numpy as np
import tqdm.auto as tqdm
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

from torch.utils.data import DataLoader

from comm.evaluation import digital_jpeg, digital_resize
from methods.proposal import AdaptiveBlock
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

        os.makedirs(experiment_path, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)
        os.makedirs(evaluation_results, exist_ok=True)

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

        ############################
        ####### DIGITAL RESIZE #####
        ############################
        # digital_resize_results = None
        # if os.path.exists(os.path.join(evaluation_results, f'digital_resize.json')):
        #     try:
        #         with open(os.path.join(evaluation_results, f'digital_resize.json'), 'w') as f:
        #             digital_resize_results = json.load(f)
        #     except:
        #         pass
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
        #         with open(os.path.join(evaluation_results, f'digital_jpeg.json'), 'w') as f:
        #             jpeg_results = json.load(f)
        #     except:
        #         pass
        #
        # jpeg_results = digital_jpeg(model=model, dataset=test_dataset, kn=kn, snr=snr, batch_size=256,
        #                             previous_results=jpeg_results)
        #
        # with open(os.path.join(evaluation_results, f'digital_jpeg.json'), 'w') as f:
        #     json.dump(jpeg_results, f, ensure_ascii=True, indent=4)
        #
        # log.info(f'digital_jpeg baselines evaluation ended')

        if 'jscc' in cfg:

            # from comm.channel import GaussianNoiseChannel
            #
            # blocks_before = model.blocks[:6]
            # blocks_after = model.blocks[6:]
            #
            # for i, b in enumerate(blocks_after):
            #     if hasattr(b, 'base_block'):
            #         model.blocks[i] = b.base_block
            #
            # channel = GaussianNoiseChannel(snr=(-50, 50))
            #
            # model.blocks = nn.Sequential(*blocks_before, channel, *blocks_after)
            #
            # model.eval()
            # channel_results = defaultdict(dict)
            #
            # last_block = [b for b in model.blocks if isinstance(b, AdaptiveBlock)][-1]
            #
            # for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            #     average_masks = 0
            #
            #     for snr in np.arange(-50, 50, 5):
            #         channel.test_snr = snr
            #
            #         c, t = 0, 0
            #         band = 0
            #
            #         for x, y in DataLoader(test_dataset, batch_size=32, shuffle=True):
            #             x = x.to(device)
            #
            #             pred = model(x, alpha=alpha)
            #             band += pred.shape[1]
            #
            #             average_masks += ((last_block.last_mask > 0).float().sum(1) / 2 * 198).sum()
            #
            #             c += (pred.argmax(-1).cpu() == y).sum().item()
            #             t += len(x)
            #
            #         average_masks = average_masks / t
            #         channel_results[int(snr)][alpha] = (c / t, average_masks.item())
            #         print(snr, alpha, channel_results[int(snr)][alpha])
            #
            # with open(f'accuracy_results.json', 'w') as f:
            #     json.dump(channel_results, f, cls=NpEncoder)
            #
            # exit()

            for experiment_key, experiment_cfg in cfg['jscc'].items():
                log.info(f'Comm experiment called {experiment_key}')

                comm_experiment_path = os.path.join(experiment_path, experiment_key)
                os.makedirs(comm_experiment_path, exist_ok=True)

                splitting_point = experiment_cfg.splitting_point

                comm_model = deepcopy(model)

                # log.info(dict(comm_model.named_parameters()).keys())

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

                fine_tuning_cfg = experiment_cfg.get('fine_tuning', None)

                # if fine_tuning_cfg is not None:
                #     if os.path.exists(os.path.join(experiment_path, 'fine_tuned_model.pt')):
                #         model_dict = torch.load(os.path.join(experiment_path, 'fine_tuned_model.pt'),
                #                                 map_location=device)
                #         log.info(model_dict.keys())
                #
                #         comm_model.load_state_dict(model_dict)
                #         log.info(f'Fine tuned model loaded')
                #     else:
                #         gradient_clipping_value = cfg.training_pipeline.schema.get('gradient_clipping_value', None)
                #
                #         loss_f = hydra.utils.instantiate(cfg.method.loss, model=comm_model)
                #
                #         optimizer = hydra.utils.instantiate(cfg.training_pipeline.optimizer,
                #                                             params=comm_model.parameters())
                #
                #         scheduler = None
                #         if 'scheduler' in cfg:
                #             scheduler = hydra.utils.instantiate(cfg.training_pipeline.scheduler,
                #                                                 optimizer=optimizer)
                #
                #         bar = tqdm.tqdm(range(experiment_cfg.epochs),
                #                         leave=False, desc='Fine tuning the model')
                #         epoch_losses = []
                #
                #         score = -1
                #
                #         for epoch in bar:
                #             comm_model.train()
                #             for x, y in train_dataloader:
                #                 x, y = x.to(device), y.to(device)
                #
                #                 pred = comm_model(x)
                #                 loss = loss_f(pred, y)
                #
                #                 epoch_losses.append(loss.item())
                #
                #                 optimizer.zero_grad()
                #                 loss.backward()
                #
                #                 if gradient_clipping_value is not None:
                #                     torch.nn.utils.clip_grad_norm_(comm_model.parameters(), gradient_clipping_value)
                #
                #                 optimizer.step()
                #
                #             if scheduler is not None:
                #                 scheduler.step()
                #
                #             log.info(f'Training epoch {epoch} ended')
                #
                #             with torch.no_grad():
                #                 comm_model.eval()
                #
                #                 if (epoch + 1) % 5 == 0:
                #                     for a in [0.1, 0.2, 0.3, 0.5, 0.6, 0.8]:
                #
                #                         c, t = 0, 0
                #                         average_dropping = defaultdict(float)
                #
                #                         for x, y in DataLoader(test_dataset, batch_size=1):
                #                             x, y = x.to(device), y.to(device)
                #
                #                             pred = comm_model(x, alpha=a)
                #
                #                             c += (pred.argmax(-1) == y).sum().item()
                #                             t += len(x)
                #
                #                             for i, b in enumerate(
                #                                     [b for b in comm_model.blocks if isinstance(b, AdaptiveBlock) if
                #                                      b.last_mask is not None]):
                #                                 average_dropping[i] += b.last_mask.shape[1]
                #
                #                         log.info(f'Model budget {a} has scores: {c}, {t}, ({c / t})')
                #                         v = {k: v / t for k, v in average_dropping.items()}
                #                         log.info(f'Model budget {a} has average scoring: {v}')
                #                 else:
                #                     t, c = 0, 0
                #
                #                     for x, y in test_dataloader:
                #                         x, y = x.to(device), y.to(device)
                #
                #                         pred = comm_model(x)
                #                         c += (pred.argmax(-1) == y).sum().item()
                #                         t += len(x)
                #
                #                     score = c / t
                #
                #             bar.set_postfix({'Test acc': score, 'Epoch loss': np.mean(epoch_losses)})
                #
                #         torch.save(comm_model.state_dict(), os.path.join(experiment_path, 'fine_tuned_model.pt'))

                comm_model_path = os.path.join(comm_experiment_path, 'comm_model.pt')
                # os.makedirs(comm_model_path, exist_ok=True)

                encoder = hydra.utils.instantiate(experiment_cfg.encoder, input_size=comm_model.num_features)
                decoder = hydra.utils.instantiate(experiment_cfg.decoder, input_size=encoder.output_size,
                                                  output_size=comm_model.num_features)

                channel = experiment_cfg.get('channel', None)

                if channel is not None:
                    channel = hydra.utils.instantiate(channel)

                communication_pipeline = CommunicationPipeline(channel=channel, encoder=encoder, decoder=decoder).to(
                    device)

                if blocks_after is not None:
                    comm_model.blocks = nn.Sequential(*blocks_before, communication_pipeline, *blocks_after)
                else:
                    comm_model.blocks = nn.Sequential(*blocks_before, communication_pipeline)

                if os.path.exists(comm_model_path):
                    model_dict = torch.load(comm_model_path, map_location=device)
                    comm_model.load_state_dict(model_dict)
                    # log.info(model_dict.keys())
                    log.info(f'Comm model loaded')
                else:
                    gradient_clipping_value = cfg.training_pipeline.schema.get('gradient_clipping_value', None)

                    loss_f = nn.CrossEntropyLoss()

                    if experiment_cfg.get('freeze_model', False):
                        for p in blocks_before.parameters():
                            if p.requires_grad:
                                p.requires_grad_(False)

                    optimizer = hydra.utils.instantiate(cfg.training_pipeline.optimizer,
                                                        params=comm_model.parameters())

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
                        communication_pipeline.train()
                        if blocks_after is not None:
                            blocks_after.train()

                        for x, y in train_dataloader:
                            x, y = x.to(device), y.to(device)

                            pred = comm_model(x)
                            loss = loss_f(pred, y)

                            epoch_losses.append(loss.item())

                            optimizer.zero_grad()
                            loss.backward()

                            if gradient_clipping_value is not None:
                                torch.nn.utils.clip_grad_norm_(comm_model.parameters(), gradient_clipping_value)

                            optimizer.step()

                        if scheduler is not None:
                            scheduler.step()

                        # log.info(f'Training epoch {epoch} ended')

                        with torch.no_grad():
                            communication_pipeline.eval()
                            if blocks_after is not None:
                                blocks_after.eval()

                            # if (epoch + 1) % 5 == 0 and False:
                            #     pass
                            #     # for a in [0.1, 0.2, 0.3, 0.5, 0.6, 0.8]:
                            #     #
                            #     #     c, t = 0, 0
                            #     #     average_dropping = defaultdict(float)
                            #     #
                            #     #     for x, y in DataLoader(test_dataset, batch_size=1):
                            #     #         x, y = x.to(device), y.to(device)
                            #     #
                            #     #         pred = model(x, alpha=a)
                            #     #
                            #     #         c += (pred.argmax(-1) == y).sum().item()
                            #     #         t += len(x)
                            #     #
                            #     #         for i, b in enumerate(
                            #     #                 [b for b in model.blocks if isinstance(b, AdaptiveBlock) if
                            #     #                  b.last_mask is not None]):
                            #     #             average_dropping[i] += b.last_mask.shape[1]
                            #     #
                            #     #     log.info(f'Model budget {a} has scores: {c}, {t}, ({c / t})')
                            #     #     v = {k: v / t for k, v in average_dropping.items()}
                            #     #     log.info(f'Model budget {a} has average scoring: {v}')
                            # else:
                            #     if channel is not None and (epoch + 1) % 5 == 0:
                            #         for snr in np.linspace(-10, 10, 20, dtype=int):
                            #             channel.test_snr = snr
                            #
                            #             t, c = 0, 0
                            #
                            #             for x, y in test_dataloader:
                            #                 x, y = x.to(device), y.to(device)
                            #
                            #                 pred = model(x)
                            #                 c += (pred.argmax(-1) == y).sum().item()
                            #                 t += len(x)
                            #
                            #             score = c / t
                            #
                            #             log.info(f'SNR {snr}: {score}')
                            #     else:
                            t, c = 0, 0

                            for x, y in test_dataloader:
                                x, y = x.to(device), y.to(device)

                                pred = comm_model(x)
                                c += (pred.argmax(-1) == y).sum().item()
                                t += len(x)

                            score = c / t

                            bar.set_postfix({'Test acc': score, 'Epoch loss': np.mean(epoch_losses)})

                    torch.save(comm_model.state_dict(), comm_model_path)

                final_evaluation = cfg.get('comm_evaluation', {})
                if final_evaluation is None:
                    final_evaluation = {}

                for key, value in final_evaluation.items():
                    overwrite = value.get('overwrite', False)

                    if not os.path.exists(os.path.join(comm_experiment_path, f'{key}.json')) or overwrite:

                        with warnings.catch_warnings(action="ignore"):
                            results = hydra.utils.instantiate(value, dataset=test_dataset, model=comm_model)

                        if results is not None:
                            with open(os.path.join(comm_experiment_path, f'{key}.json'), 'w') as f:
                                json.dump(results, f, ensure_ascii=True, indent=4)

                        log.info(f'{key} evaluation ended')


if __name__ == "__main__":
    main()
