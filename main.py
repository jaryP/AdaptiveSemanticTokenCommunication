import json
import logging
import os
import random
import warnings
from collections import defaultdict

import hydra

import numpy as np
import tqdm.auto as tqdm
from omegaconf import DictConfig, OmegaConf
import torch

from torch.utils.data import DataLoader

from methods.proposal import AdaptiveBlock
from serialization import process_saving_path, get_hash
from utils import get_pretrained_model


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# def process_saving_path(sstring: str) -> str:
#     ll = sorted([s.split('=') for s in sstring.split('__')], key=lambda a: a[0])
#
#     return '__'.join(s[1] for s in ll)

def prova(prova, *, _root_):
    return None

OmegaConf.register_new_resolver("to_hash", get_hash)
OmegaConf.register_new_resolver("method_path", lambda x: x.split('.')[-1] if x is not None else 'null')
OmegaConf.register_new_resolver("process_saving_path", process_saving_path)


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

    print(OmegaConf.to_yaml(cfg.training_pipeline.dataset.train, resolve=True))

    to_download = not os.path.exists(cfg.training_pipeline.dataset.train.root)

    train_dataset = hydra.utils.instantiate(cfg.training_pipeline.dataset.train, download=to_download)
    test_dataset = hydra.utils.instantiate(cfg.training_pipeline.dataset.test)

    training_schema = cfg.training_pipeline.schema
    dev_split = training_schema.get('dev_split', 0.1)

    # TODO: IMPLEMENTARE CONTROLLO SUL FILE DI CONFIG
    #
    # d = OmegaConf.to_container(cfg)
    # del d['device']
    # del d['core']
    # del d['training_pipeline']['schema']['experiments']
    #
    # path_hash = get_hash(d)
    outer_experiment_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # outer_experiment_path = os.path.join(outer_experiment_path, path_hash)

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

        if dev_split is not None:
            idx = np.arange(len(train_dataset))
            np.random.RandomState(0).shuffle(idx)

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

            hash_path = get_hash(pre_cfg)

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

        model = hydra.utils.instantiate(cfg.method.model, model=model)
        # model = SemanticVit(model)

        if 'training_pipeline' in cfg:
            if os.path.exists(model_path):
                model_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(model_dict)
                log.info(f'Model loaded')
            else:
                log.info(f'Training of the model')

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

                for _ in bar:
                    model.train()
                    for x, y in train_dataloader:
                        x, y = x.to(device), y.to(device)

                        pred = model(x)
                        loss = loss_f(pred, y)

                        epoch_losses.append(loss.item())

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

                    model.eval()
                    with torch.no_grad():
                        t, c = 0, 0

                        for x, y in test_dataloader:
                            x, y = x.to(device), y.to(device)

                            pred = model(x)
                            c += (pred.argmax(-1) == y).sum().item()
                            t += len(x)

                    bar.set_postfix({'Test acc': c / t, 'Epoch loss': np.mean(epoch_losses)})

                torch.save(model.state_dict(), model_path)

                # with open(os.path.join(experiment_path, 'training_results.json'),
                #           'w') as f:
                #     json.dump(all_results, f, ensure_ascii=False, indent=4,
                #               cls=NpEncoder)

        model.eval()
        with torch.no_grad():
            t, c = 0, 0

            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                c += (pred.argmax(-1) == y).sum().item()
                t += len(x)


            log.info(f'Model score: {c}, {t}, ({c / t})')

            for a in [0.1, 0.2, 0.3, 0.5, 0.6, 0.8]:

                c, t = 0, 0
                average_dropping = defaultdict(float)

                for x, y in DataLoader(test_dataset, batch_size=1):
                    x, y = x.to(device), y.to(device)

                    pred = model(x, alpha=a)

                    c += (pred.argmax(-1) == y).sum().item()
                    t += len(x)

                    for i, b in enumerate([b for b in model.blocks if isinstance(b, AdaptiveBlock)]):
                        average_dropping[i] += b.last_mask.shape[1]

                        print({k: v / t for k, v in average_dropping.items()})

                log.info(f'Model budget {a} has scores: {c}, {t}, ({c / t})')
                v = {k: v / t for k, v in average_dropping.items()}
                log.info(f'Model budget {a} has average scoring: {v}')

        #     # if isinstance(model, HaltingAlgorithm):
        #     # with eval_mode(model), torch.no_grad():
        #     #     old_metrics = [EvaluateAllExits(model=model),
        #     #                HaltedFlops(model=model),
        #     #                OptimalHalted(model=model),
        #     #                OptimalHaltedFlops(model=model),
        #     #                HaltedAccuracy(model=model)]
        #     #
        #     # for x, y in test_dataloader:
        #     #     x = x.to(device)
        #     #     y = y.to(device)
        #     #
        #     #     d = model(x)
        #     #
        #     #     with halted_inference_context(model, 0.1) as halted_model:
        #     #         d = halted_model(x)
        #     #
        #     #     for m in metrics:
        #     #         m.update(target=y, **d)
        #     #
        #     # for m in metrics:
        #     #     print(m.compute())
        #
        # # def unpack(obj):
        # #     return {str(o): unpack(o) for o in obj.get_items()}
        #
        # def unpack(dic, keys, value):
        #     for key in keys[:-1]:
        #         dic = dic.setdefault(key, {})
        #     dic[keys[-1]] = value
        #
        # def mask_dictionary(d, mask):
        #     for k, v in d.items():
        #         if isinstance(v, dict):
        #             mask_dictionary(v, mask)
        #         elif isinstance(v, torch.Tensor):
        #             v = v[mask]
        #             d[k] = v
        #         elif isinstance(v, np.ndarray):
        #             d[k] = v[mask.cpu().numpy()]
        #
        # if isinstance(model, HaltingAlgorithm):
        #     thresholding_results = {}
        #     with eval_mode(model), torch.no_grad():
        #         testing_protocol = cfg.get('final_evaluation', None)
        #
        #         if testing_protocol is not None:
        #             outer_evaluation_results = {}
        #
        #             for k, d in testing_protocol.items():
        #                 if os.path.exists(os.path.join(evaluation_results, f'{k}.json')):
        #                     continue
        #
        #                 inner_evaluation_results = {}
        #
        #                 parameters = d.get('parameters', None)
        #                 parameters = OmegaConf.to_container(parameters, resolve=True)
        #
        #                 other_parameters = {k: v for k, v in parameters.items() if not isinstance(v, Sequence)}
        #                 parameters = {k: v for k, v in parameters.items() if isinstance(v, Sequence)}
        #
        #                 keys = None
        #                 # all_parameters = product(parameters.values(), repeat=len(parameters))
        #
        #                 halting_dict = d.get('halting_mechanism', None)
        #
        #                 for p in tqdm.tqdm(ParameterGrid(parameters)):
        #                     if keys is None:
        #                         keys = list(p.keys())
        #
        #                     # p.update(other_parameters)
        #                     # pp = deepcopy(0)
        #                     testing_model = hydra.utils.instantiate(halting_dict,
        #                                                             model=model.model,
        #                                                             enable_halting=True,
        #                                                             **p,
        #                                                             **other_parameters)
        #                     # testing_model.enable_halting = True
        #
        #                     metrics = [
        #                         # EvaluateAllExits(model=testing_model),
        #                         HaltedFlops(model=testing_model),
        #                         # OptimalHalted(model=testing_model),
        #                         OptimalHaltedFlops(model=testing_model),
        #                         HaltedAccuracy(model=testing_model)]
        #
        #                     discarded_metrics = None
        #
        #                     for x, y in test_dataloader:
        #                         x = x.to(device)
        #                         y = y.to(device)
        #
        #                         d = testing_model(x)
        #
        #                         if 'drop_mask' in d:
        #                             if discarded_metrics is None:
        #                                 discarded_metrics = [
        #                                     HaltedFlops(model=testing_model),
        #                                     HaltedAccuracy(model=testing_model)]
        #
        #                             mask = d['drop_mask']
        #                             dd = deepcopy(d)
        #                             # d1 = deepcopy(d)
        #                             mask_dictionary(dd, mask)
        #
        #                             for m in discarded_metrics:
        #                                 m.update(target=y[mask], **dd)
        #
        #                             mask_dictionary(d, ~mask)
        #
        #                             for m in metrics:
        #                                 m.update(target=y[~mask], **d)
        #                         else:
        #                             for m in metrics:
        #                                 m.update(target=y, **d)
        #
        #                     metrics_results = {}
        #
        #                     if discarded_metrics is not None:
        #                         for m in discarded_metrics:
        #                             metrics_results['discarded_' + str(m)] = m.compute()
        #                             m.reset()
        #
        #                         for m in metrics:
        #                             metrics_results[str(m)] = m.compute()
        #                             m.reset()
        #                     else:
        #                         for m in metrics:
        #                             metrics_results[str(m)] = m.compute()
        #                             m.reset()
        #
        #                     # unpack(inner_evaluation_results, list(p.values()), metrics_results)
        #                     #
        #                     # unpack(inner_evaluation_results, list(p.values()), metrics_results)
        #                     inner_evaluation_results[str(tuple(p.values()))] = metrics_results
        #
        #                 inner_evaluation_results = {'parameters': keys, 'results': inner_evaluation_results}
        #                 outer_evaluation_results[k] = inner_evaluation_results
        #
        #                 with open(os.path.join(evaluation_results, f'{k}.json'), 'w') as f:
        #                     json.dump(inner_evaluation_results, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        #
        #             with open(os.path.join(experiment_path, f'testing_results.json'), 'w') as f:
        #                 json.dump(outer_evaluation_results, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        #
        # if model.model.__class__.__name__ == CumulativeScoreModel.__name__ and dev_dataset is not None:
        #     def_dataloader = datalaoder_wrapper(dev_dataset)
        #
        #     bins_scores = defaultdict(lambda: defaultdict(float))
        #     bins_counters = defaultdict(float)
        #
        #     with eval_mode(model), torch.no_grad():
        #         for x, y in def_dataloader:
        #             x, y = x.to(device), y.to(device)
        #
        #             output = model(x)
        #
        #             prior = output['prior']
        #             exits = {k: v.argmax(-1) for k, v in output[Dict_Exits_Key].items()}
        #             final_exit = output[Final_Exit_Key].argmax(-1)
        #
        #             # prior = torch.softmax(prior, -1)
        #             entropy = (-(prior * prior.log()).sum(-1) / math.log(prior.shape[-1])).cpu().numpy()
        #
        #             bins = np.digitize(entropy, bins=np.linspace(0, 1, 10, endpoint=True), right=True)
        #
        #             for h in np.unique(bins):
        #                 h = int(h)
        #                 mask = bins == h
        #                 bins_counters[h] += mask.sum().item()
        #
        #                 for ee, v in exits.items():
        #                     bins_scores[h][ee] += (v[mask] == y[mask]).sum().item()
        #
        #                 bins_scores[h][Final_Exit_Key] += (final_exit[mask] == y[mask]).sum().item()
        #
        #         results = defaultdict(lambda: defaultdict(float))
        #
        #         for k in bins_counters:
        #             for v in bins_scores[k]:
        #                 results[k][v] = bins_scores[k][v] / bins_counters[k]
        #
        #             a = 0
        #             # bins = torch.histogram(entropy, bins=10, range=(0, 1))
        #
        #         with open(os.path.join(experiment_path, 'entropy.json'), 'w') as f:
        #             json.dump(dict(results), f, ensure_ascii=False, indent=4, cls=NpEncoder)
        #
        #         with open(os.path.join(experiment_path, 'counters.json'), 'w') as f:
        #             json.dump(dict(bins_counters), f, ensure_ascii=False, indent=4, cls=NpEncoder)
        #
        #         with open(os.path.join(experiment_path, 'flops.json'), 'w') as f:
        #             json.dump(model.flops, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        #

if __name__ == "__main__":
    main()
