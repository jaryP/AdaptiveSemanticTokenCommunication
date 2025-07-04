import collections
import json
import logging
import os
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from ast import literal_eval

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
from comm.evaluation import digital_jpeg, digital_resize, analog_resize
from methods import SemanticVit
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

    outer_experiment_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    log.info(f'Saving path: {outer_experiment_path}')
    training_schema = cfg.training_pipeline.schema

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

        final_evaluation = cfg.get('final_evaluation', {})
        if final_evaluation is None:
            final_evaluation = {}

        log.info(f'Final evaluation')

        eval_results = {}

        for key, value in final_evaluation.items():
            if os.path.exists(os.path.join(evaluation_results, f'{key}.json')):
                with open(os.path.join(evaluation_results, f'{key}.json'), 'r') as f:
                    eval_results[key] = json.load(f)

        if cfg.get('jscc', None) is not None:

            average_results = defaultdict(lambda: defaultdict(dict))
            models = {}

            for experiment_key, experiment_cfg in cfg['jscc'].items():
                compression = experiment_cfg.encoder.output_size
                use_for_opt_problem = experiment_cfg.get('use_for_opt_problem', False)
                comm_experiment_path = os.path.join(experiment_path, experiment_key)

                log.info(f'Comm experiment called {experiment_key}')

                # if use_for_opt_problem:
                #     models[str(compression)] = deepcopy(comm_model).eval().cpu()

                final_evaluation = cfg.get('comm_evaluation', {})
                if final_evaluation is None:
                    final_evaluation = {}

                keys_to_use = []

                for key, value in final_evaluation.items():

                    if use_for_opt_problem:
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

                                    average_results[snr][k][compression] = {'accuracy': acc, 'kn': size / (224 * 224 * 3)}
                            else:
                                average_results[snr][compression] = {'accuracy': accuracy, 'kn': vals['symbols'] / (224 * 224 * 3)}

            # MINIMIZATION PROBLEM

            # models.eval()

            def z_f(t, zetas, gammas, gamma_avg, mu):
                if t == 0:
                    return 0

                return max(0, zetas[t-1] + mu * (gammas[t-1] - gamma_avg))

            def get_mapping(snr, results):
                def flatten(dictionary, parent_key='', separator='.'):

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
                                items.extend(flatten({str(k): v}, new_key).items())
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
                '10_2.5': {
                    'v': [1, 10, 100, 1000, 10000, 100000],
                    'mu': [1, 10, 100],
                    'kn': [0.005, 0.0025, 0.001, 0.02, 0.05, 0.02, 0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4],
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
                '20_0': {
                    'v': [1, 10, 100, 1000, 10000, 100000],
                    'mu': [1, 10, 100],
                    'kn': [0.005, 0.0025, 0.001, 0.02, 0.05, 0.02, 0.01, 0.1],
                    'T': 10000,
                    'snr_mu': 20,
                    'snr_sigma': 0
                },
                '10_0': {
                    'v': [1, 10, 100, 1000, 10000, 100000],
                    'mu': [1, 10, 100],
                    'kn': [0.005, 0.0025, 0.001, 0.02, 0.05, 0.02, 0.01, 0.1],
                    'T': 10000,
                    'snr_mu': 10,
                    'snr_sigma': 0
                }
            }

            for opt_name, vals in opt_cfg.items():

                snr_f = lambda: 1 * np.random.normal(vals['snr_mu'], vals['snr_sigma'], 1)[0]

                Vs = vals['v']
                Ms = vals['mu']
                KNs = vals['kn']
                T = vals['T']

                opt_results = {}

                # if os.path.exists(os.path.join(opt_results_path, f'{opt_name}.json')):
                #     with open(os.path.join(opt_results_path, f'{opt_name}.json'), 'r') as f:
                #         opt_results = json.load(f)

                for kn in KNs:
                    for V in Vs:
                        for mu in Ms:
                            if str((kn, V, mu)) in opt_results:
                                continue

                            acc = []

                            gamma_s = []
                            zeta_s = []

                            for i in tqdm.tqdm(range(T)):
                                snr = snr_f()
                                z = z_f(i, mu=mu, gamma_avg=kn, gammas=gamma_s, zetas=zeta_s)

                                mapping = get_mapping(snr, results=average_results)

                                min_ac = np.argmin([-V * accuracy + z * kn for _, kn, accuracy in mapping])
                                _, kn_s, accuracy = mapping[min_ac]

                                # print(kn, V, mu, alpha, c, kn_s)
                                acc.append(accuracy)

                                gamma_s.append(kn_s)
                                zeta_s.append(z)

                            opt_results[str((kn, V, mu))] = {'gamma': gamma_s, 'accuracy': acc, 'zeta': zeta_s}

                            with open(os.path.join(opt_results_path, f'{opt_name}.json'), 'w') as f:
                                json.dump(opt_results, f, cls=NpEncoder)

                            print('Gamma', np.mean(gamma_s[-1000:]), kn, V, mu)
                            print('Accuracy', np.mean(acc[-1000:]), ((np.asarray(gamma_s)[-100 + 1:] - np.asarray(gamma_s)[-100:-1]) / np.asarray(gamma_s)[-100 + 1:]).mean())

                            # if (np.asarray(zeta_s)[-100:-1] / np.asarray(zeta_s)[-100 + 1:]).mean():
                            # if np.mean(np.mean(gamma_s[-1000:])) < kn:
                            #     # c, t = 0, 0
                            #     # for x, y in DataLoader(test_dataset, shuffle=True, batch_size=128):
                            #     #     pred = model(x, alpha=float(alpha))
                            #     #
                            #     #     c += (pred.argmax(-1) == y).sum().item()
                            #     #     t += len(x)
                            #     best_values[kn].append((V, mu, np.mean(acc[-1000:])))

                            # if np.mean(np.mean(gamma_s[-1000:])) < kn:
                            #     # c, t = 0, 0
                            #     # for x, y in DataLoader(test_dataset, shuffle=True, batch_size=128):
                            #     #     pred = model(x, alpha=float(alpha))
                            #     #
                            #     #     c += (pred.argmax(-1) == y).sum().item()
                            #     #     t += len(x)
                            #     best_values[kn].append((V, mu, np.mean(acc[-1000:])))

                            continue
                            fig, ax = plt.subplots(1, 1)
                            plt.plot(range(len(zeta_s)), zeta_s)
                            plt.show()

                            # plt.savefig(os.path.join('./opti_plots', f'min_{kn}_{V}_{mu}.png'), bbox_inches='tight')
                            plt.close(fig)

                            fig, ax = plt.subplots(1, 1)
                            plt.plot(range(len(acc)), acc)
                            plt.show()
                            # plt.savefig(os.path.join('./opti_plots', f'acc_{kn}_{V}_{mu}.png'), bbox_inches='tight')
                            plt.close(fig)

                            # if (np.asarray(zeta_s)[-100:-1] / np.asarray(zeta_s)[-100 + 1:]).mean():

                best_values = defaultdict(list)

                for key, value in opt_results.items():
                    kn, V, mu = literal_eval(key)

                    if np.mean(np.mean(value['gamma'][-1000:])) < kn:
                        # c, t = 0, 0
                        # for x, y in DataLoader(test_dataset, shuffle=True, batch_size=128):
                        #     pred = model(x, alpha=float(alpha))
                        #
                        #     c += (pred.argmax(-1) == y).sum().item()
                        #     t += len(x)
                        best_values[kn].append((V, mu, np.mean(value['accuracy'][-1000:])))

                keys = sorted(best_values.keys())
                for k in keys:
                    mx = max(best_values[k], key=lambda x: x[0])
                    print(k, mx)


if __name__ == "__main__":
    main()
