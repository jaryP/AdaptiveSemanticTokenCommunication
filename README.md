# Adaptive Semantic Token Communication for Transformer-based Edge Inference

[Alessio Devoto](https://alessiodevoto.github.io/), [Jary Pomponi](https://jarypomponi.com/), [Mattia Merluzzi](https://scholar.google.com/citations?user=jFA-Mp4AAAAJ&hl=en), [Paolo Di Lorenzo](https://sites.google.com/site/paolodilorenzohp/home), [Simone Scardapane](https://www.sscardapane.it/)

This paper presents an adaptive framework for edge inference based on a dynamically configurable transformer-powered deep joint source channel coding (DJSCC) architecture. Motivated by a practical scenario where a resource constrained edge device engages in goal oriented semantic communication, such as selectively transmitting essential features for object detection to an edge server, our approach enables efficient task aware data transmission under varying bandwidth and channel conditions. To achieve this, input data is tokenized into compact high level semantic representations, refined by a transformer, and transmitted over noisy wireless channels. As part of the DJSCC pipeline, we employ a semantic token selection mechanism that adaptively compresses informative features into a user specified number of tokens per sample. These tokens are then further compressed through the JSCC module, enabling a flexible token communication strategy that adjusts both the number of transmitted tokens and their embedding dimensions. We incorporate a resource allocation algorithm based on Lyapunov stochastic optimization to enhance robustness under dynamic network conditions, effectively balancing compression efficiency and task performance. Experimental results demonstrate that our system consistently outperforms existing baselines, highlighting its potential as a strong foundation for AI native semantic communication in edge intelligence applications. 

## Requirements

We suggest creating a [conda](https://conda.io/) environment and installing the packages in the requirements.txt file using pip. 

[//]: # (A suitable [conda]&#40;https://conda.io/&#41; environment named `ldm` can be created)

[//]: # (and activated with:)

[//]: # ()
[//]: # (```)

[//]: # (conda env create -f environment.yaml)

[//]: # (conda activate ldm)

[//]: # (```)

## Code structure

Entry points for our code are `main.py` and `main_opt.py`. The first can be used to train the backbones and the JSCC models, while the latter is used to run the optimization problem (Section V of the paper).

Our adaptive ViT proposal can be find in `methods/proposal/base` file. It acts as a wrapper for a hugging-face ViT model. The folder also contains the regularization loss and the evaluation strategy for our approach.

Instead, `comm` folder contains all the functions for evaluating the baselines (`comm/evaluation`), build the JSCC pipeline (`comm/pipeline`), and creating the AWGN channel (`comm/channel`).   


## Running the experiments

The structure for running all experiments is based on [Hydra](https://pypi.org/project/hydra-core/). Config folder contains all the necessary files to run all the experiments present in the paper. 

To run the experiments we used slurm files, which you can find in the folder `./bash/slurm`. Use them as reference for running yours. You can easily extend this work or test other configurations by adding or modifying config files. 
 
## BibTeX

```
@article{devoto2025adaptive,
  title={Adaptive Semantic Token Communication for Transformer-based Edge Inference},
  author={Devoto, Alessio and Pomponi, Jary and Merluzzi, Mattia and Di Lorenzo, Paolo and Scardapane, Simone},
  journal={arXiv preprint arXiv:2505.17604},
  year={2025}
}
```