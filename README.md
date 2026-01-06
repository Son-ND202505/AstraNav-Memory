# AstraNav-Memory: Contexts Compression for Long Memory

[![Home Page](https://img.shields.io/badge/Project-AstraNav_Memory-blue.svg)](https://astra-amap.github.io/AstraNav-Memory.github.io/)
[![arXiv](https://img.shields.io/badge/Arxiv-2512.21627-b31b1b.svg?logo=arXiv)](https://www.arxiv.org/abs/2512.21627)

## 🔥 Latest News!!
* December 30, 2025: We release the training and inference code.

## Coming Soon
* The inference checkpoint.

## Quickstart
### 🧰 Installation

Clone the repo:

```
git clone https://github.com/amap-cvlab/AstraNav-Memory.git
```

Install Training dependencies:
```
# Ensure torch >= 2.6.0
cd train_code
pip install -r requirements.txt
```

Install habitat-sim and habitat-lab for inference
```
● habitat-sim
git clone https://github.com/facebookresearch/habitat-sim.git && cd habitat-sim && git checkout v0.2.3
pip install -r requirements.txt
python setup.py install --headless
● habitat-lab
git clone https://github.com/chongchong2025/habitat-lab && cd habitat-lab && git checkout v0.2.3_waypoint
python -m pip install -r habitat-baselines/habitat_baselines/rl/requirements.txt
python -m pip install -r habitat-baselines/habitat_baselines/rl/ddppo/requirements.txt
pip install -e .
cd habitat-baselines
pip install -e .
```

### ⚡ Inference
● Goat-Bench
``` python
cd inference_code/hm3d-online
python goat-nav.py
```

● OVON
``` python
cd inference_code/hm3d-online
python ovon-nav.py
```

### ⚡ Training
``` sh
cd train_code
bash run_train.sh
```

## 🏛️ Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@article{ren2025astranav-memory,
    title={AstraNav-Memory: Contexts Compression for Long Memory}, 
    author={Botao Ren and Junjun Hu and Xinda Xue and Minghua Luo and Jintao Chen and Haochen Bai and Liangliang You and Mu Xu},
    year={2025},
    eprint={2512.21627},
}
```

## Acknowledgments
Thanks to [OmniNav](https://github.com/amap-cvlab/OmniNav), [MTU3D](https://github.com/MTU3D/MTU3D), and [OVON](https://github.com/naokiyokoyama/ovon) for open-sourcing the construction of training data and the closed-loop inference code. Their contributions have significantly enriched the open-source community.
