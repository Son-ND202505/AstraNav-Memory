# MuJoCo G1 object-goal navigation with AstraNav-Memory

AstraNav-Memory drives a Unitree G1 humanoid through two furnished MuJoCo houses (`home25`, 25 x 25 m; `l28`, L-shaped). At every stop the robot turns
360 degrees, the model sees the new views plus its image history and answers "found at (x, z)" or "go to frontier (x, z)"; a controller walks there.
Goals of the same scene are chained and the memory (image history, explored map) is kept, as in the original algorithm.

## Layout

```
workspace/
├── OVSegDT/                          the OVSegDT fork: provides mujoco_g1_nav (houses, robot, evaluation) and third_party/
└── AstraNav-Memory/                  this repository
    ├── mujoco_g1_nav -> ../OVSegDT/mujoco_g1_nav      (symlink, step 1)
    ├── train_code/transformers-4.57.1  the transformers version the model needs (original code)
    └── astranav_bench/
        ├── scripts/play_house_astranav.py   the demo
        ├── server/                          the model server (original weights / with the fine-tune)
        └── src/astranav_adapter/            camera, prompt, memory bank, frontiers, controllers
```

## 1. Clone

```bash
mkdir workspace && cd workspace
git clone https://github.com/Son-ND202505/OVSegDT.git        # then steps 1-2 (MuJoCo side only) of OVSegDT/mujoco_g1_nav/README.md
git clone https://github.com/Son-ND202505/AstraNav-Memory.git
cd AstraNav-Memory && ln -s ../OVSegDT/mujoco_g1_nav mujoco_g1_nav
```

(Instead of the symlink: `export G1NAV_ROOT=/path/to/OVSegDT/mujoco_g1_nav`.) The OVSegDT policy environment is not needed here.

## 2. Python

The MuJoCo environment of the OVSegDT fork (`g1nav`, Python 3.11) runs both the demo and the model server. Add:

```bash
conda activate g1nav
pip install -r astranav_bench/requirements.txt
pip install --target astranav_bench/pylibs huggingface_hub==0.36.2 tokenizers==0.22.2     # for the vendored transformers 4.57.1
```

## 3. Weights

```bash
bash astranav_bench/scripts/download_weights.sh
#   astranav_bench/weights/ds16_context100     original AstraNav-Memory weights (ModelScope, ~7.4 GB)
#   astranav_bench/weights/astranav_g1_ft.pt   fine-tune for the MuJoCo houses
```

## 4. Run (from this repo's root)

```bash
python astranav_bench/scripts/play_house_astranav.py                     # fine-tuned model, home25
python astranav_bench/scripts/play_house_astranav.py --house l28
python astranav_bench/scripts/play_house_astranav.py --base              # the original weights
python astranav_bench/scripts/play_house_astranav.py --no-viewer --speed 0 --cmd "sofa" "bed"
```

The model server starts with the demo (one CUDA GPU, about 9 GB, ~40 s to load). Every decision of the model is printed in the terminal; a second window shows
what the model sees. Type `help` for the commands.

## License

`astranav_bench/` is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The rest of this repository belongs to the AstraNav-Memory authors; the weights
follow the licenses of the models they derive from. If you use it, please cite AstraNav-Memory (citation in the original README below).
