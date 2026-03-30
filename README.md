## Curiosity-Based Reinforcement Learning

This repository contains a PyTorch implementation of curiosity-driven reinforcement learning agents (ICM, RND, NovelD) trained with PPO on MiniGrid-style environments.

### Features

- Curiosity modules: ICM, RND, NovelD (pluggable via a factory interface)
- PPO with GAE and rollout storage
- MiniGrid environment wrappers and configuration-driven experiments
- Logging with running statistics for rewards and intrinsic bonuses

### Installation

```bash
pip install -r requirements.txt
```

### Training

Run training with a chosen configuration file, for example:

```bash
# Vanilla PPO-LSTM (Gate 1 baseline)
python train.py

# PPO-LSTM + ICM
python train.py curiosity.method=icm

# PPO-LSTM + RND
python train.py curiosity.method=rnd

# PPO-LSTM + NovelD
python train.py curiosity.method=noveld
```

### Switch environments

```bash
# MultiRoom
python train.py env=multiroom curiosity.method=noveld

# ObstructedMaze (v1 only)
python train.py env=obstructed_maze curiosity.method=noveld
```

### Override any hyperparameter inline

```bash
# Different seed
python train.py run.seed=1

# More envs, longer rollouts
python train.py env.n_envs=32 ppo.n_steps=256

# Enable W&B
python train.py run.use_wandb=true run.wandb_project=my-project

# Full combo
python train.py env=multiroom curiosity.method=noveld run.seed=2 ppo.total_timesteps=5_000_000
```

---

## Monitor training

```bash
tensorboard --logdir runs/
```

Then open `http://localhost:6006`. Key charts to watch for the success gates:

| Gate | What to check |
|------|--------------|
| Gate 1 | `charts/success_rate > 0.7` within 2M steps, DoorKey vanilla |
| Gate 2 | All 3 curiosity methods above vanilla on DoorKey |
| Gate 3 | NovelD clearly above RND on MultiRoom |
| Gate 4 | Any curiosity gets nonzero return on ObstructedMaze within 10M steps |

---

## Reproducing the full benchmark (4 methods × 3 envs)

```bash
for method in none icm rnd noveld; do
  for env in doorkey multiroom obstructed_maze; do
    python train.py env=$env curiosity.method=$method run.seed=42 &
  done
done
```

Drop the `&` if you don't have multiple GPUs and want to run sequentially.

---

## Checkpoints

Saved every 100 updates to `runs/<run_name>/checkpoints/`. Each checkpoint contains model weights, optimizer state, and normalizer stats (RND obs normalizer + reward RunningMeanStd), so runs can be resumed cleanly. To resume, load with:

```python
from train import load_checkpoint
step = load_checkpoint("runs/.../checkpoints/ckpt_1000000.pt", agent, optimizer, curiosity_module)
```

# Recommended Lambda Values
```yaml
# configs/doorkey.yaml
curiosity:
  lambda_int: 0.5   # easier env, extrinsic signal less sparse

# configs/multiroom.yaml
curiosity:
  lambda_int: 1.0   # moderate sparsity, needs more intrinsic push

# configs/obstructed_maze.yaml
curiosity:
  lambda_int: 0.1   # very hard env — high lambda causes pure novelty-seeking
```


### Project Structure

- `agent/` – PPO agent implementation
- `curiosity/` – curiosity modules and base interfaces
- `envs/` – environment wrappers
- `ppo/` – GAE, rollout buffer, and update logic
- `utils/` – logging and running statistics

### License

This code is provided for research and educational purposes.

