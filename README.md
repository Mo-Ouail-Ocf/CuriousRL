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
python train.py --config configs/doorkey.yaml
```

Other example configs are available in the `configs` directory.

### Project Structure

- `agent/` – PPO agent implementation
- `curiosity/` – curiosity modules and base interfaces
- `envs/` – environment wrappers
- `ppo/` – GAE, rollout buffer, and update logic
- `utils/` – logging and running statistics

### License

This code is provided for research and educational purposes.