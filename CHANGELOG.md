# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-10-21

### Highlights
- First public, end-to-end pipeline and testbed for autonomous learning and autonomous racing in Assetto Corsa.
- Distributed training architecture (server / trainer / worker) with real-time perception-to-control loop.
- Soft Actor-Critic (SAC) with CNN and optional CNN+RNN architectures; HumanActor for expert baseline.

### Features
- Real-time integration with Assetto Corsa:
  - On-screen capture (MSS + Win32) and UDP telemetry (minimal in-game Data Logger app included).
  - Control via virtual Xbox 360 gamepad (vgamepad).
- Algorithms and models:
  - SAC as the primary algorithm.
  - CNN and CNN+RNN actor/critic variants.
  - Optional visualization of actor input tensor.
- Training and logging:
  - Offline asynchronous training (memory + trainer) with periodic checkpoints.
  - TensorBoard logging; optional Weights & Biases integration from the trainer.
  - Best-model saving on new test reward record.
- Environment and utilities:
  - Reward and termination logic modularized.
  - Multi-start positions support for training diversity.
  - Sample compression for image histories in replay buffers.

### Repository structure and tooling
- Entry points: `acrl/server.py`, `acrl/trainer.py`, `acrl/worker.py`.
- Modules moved under `acrl/modules/` (environment, models, algorithms, worker customization, memories, rewards, window/UDP interfaces, utils).
- Inputs kept under `acrl/inputs/` (controller emulation and real controller input).

### Documentation
- Full README: installation, in-game Data Logger setup, configuration, running, troubleshooting.
- Added training terminology section clarifying epochs/rounds/steps.
- Added MIT LICENSE and a clear non-affiliation disclaimer.

### Infrastructure and housekeeping
- requirements.txt normalized and pinned (including CUDA 12.4 wheels for torch/torchvision).
- `.gitignore` fixed to ignore the top-level `dummytest/` sandbox; previously tracked files untracked.

### Breaking changes
- Internal modules reorganized under `acrl/modules/`.
  - If you had custom code importing internal modules directly, update imports to the new paths (e.g., `from acrl.modules.environment import ...`).

### Known limitations
- Windows-only (relies on Win32 window capture and vgamepad).
- Requires Assetto Corsa (PC) and enabling the included Data Logger app.
- Only SAC is enabled in this release; REDQ/DroQ are planned as future work.
- No built-in replay saving yet (planned).

### Upgrade notes
- Review the README for environment setup and configuration keys.
- Ensure the in-game Data Logger app is installed and enabled.
- Set `SERVER_IP_FOR_WORKER` to your server host before starting the worker.

### Disclaimer
Not affiliated with Kunos Simulazioni or Assetto Corsa. For educational and research purposes only.