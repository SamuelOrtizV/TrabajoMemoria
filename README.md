# Assetto Corsa RL - End to end autonomous driving framework

Reinforcement Learning agent that drives in Assetto Corsa using on-screen vision and basic telemetry.
The agent consumes a history of frames (stacked images) plus simple signals (speed, gear, RPM) and outputs controls to a virtual Xbox 360 gamepad (vgamepad). The pipeline follows TMRL’s architecture (Trainer/Server/RolloutWorker) and currently uses Soft Actor-Critic (SAC). RNN variants are supported.


## ❗ Requirements

- Windows 10/11 (tested; relies on pywin32 and vgamepad)
- Python 3.10
- Optional: NVIDIA GPU with CUDA 12.4 for the pinned PyTorch wheels
- Assetto Corsa (PC) running
- Assetto Corsa in‑game apps (required for the RolloutWorker):
  - Data Logger (for UDP telemetry)
  - IS Add Shortcut Key (to start/restart sessions)


## 📦 Installation

1) Create and activate a virtual environment (recommended):

```bat
python -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies:

```bat
pip install -r requirements.txt
```

Notes:
- The requirements pin `torch`/`torchvision` with CUDA 12.4 and add `--extra-index-url https://download.pytorch.org/whl/cu124`.
  - For CPU-only installs, comment that line and use wheel versions without the `+cu124` suffix.
- vgamepad may require the ViGEm Bus driver on Windows; if the virtual gamepad isn’t detected, check that installation.


## 🧩 Installing the in‑game Data Logger (telemetry)

This repository includes a minimal telemetry app under `dataLogger/`. To enable it in Assetto Corsa:

1) Copy the `dataLogger` folder into your Assetto Corsa Python apps directory, typically:
   - `<your AC install>\apps\python\`
   - Example (Steam default): `C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\python\`
2) Launch the game and enable the telemetry app in the in‑game apps list. It intentionally appears without a name (an empty field) so it doesn’t draw UI on screen.
3) Set the camera to first-person view looking straight forward (no car exterior or interior visible in the frame). The capture module will detect the window and rescale images as configured; the game resolution itself does not matter.

Multi-start positions:
- For two starting points, edit the track to move the pits to the middle of the map and still start from the official start line.
- For three starting points, start from the Hotlap Start and adjust both Hotlap and Pits positions in the track editor accordingly.


## 🗂️ Project structure

```
acrl/
  server.py            # TMRL server (central router/registry)
  trainer.py           # Training loop (SAC) using TorchTrainingOffline
  worker.py            # RolloutWorker running the policy in AC
  modules/
    custom_worker.py   # Custom worker (best-model saving, TB logging)
    custom_models.py   # CNN / CNN+RNN actors & critics, utilities
    custom_algorithms.py # SAC agent (AMP optional)
    environment.py     # rtgym interface to AC (capture + telemetry)
    memories.py        # Replay buffers + network sample compression
    rewards.py         # Reward function and termination logic
    UDP_listener.py    # UDP -> dict telemetry (speed, rpms, gear, ...)
    window_interface.py# mss + win32gui window capture
    util.py            # Simple image visualizer (optional)
  inputs/
    game_control.py
    GameInputs.py
    xbox_controller_emulator.py
    xbox_controller_inputs.py
```

Only the entry points (`server.py`, `trainer.py`, `worker.py`) live at the top-level of `acrl/`. All support modules were reorganized under `acrl/modules`, and inputs remain under `acrl/inputs`.


## ⚙️ Configuration

The project uses `tmrl.config.config_constants` (cfg) and `tmrl.config.config_objects.CONFIG_DICT`.
The main configuration typically resides in `config.json` (TMRL default path) and includes:

- TMRL_CONFIG (key knobs used in code):
  - `HUMAN_WORKER`: true to control with a real controller (expert mode)
  - `USE_RNN`: switch between CNN-only and CNN+RNN actor/critic
  - `IMG_STRIDE`, `IMG_HIST_LEN`: history stride and length for images
  - `ACT_BUF_LEN`: number of past actions fed to the model
  - `ALG`:
    - `ALGORITHM`: "SAC" (Only SAC available for now)
    - `LR_ACTOR`, `LR_CRITIC`, `LR_ENTROPY`, `GAMMA`, `POLYAK`, etc.
  - `VIEW_INPUT_TENSOR`: true to visualize concatenated inputs via OpenCV
- ENV_CONFIG:
  - `FULL_SCREEN`: fullscreen capture vs. cropped client area
  - `RTGYM_CONFIG.time_step_duration`: step duration in seconds
  - `MAX_SPEED`, `MIN_SPEED`: automatic throt/brake helpers in the env
  - `MULTI_START_POSITION`: toggles multi start positions logic in training
- REWARD_CONFIG:
  - Weights and thresholds: `REWARD_CHECKPOINT`, `REWARD_PROGRESS`, `PENALTY_*`, `THRESHOLD_*`, etc.

TMRL networking:
- `SERVER_IP_FOR_WORKER`: Server IP the worker connects to
- `PUBLIC_IP_SERVER` and `PORT`: server bind/public address
- `PASSWORD`, `SECURITY`: credentials and security mode


## 🔌 Telemetry, vision, and control

- Telemetry: `acrl/modules/UDP_listener.py` receives one UDP message (default port 5005) and parses into: `speed`, `rpms`, `gear`, `laps`, `track_position`, `tyres_out`, `car_damage[4]`, `acc_x`, `transmitting`.
- Vision: `acrl/modules/window_interface.py` captures the AC window with mss + win32gui and optionally crops borders.
- Control: `acrl/inputs/xbox_controller_emulator.py` (vgamepad) applies actions `[throttle_brake, steering]` in [-1, 1]. In `HUMAN_WORKER=true`, `HumanActor` reads from the real Xbox controller (`xbox_controller_inputs.py`).

Observation and action shapes:
- obs = [speed (1,), gear (1,), rpm (1,), imgs (HIST x H x W x C)]
- act = np.array([throttle_brake, steering]) in [-1, 1]


## 🤖 Models and algorithms

- Algorithm: SAC (Soft Actor-Critic) only at present. REDQ-SAC is disabled here; see Future work.
- Actors/Critics:
  - CNN with stacked frames (StackedChannelCNN*).
  - CNN+RNN via `PreTrainedCNN` (optionally torchvision) + GRU/LSTM (`CNNRNN*`).
- Offline asynchronous training (TorchTrainingOffline) with `MemoryFull`.
- Network sample compression reduces image history to the last frame per step.


## 🧪 Running

3 processes (can be on different machines):

1) Server (central):
```bat
.venv\Scripts\activate
python -m acrl.server
```

2) Trainer (can use GPU):
```bat
.venv\Scripts\activate
python -m acrl.trainer
```

3) Worker (on the Assetto Corsa machine):
```bat
.venv\Scripts\activate
python -m acrl.worker
```

Make sure:
- `cfg.SERVER_IP_FOR_WORKER` points to the Server IP.
- Assetto Corsa is running and telemetry updates (`transmitting=true`).


## 📈 Logging and checkpoints

- TensorBoard: worker episode logs are stored under `runs/<RUN_NAME>/`.
  - Start TensorBoard:
    ```bat
    tensorboard --logdir runs
    ```
- Actor weights: the worker writes the latest model and keeps periodic history snapshots.
- Best model (test): when the test `episode_reward` record is beaten, it saves a `rec_<reward>.tmod` file in the weights folder.


## 📊 Training parameters and terminology (TMRL)

There are no “epochs” in RL in the classical supervised-learning sense. In TMRL, an epoch is simply the moment when the Trainer checkpoints the training session on disk and pushes training metrics to Weights & Biases (wandb).

- One epoch = a configurable number of rounds.
- One round = a configurable number of training steps.

Relevant config keys (see `cfg.TMRL_CONFIG`):
- `MAX_EPOCHS`: how many epochs to run (i.e., how many checkpoint/logging cycles)
- `ROUNDS_PER_EPOCH`: rounds inside each epoch
- `TRAINING_STEPS_PER_ROUND`: gradient steps per round

Practically, logs are printed at the end of each round and synced to wandb at the end of each epoch (when using `trainer.run_with_wandb()`).


## 🧩 Quick knobs

- Select algorithm (SAC only for now): `cfg.TMRL_CONFIG["ALG"]["ALGORITHM"] = "SAC"`
- Enable RNN: `cfg.TMRL_CONFIG["USE_RNN"] = true`
- Visualize actor input: `cfg.TMRL_CONFIG["VIEW_INPUT_TENSOR"] = true`
- Human (expert) control: `cfg.TMRL_CONFIG["HUMAN_WORKER"] = true`


## 🛠️ Troubleshooting

- Worker idle and `transmitting=false`:
  - Ensure the AC telemetry app is enabled and the UDP port matches (default 5005).
- Virtual gamepad not responding:
  - Check `vgamepad` and the ViGEm Bus driver installation.
- PyTorch CUDA install issues:
  - Verify your GPU/driver supports CUDA 12.4 or switch to CPU-only wheels (remove the extra-index line and `+cu124` suffixes).
- AC window not found:
  - The window interface searches for the title "Assetto Corsa"; adjust if your locale/edition uses a different name.


## 📚 Credits and licenses

- Built on TMRL (TrackMania RL) – MIT License.
- Some agent ideas come from Spinning Up (OpenAI).

## Disclaimer

Not affiliated with Kunos Simulazioni or Assetto Corsa. For educational and research purposes only.

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for the full text.


## 🚧 Future work

- Re-enable and benchmark off-policy improvements over SAC (e.g., REDQ-SAC, DroQ) once stable here.
- Add an option to save replays directly from the environment.
- Improve the AC in‑game apps installation guide with screenshots.
- Add scripts to launch Server/Trainer/Worker with profile-based configs.
