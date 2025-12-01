# MouseRacing Project

This repository contains a custom Gymnasium environment (`MouseRacing`) and a training/evaluation pipeline using Stable Baselines3 (PPO). The project includes visualization code, environment wrappers, and support for multi-track evaluation and video recording.

Use this README as a quick reference for the files currently in the project and how to run/train/evaluate the agent.

## Files in this project

- `agent3.py` — Main training/evaluation script.
  - Creates vectorized training and evaluation environments using `make_vec_env` and `VecFrameStack`.
  - Instantiates a PPO `CnnPolicy` agent and trains it (default `total_timesteps` in the script).
  - Contains `VideoEvalCallback` to run evaluations and record videos with `VecVideoRecorder`.
  - Includes helper functions: `make_env`, `make_eval_env`, `record_final_video`, and plotting utilities.

- `MouseRacing.py` — The custom Gymnasium environment implementation.
  - Track and tile generation, road/track barriers, rendering (human / `rgb_array` / `state_pixels`).
  - Contact handling (tile visit detection) and reward logic.
  - A small wall-contact penalty was added to discourage scraping the walls.

- `mouse_dynamics.py` — Body and dynamics for the agent (originally wheels; visuals updated to show legs/quadruped).
  - Implements the `Mouse` class: physics bodies, wheels (used for physics), and drawing code.
  - Visuals: wheel polygons replaced or augmented with decorative leg/foot visuals attached to wheel positions.

- `MultiTrackMouseRacing.py` — A wrapper that creates a fresh `MouseRacing` instance per episode using different seeds.
  - Registerd as `MultiTrackMouseRacing-v0` for quick use in `gym.make`.
  - Useful to produce varied track layouts across episodes without modifying the core `MouseRacing` logic.

- `requirements.txt` — Python package dependencies required for development and running the project.
  - Includes `gymnasium`, `stable-baselines3`, `torch`, `pygame`, `box2d-py`, `numpy`, `matplotlib`, `pandas`, `scipy`, `imageio`, `imageio-ffmpeg`, `opencv-python`, and `tensorboard`.

- `GrayWrapper.py` — (optional) observation preprocessing wrapper used by `agent3.py` if present in the repo.
  - Converts `MouseRacing` observations to grayscale/resized frames suitable for CNNs.
  - If you don't have this file, `agent3.py` may still run but will require a compatible wrapper or preprocessing.


## Quick Install (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- On Windows, `torch` sometimes needs a platform-specific wheel. If `pip install torch` fails, follow PyTorch's install instructions at https://pytorch.org/get-started/locally/.
- `gymnasium[box2d]` and `box2d-py` provide Box2D bindings. If installation fails, consider using conda or installing the binary wheel.


## Run training (default)

```powershell
python agent3.py
```

This will run training, periodic evaluation with the configured callbacks, save the best model to `./logs/MouseRacing_v`, and attempt to record evaluation videos to `./logs/MouseRacing_v/videos`.


## Quick smoke test

To quickly verify setup, edit `agent3.py` and set `model.learn(total_timesteps=5000)` (or pass a small number) and run `python agent3.py` to make sure training/eval/video flow works without running a full training job.


## Using `MultiTrackMouseRacing`

Replace `gym.make("MouseRacing-v0", ...)` with:

```python
env = gym.make("MultiTrackMouseRacing-v0", num_tracks=6, render_mode="rgb_array")
```

When used with `make_vec_env` and `VecFrameStack`, this will provide varied tracks across episodes.


## Video names and avoiding overwrite

- `agent3.py`'s `make_eval_env` was updated to create unique recordings; video recorder `name_prefix` includes a timestamp when recording so files are not overwritten. Videos are saved in `./logs/MouseRacing_v/videos`.


## Notes for developers

- Visuals: `mouse_dynamics.py::Mouse.draw()` contains the rendering code (legs/feet). Modify it to change how the agent looks.
- Reward shaping: `MouseRacing.py` implements base rewards and includes penalties for wall contact and missed leaky tiles. Tune `self.reward` adjustments in `FrictionDetector` and `step()` as needed.
- Observation shapes: `agent3.py` uses `make_vec_env` + `VecFrameStack` + `VecTransposeImage` so CNN policies receive `(n_envs, 4, H, W)` shaped observations.


## Troubleshooting

- "Unexpected observation shape": ensure `make_vec_env` + `VecFrameStack` + `VecTransposeImage` are used (the repo's `agent3.py` already does this).
- Video not recorded: check `./logs/MouseRacing_v/videos` and verify `VecVideoRecorder` is wrapped on a vec env with `render_mode='rgb_array'`.
- Box2D install issues: use conda or follow the `mouse_dynamics.py` instructions (install `swig` then `gymnasium[box2d]`).
