"""
Custom RolloutWorker adapted from TMRL (MIT License).

Base source (modified): https://github.com/trackmania-rl/tmrl/blob/master/tmrl/networking.py

This file customizes the RolloutWorker to:
- Track best test episode reward and save corresponding model weights
- Log episode statistics to TensorBoard
- Optionally visualize input tensors

Copyright (c) TMRL authors
Modifications (c) project authors. Licensed under the MIT License.
"""
import os
import itertools
import datetime

import cv2
import numpy as np
import tmrl.config.config_constants as cfg
from tmrl.networking import RolloutWorker, print_with_timestamp
from torch.utils.tensorboard import SummaryWriter

from modules.memories import MemoryInference
from modules.util import ImageVisualizer


class CustomRolloutWorker(RolloutWorker):
    def __init__(self, *args, **kwargs):
        """Initialize the CustomRolloutWorker.

        This worker extends TMRL's RolloutWorker with best-model saving,
        TensorBoard logging, and optional input visualization.
        """
        super().__init__(*args, **kwargs)
        self.weights_folder = cfg.WEIGHTS_FOLDER
        self.run_name = cfg.RUN_NAME
        # Best test reward record loaded from weight history if available
        self.best_test_reward = self.init_best_test_reward()
        self.weights = None
        # Whether to display the model inputs for debugging/inspection
        self.view_input = cfg.TMRL_CONFIG["VIEW_INPUT_TENSOR"]
        self.visualizer = ImageVisualizer(title="Input tensor visualization")
        log_dir = "runs/" + self.run_name
        train_tag = "train/episode_reward"
        test_tag = "test/episode_reward"
        self.episode_counter_train = self.get_last_step(log_dir, train_tag)
        self.episode_counter_test = self.get_last_step(log_dir, test_tag)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.stride = cfg.TMRL_CONFIG["IMG_STRIDE"]
        self.img_hist_len = cfg.IMG_HIST_LEN
        self.act_buf_len = cfg.ACT_BUF_LEN
        self.infer_memory = MemoryInference(
            capacity=(self.img_hist_len + 1) * self.stride * 2,
            hist_len=self.img_hist_len,
            act_len=self.act_buf_len,
            stride=self.stride,
        )

    def get_last_step(self, log_dir, tag):
        """Return the next episode index from TensorBoard logs for a given tag.

        Scans the latest TensorBoard event file in `log_dir` and retrieves the
        last scalar step for `tag`. If none is found, prompts for a starting
        index. Returns 0 in standalone mode or when no log exists.
        """
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        if not os.path.exists(log_dir) or self.standalone:
            return 0  # No log directory or running standalone

        event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
        if not event_files:
            return 0  # No event files

        event_file = max([os.path.join(log_dir, f) for f in event_files], key=os.path.getctime)
        ea = EventAccumulator(event_file)
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            if events:
                print_with_timestamp(f"Last step for tag '{tag}': {events[-1].step + 1}")
                return events[-1].step + 1  # Next episode
        # If tag not found, ask the user for an episode start index
        episodes_str = input(
            f"Could not load last step for tag '{tag}'. Enter the number of episodes to start from: "
        )
        if episodes_str.isdigit():
            return int(episodes_str)
        else:
            return 0  # Non-numeric input: start from 0

    def init_best_test_reward(self):
        """Find the best recorded test reward from saved weight files.

        Looks for files named like: <run_name>...rec_<reward>.tmod and extracts the
        maximum reward found.
        """
        if hasattr(self, "model_path_history"):
            files = [
                f
                for f in os.listdir(self.weights_folder)
                if "rec" in f and self.run_name in f and f.endswith(".tmod")
            ]
            if files:
                # Extract the reward number from each file name
                rewards = []
                for fname in files:
                    # Example: rec_123.45.tmod
                    try:
                        num_str = fname.split("_")[-1].replace(".tmod", "")
                        reward = float(num_str)
                        rewards.append(reward)
                    except Exception:
                        continue
                if rewards:
                    print_with_timestamp(f"Best test reward found in history: {max(rewards)}")
                    return max(rewards)
        return 0.0

    def act(self, obs, test=False):
        """Select an action, optionally building a strided observation for inference."""
        if self.stride > 1:  # FIXME: May not work well with stride > 3 at 20 Hz
            self.infer_memory.append(obs)

            if len(self.infer_memory) > (self.img_hist_len + 1) * self.stride:
                obs_for_model = self.infer_memory.get_transition()

                action = self.actor.act_(obs_for_model, test=test)

                if self.view_input:
                    self.view_input_tensor(obs_for_model)
            else:
                if self.view_input:
                    self.view_input_tensor(obs)
                action = self.actor.act_(obs, test=test)
            return action
        else:
            if self.view_input:
                self.view_input_tensor(obs)
            # Fallback to the base class action selection
            return super().act(obs, test=test)

    def collect_train_episode(self, max_samples=None):
        """Collect one training episode and log basic stats."""
        self.env.env.env._RealTimeEnvTS__interface.train_mode = True
        super().collect_train_episode(max_samples=max_samples)

        self.infer_memory.clear()

    # Debug: per-episode progress
    # print(f"\nTraining episode progress: {self.env.env.env._RealTimeEnvTS__interface.reward_function.track_position:.3}\n")

        if hasattr(self, "tb_writer") and not self.standalone:
            self.tb_writer.add_scalar(
                "train/episode_reward", self.buffer.stat_train_return, self.episode_counter_train
            )
            self.tb_writer.add_scalar(
                "train/episode_length", self.buffer.stat_train_steps, self.episode_counter_train
            )
            self.tb_writer.add_scalar(
                "train/episode_progress",
                round(self.env.env.env._RealTimeEnvTS__interface.reward_function.track_position, 3),
                self.episode_counter_train,
            )
            self.episode_counter_train += 1

    def run_episode(self, max_samples=None, train=False):
        """
        Run one episode and optionally save model weights on new best test reward.

        Args:
            max_samples (int): Maximum number of samples to collect in the episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): Whether this is a training or test episode. `step` is called with `test=not train`.
        """

        self.env.env.env._RealTimeEnvTS__interface.train_mode = train  # Very nested attribute in the real-time env

        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0

        obs, info = self.reset(collect_samples=False)
        for _ in iterator:
            obs, rew, terminated, truncated, info = self.step(
                obs=obs, test=not train, collect_samples=False
            )
            ret += rew
            steps += 1
            if terminated or truncated:
                break

        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

        self.infer_memory.clear()

    # Debug: per-episode progress
    # print(f"\nTest episode progress: {self.env.env.env._RealTimeEnvTS__interface.reward_function.track_position:.3}")

        if hasattr(self, "tb_writer") and not self.standalone:
            self.tb_writer.add_scalar("test/episode_reward", ret, self.episode_counter_test)
            self.tb_writer.add_scalar("test/episode_length", steps, self.episode_counter_test)
            self.tb_writer.add_scalar(
                "test/episode_progress",
                round(self.env.env.env._RealTimeEnvTS__interface.reward_function.track_position, 3),
                self.episode_counter_test,
            )
            self.episode_counter_test += 1

        # Save model weights if a new best test reward is achieved (test episodes only)
        if not train and ret > self.best_test_reward and not self.standalone:
            self.best_test_reward = ret
            print_with_timestamp(
                f"\n\n\n--------NEW RECORD in test episode: {self.best_test_reward}--------\n\n\n"
            )
            self.save_best_model()
        print("\n")

    def save_best_model(self):
        """Save current model weights when the best reward record is broken."""
        if self.weights is not None:
            path = self.model_path_history + "rec_" + str(self.best_test_reward) + ".tmod"
            with open(path, "wb") as f:
                f.write(self.weights)
            print_with_timestamp(f"Weights saved in: {path}")
        else:
            print("NO WEIGHTS!")

    def update_actor_weights(self, verbose=True, blocking=False):
        """
        Updates the actor with new weights received from the `Server` when available.

        Args:
            verbose (bool): whether to log INFO messages.
            blocking (bool): if True, blocks until a model is received; otherwise, can be a no-op.

        Returns:
            int: number of new actor models received from the Server (the latest is used).
        """
        weights_list = self._RolloutWorker__endpoint.receive_all(blocking=blocking)
        nb_received = len(weights_list)
        if nb_received > 0:
            weights = weights_list[-1]
            self.weights = weights  #
            with open(self.model_path, "wb") as f:
                f.write(weights)
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(
                        self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".tmod", "wb"
                    ) as f:
                        f.write(weights)
                    self._cur_hist_cpt = 0
                    if verbose:
                        print_with_timestamp("model weights saved in history")
            self.actor = self.actor.load(self.model_path, device=self.device)
            if verbose:
                print_with_timestamp("model weights have been updated")
        return nb_received

    def view_input_tensor(self, x, print_data=False):
        """
        Visualize the information passed to the agent using OpenCV.
        Historical images are concatenated horizontally (oldest on the left).
        The title can be optionally printed to console.
        """

        """ print("\n--- [DEBUG] Input types in forward ---")
        for i, item in enumerate(x):
            print(f"x[{i}] type: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}") """

        # Unpack the batch data
        speed, gear, rpm, images, *prev_act = x

        # De-normalize display values (if applicable)
        max_speed = 400
        max_gear = 10
        max_rpm = 20000

        s_val = speed.item() * max_speed
        g_val = gear.item() * max_gear
        r_val = rpm.item() * max_rpm
        acts_vals = [pa for pa in prev_act]

        # Build title string
        acts_str = ", ".join([f"Prev Act{j+1}: {val}" for j, val in enumerate(acts_vals)])
        title_str = f"Speed: {s_val:.0f}, Gear: {g_val:.0f}, RPM: {r_val:.0f}, {acts_str}"
        if print_data:
            print(title_str)

        # Extract images
        img_tensor = images  # shape: (num_imgs, H, W, C)

        # Convert to numpy and uint8 if needed
        imgs = img_tensor
        if imgs.dtype != np.uint8:
            imgs = (imgs * 255).astype(np.uint8) if imgs.max() <= 1.0 else imgs.astype(np.uint8)

        # Convert to list of individual images; each img is (H, W, C)
        img_list = [img for img in imgs]

        # Concatenate horizontally using OpenCV
        concat_img = cv2.hconcat(img_list)  # (H, num_imgs*W, C)

        # Visualization with OpenCV (convert to BGR if needed)
        cv2.imshow("Input tensor visualization", concat_img[..., ::-1])
        cv2.waitKey(1)
