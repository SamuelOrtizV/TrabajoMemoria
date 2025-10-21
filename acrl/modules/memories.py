"""
Custom memories adapted from TMRL (MIT License).

Base source (modified): https://github.com/trackmania-rl/tmrl/tree/master/tmrl/custom

This module customizes sample compression and replay buffers for local storage,
training, and inference.

Copyright (c) TMRL authors
Modifications (c) project authors. Licensed under the MIT License.
"""

import numpy as np
import random
import tmrl.config.config_constants as cfg

# local imports
from tmrl.memory import TorchMemory


# FUNCTIONS ====================================================


def last_true_in_list(li):
    for i in reversed(range(len(li))):
        if li[i]:
            return i
    return None

# LOCAL BUFFER COMPRESSION ===============================================================

def get_local_buffer_sample_imgs(prev_act, obs, rew, terminated, truncated, info):
    """Sample compressor for image-based observations.

    Args:
        prev_act: action applied to obtain obs in the transition (comes BEFORE obs)
        obs, rew, terminated, truncated, info: outcome of the transition

    Returns a compressed sample tuple stored in local buffers for networking.
    The compressor reduces image history to the last image only to minimize bandwidth.
    """

    prev_act_mod = prev_act

    obs_mod = (obs[0], obs[1], obs[2], obs[3][-1])
    rew_mod = rew
    terminated_mod = terminated
    truncated_mod = truncated
    info_mod = info
    return prev_act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod

# SUPPORTED CUSTOM MEMORIES ============================================================================================

class GenericTorchMemory(TorchMemory):
    """Generic Torch replay memory storing raw lists per field."""
    def __init__(self,
                 memory_size=1e6,
                 batch_size=1,
                 dataset_path="",
                 nb_steps=1,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def append_buffer(self, buffer):
        """Append a local buffer of compressed samples to the global memory."""

        # parse:
        d0 = [b[0] for b in buffer.memory]  # actions
        d1 = [b[1] for b in buffer.memory]  # observations
        d2 = [b[2] for b in buffer.memory]  # rewards
        d3 = [b[3] for b in buffer.memory]  # terminated
        d4 = [b[4] for b in buffer.memory]  # truncated
        d5 = [b[5] for b in buffer.memory]  # info
        d6 = [b[3] or b[4] for b in buffer.memory]  # done

        # append:
        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)

        # trim
        to_trim = int(self.__len__() - self.memory_size)
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]

    def __len__(self):
        """Number of valid transitions available."""
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):
        """Get a valid (last_obs, new_act, rew, new_obs, terminated, truncated, info) transition.

        Skips invalid transitions that cross end-of-episode boundaries.
        """

        # This is a hack to avoid invalid transitions from terminal to initial
        # TODO: find a way to only index valid transitions instead
        while self.data[6][item]:
            item = random.randint(a=0, b=self.__len__() - 1)

        idx_last = item
        idx_now = item + 1

        last_obs = self.data[1][idx_last]
        new_act = self.data[0][idx_now]
        rew = self.data[2][idx_now]
        new_obs = self.data[1][idx_now]
        terminated = self.data[3][idx_now]
        truncated = self.data[4][idx_now]
        info = self.data[5][idx_now]

        return last_obs, new_act, rew, new_obs, terminated, truncated, info


class MemoryEnv(TorchMemory):
    """Base environment memory handling images with history and action buffers."""
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=2,
                 nb_steps=1,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        self.img_stride = cfg.TMRL_CONFIG["IMG_STRIDE"]

        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        #self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_imgs_offset = max(0, self.min_samples - (self.imgs_obs * self.img_stride))
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def append_buffer(self, buffer):
        raise NotImplementedError

    def __len__(self):
        """Number of valid transitions accounting for history/stride offsets."""
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):
        raise NotImplementedError
    
class MemoryFull(MemoryEnv):
    def get_transition(self, item):
        """Return one transition with image history and action buffers.

        Notes:
        - item is the first index of the images in the history of the OLD observation
        - in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
          i.e. the observation is what step returned after being fed act (and preprocessed)
          therefore, in the RTRL setting, act is appended to obs
        - keeps info dict for CRC debugging
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:  # if first item of the buffer
                item += 1
            elif item == self.__len__() - 1:  # if last item of the buffer
                item -= 1
            elif random.random() < 0.5:  # otherwise, sample randomly
                item += 1
            else:
                item -= 1

        # Convert item to indices of the last elements of old/new observations
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        # Load images and actions for last_obs and new_obs separately
        imgs_last_obs = self.load_imgs(idx_last)
        imgs_new_obs = self.load_imgs(idx_now)
        acts_last_obs = self.load_acts(idx_last)
        acts_new_obs = self.load_acts(idx_now)

        # Compute indices window used for images
        all_indices = list(range(idx_now - self.imgs_obs * self.img_stride, idx_now + 1))

        # Search EOE (end-of-episode) within the window
        eoe_positions = [i for i, idx in enumerate(all_indices) if self.data[4][idx]]
        if eoe_positions:
            eoe_idx = eoe_positions[0]
            dist_to_start = eoe_idx
            dist_to_end = len(all_indices) - 1 - eoe_idx
            if dist_to_start < dist_to_end:
                # Move item forward to exclude EOE on the left
                item = item + (eoe_idx + 1)
            else:
                # Move item backward to exclude EOE on the right
                item = item - (len(all_indices) - eoe_idx)
            # Retry if new item is valid
            if 0 <= item < self.__len__():
                return self.get_transition(item)
            else:
                # Fallback: choose a random valid item
                new_item = random.randint(0, self.__len__() - 1)
                return self.get_transition(new_item)

        last_obs = (self.data[2][idx_last], self.data[7][idx_last], self.data[8][idx_last], imgs_last_obs, *acts_last_obs)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], self.data[7][idx_now], self.data[8][idx_now], imgs_new_obs, *acts_new_obs)
        terminated = self.data[9][idx_now]
        truncated = self.data[10][idx_now]
        info = self.data[6][idx_now]

        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, idx_final):
        # Compute indices to retrieve past images with stride
        indices = [idx_final - i * self.img_stride for i in reversed(range(self.imgs_obs))]
        res = [self.data[3][idx] for idx in indices]
        return np.stack(res)

    def load_acts(self, idx_final):
        """Load past actions, averaging within stride windows if stride > 1."""
        acts = []
        for i in reversed(range(self.act_buf_len)):
            idx = idx_final - i * self.img_stride
            if self.img_stride == 1:
                acts.append(self.data[1][idx])
            else:
                start = idx - self.img_stride + 1 if idx - self.img_stride + 1 >= 0 else 0
                idxs = [j for j in range(start, idx + 1)]
                vals = [self.data[1][j] for j in idxs]
                acts.append(np.mean(vals, axis=0) if vals else self.data[1][idx])
        return acts

    def append_buffer(self, buffer):
        """Append a list of samples to the memory.

        buffer items are tuples: (act, obs, rew, terminated, truncated, info)
        Keeps the info dictionary for CRC debugging.
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][3] for b in buffer.memory]  # images
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[1][1] for b in buffer.memory]  # gears
        d8 = [b[1][2] for b in buffer.memory]  # rpms
        d9 = [b[3] for b in buffer.memory]  # terminated
        d10 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
            self.data[10] += d10
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)
            self.data.append(d10)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]

        return self
    
# INFERENCE BUFFER =================================================================

from collections import deque

class MemoryInference:
    """Lightweight buffer to build strided inference observations on the fly."""
    def __init__(self, capacity, hist_len=4, act_len =2, stride=1):
        self.buffer = deque(maxlen=capacity)
        self.hist_len = hist_len
        self.act_len = act_len
        self.stride = stride

    def append(self, obs):
        """Format obs as a tuple (speed, gear, rpm, last_image, last_action)."""

        last_obs = (obs[0],  # speed
                    obs[1],  # gear
                    obs[2],  # rpm
                    obs[3][-1],  # last image
                    obs[-1]) # last action

        self.buffer.append(last_obs)

    def __len__(self):
        return len(self.buffer)
    
    def get_transition(self):
        """Build the current transition for inference with strided history."""
        last_idx = len(self.buffer) - 1

        imgs_indices = [last_idx - i * self.stride for i in reversed(range(self.hist_len))]
        #acts_indices = [last_idx - i for i in reversed(range(self.act_len))]
        acts_indices = [last_idx - i * self.stride for i in reversed(range(self.act_len))]

        imgs = [self.buffer[idx][3] for idx in imgs_indices]
        imgs = np.stack(imgs)

        # Acciones: promediamos los valores intermedios entre cada stride
        actions = []

        for idx in acts_indices:
            if self.stride == 1:
                actions.append(self.buffer[idx][4])
            else:
                start = idx - self.stride + 1 if idx - self.stride + 1 >= 0 else 0
                idxs = [i for i in range(start, idx + 1)]
                acts = [self.buffer[i][4] for i in idxs]
                actions.append(np.mean(acts, axis=0) if acts else self.buffer[idx][4])

        last_obs = (self.buffer[last_idx][0],
                    self.buffer[last_idx][1],
                    self.buffer[last_idx][2],
                    imgs,
                    *actions)

        return last_obs

    def clear(self):
        """Clear the inference memory."""
        self.buffer.clear()