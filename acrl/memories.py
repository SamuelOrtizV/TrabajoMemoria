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
    """
    Sample compressor for MemoryTMFull
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        obs, rew, terminated, truncated, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the memory
    the user must define both this function and the append() method of the memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
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
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):

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
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
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

        # Conversion de item (primer indice si stride = 1) a indices de los ultimos elementos
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        # Cargar imágenes y acciones para last_obs y new_obs por separado
        imgs_last_obs = self.load_imgs(idx_last)
        imgs_new_obs = self.load_imgs(idx_now)
        acts_last_obs = self.load_acts(idx_last)
        acts_new_obs = self.load_acts(idx_now)

        # Calcula los índices usados para imgs_last_obs y imgs_new_obs
        all_indices = list(range(idx_now - self.imgs_obs * self.img_stride, idx_now + 1))

        # Busca todos los EOE en la ventana
        eoe_positions = [i for i, idx in enumerate(all_indices) if self.data[4][idx]]
        if eoe_positions:
            eoe_idx = eoe_positions[0]  # Primer EOE encontrado
            dist_to_start = eoe_idx
            dist_to_end = len(all_indices) - 1 - eoe_idx
            if dist_to_start < dist_to_end:
                # Mueve item hacia adelante para dejar el EOE fuera por la izquierda
                item = item + (eoe_idx + 1)
            else:
                # Mueve item hacia atrás para dejar el EOE fuera por la derecha
                item = item - (len(all_indices) - eoe_idx)
            # Si el nuevo item es válido, vuelve a intentar
            if 0 <= item < self.__len__():
                return self.get_transition(item)
            else:
                # Si no es válido, elige uno random como fallback
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
        #res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        indices = [idx_final - i * self.img_stride for i in reversed(range(self.imgs_obs))]
        res = [self.data[3][idx] for idx in indices]
        return np.stack(res)

    def load_acts(self, idx_final):
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
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
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
    def __init__(self, capacity, hist_len=4, act_len =2, stride=1):
        self.buffer = deque(maxlen=capacity)
        self.hist_len = hist_len
        self.act_len = act_len
        self.stride = stride

    def append(self, obs):
        """ Format obs as a tuple (speed, gear, rpm, images, *actions) """

        last_obs = (obs[0],  # speed
                    obs[1],  # gear
                    obs[2],  # rpm
                    obs[3][-1],  # last image
                    obs[-1]) # last action

        self.buffer.append(last_obs)

    def __len__(self):
        return len(self.buffer)
    
    def get_transition(self):
        """Obtiene la transición actual para la inferencia."""
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