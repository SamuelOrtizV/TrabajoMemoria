"""
Custom models adapted from TMRL (MIT License).

Base source (modified): https://github.com/trackmania-rl/tmrl/tree/master/tmrl/custom

This module provides various actor-critic networks (MLP, CNN, CNN+RNN) and helpers
tailored to this project, derived from TMRL's customizable components.

Copyright (c) TMRL authors
Modifications (c) project authors. Licensed under the MIT License.
"""

# === Trackmania =======================================================================================================


# standard library imports

# third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from math import floor, sqrt
from torch.nn import Conv2d, Module, ModuleList
# import torchvision

# local imports
from tmrl.util import prod
from tmrl.actor import TorchActorModule
import tmrl.config.config_constants as cfg

from inputs.xbox_controller_inputs import XboxControllerReader


# SUPPORTED ============================================================================================================


def view_input_tensor_matplotlib(x):
    """Visualize the information passed down to the agent using Matplotlib.

    This helper unpacks a batched observation tuple and displays the historical
    RGB frames horizontally along with a title summarizing telemetry and previous actions.
    """

    import matplotlib.pyplot as plt
    print("\n--- [DEBUG] Tipos de entrada en forward ---")
    for i, item in enumerate(x):
        print(f"x[{i}] type: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}")         

    # Index of the batch item to visualize
    i = 0

    # Unpack the batch data
    speed, gear, rpm, images, *prev_act = x

    # Seleccionar el i-ésimo elemento de cada uno
    s_val = speed[i].cpu().numpy().item()
    g_val = gear[i].cpu().numpy().item()
    r_val = rpm[i].cpu().numpy().item()
    
    acts_vals = [pa[i].detach().cpu().numpy() for pa in prev_act]

    # Build title string
    acts_str = ", ".join([f"Prev Act{j+1}: {val}" for j, val in enumerate(acts_vals)])
    title_str = f"Speed: {s_val:.2f}, Gear: {g_val:.2f}, RPM: {r_val:.2f}, {acts_str}"

    # Extract images
    img_tensor = images[i]  # shape: (num_imgs, H, W, C)

    print(f"Imagenes tensor shape: {img_tensor.shape}")

    # Number of historical RGB frames
    num_rgb_frames = img_tensor.shape[0]

    # Crear figura
    fig, axs = plt.subplots(1, num_rgb_frames, figsize=(4 * num_rgb_frames, 4))
    fig.suptitle(title_str, fontsize=12)

    # Asegurar iterabilidad
    if num_rgb_frames == 1:
        axs = [axs]

    for j in range(num_rgb_frames):
        rgb = img_tensor[j].cpu().numpy()  # (H, W, C)
        axs[j].imshow(rgb)
        axs[j].set_title(f"Frame {j}")
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()


# Spinup MLP: =======================================================
# Adapted from the SAC implementation of OpenAI Spinning Up


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-7


class SquashedGaussianMLPActor(TorchActorModule):
    """Gaussian policy with Tanh squashing over an MLP encoder.

    Handles both tuple and flat observation spaces and outputs actions in the
    bounds of the action space. Returns the action and optional log-prob.
    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__(observation_space, action_space)
        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self.tuple_obs = False
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit


    def forward(self, obs, test=False, with_logprob=True):
        """Forward pass.

        Args:
            obs: Observation tensor or tuple of tensors.
            test: If True, returns deterministic action (mean).
            with_logprob: If True, also returns log-probabilities.
        """
        x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        # pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        """Convenience method to return a numpy action without grad."""
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class MLPQFunction(nn.Module):
    """Q-function implemented as an MLP over (obs, act)."""
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        try:
            obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
            self.tuple_obs = True
        except TypeError:
            obs_dim = prod(obs_space.shape)
            self.tuple_obs = False
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        """Compute Q(s,a)."""
        x = torch.cat((*obs, act), -1) if self.tuple_obs else torch.cat((torch.flatten(obs, start_dim=1), act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.  # FIXME: understand this


class MLPActorCritic(nn.Module):
    """Actor-Critic wrapper combining a policy and two Q-functions (SAC)."""
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        """Return a numpy action from the actor without grad."""
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


# REDQ MLP: =====================================================


class REDQMLPActorCritic(nn.Module):
    """Actor-Critic for REDQ with a shared actor and N critics."""
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU,
                 n=10):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.n = n
        self.qs = ModuleList([MLPQFunction(observation_space, action_space, hidden_sizes, activation) for _ in range(self.n)])

    def act(self, obs, test=False):
        """Return a numpy action from the actor without grad."""
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()

# Human Actor: =========================================================================================================

class HumanActor(TorchActorModule):
    """Human-controlled actor that reads actions from an Xbox controller."""
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.action_space = action_space
        self.controller = XboxControllerReader(total_wait_secs=7)  # Instancia para leer el controlador Xbox

    def forward(self, obs, test=False, with_logprob=True):
        """Not used in human mode since actions come from the controller."""
        raise NotImplementedError("'forward' is not required for HumanActor with human inputs.")

    def act(self, obs, test=False):
        """Capture Xbox controller inputs and clip them to the action space."""
        # Leer las entradas del controlador
        controller_inputs = self.controller.read()

        # Asegurarse de que la acción esté dentro de los límites del espacio de acciones
        action = self._clip_action(controller_inputs)

        return action

    def _clip_action(self, action):
        """Ensure the action stays within the action space bounds."""
        return np.clip(action, self.action_space.low, self.action_space.high)

# Vanilla CNN: ====================================================================================


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out

def replace_first_conv(model, in_channels):
    """Recursively replace the first Conv2d layer to accept in_channels."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            if module.in_channels != in_channels:
                new_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=module.bias is not None
                )
                setattr(model, name, new_conv)
            return True  # Ya reemplazado
        elif replace_first_conv(module, in_channels):
            return True
    return False

class VanillaCNN(Module):
    """Basic CNN encoder over stacked historical frames (no batch norm)."""
    def __init__(self, action_space_size):
        super(VanillaCNN, self).__init__()
        #self.q_net = q_net
        self.h_out, self.w_out = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
        self.hist_len = cfg.IMG_HIST_LEN
        self.num_channels = 1 if cfg.GRAYSCALE else 3
        self.action_space_size = action_space_size
        self.act_buf_len = cfg.ACT_BUF_LEN
   
        self.conv1 = Conv2d(self.hist_len * self.num_channels, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels
        self.flat_features = self.out_channels * self.h_out * self.w_out  

    def forward(self, x):
        # Normalize to [0,1]
        x = x.float() / 255.0

        x_conv = F.relu(self.conv1(x))
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = F.relu(self.conv3(x_conv))
        x_conv = F.relu(self.conv4(x_conv))

        flat_features = num_flat_features(x_conv)
        #assert flat_features == self.flat_features, f"x.shape:{x_conv.shape}, flat_features:{flat_features}, self.out_channels:{self.out_channels}, self.h_out:{self.h_out}, self.w_out:{self.w_out}"
        x_conv = x_conv.view(-1, flat_features)

        return x_conv
    
class VanillaCNNBN(Module):
    """Basic CNN encoder with BatchNorm over stacked historical frames."""
    def __init__(self, h_out = cfg.IMG_HEIGHT, w_out = cfg.IMG_WIDTH, hist_len=cfg.IMG_HIST_LEN,
                  num_channels=1 if cfg.GRAYSCALE else 3, act_buf_len = cfg.ACT_BUF_LEN):
        super(VanillaCNNBN, self).__init__()
        self.h_out, self.w_out = h_out, w_out
        self.hist_len = hist_len
        self.num_channels = num_channels
        self.act_buf_len = act_buf_len

        self.conv1 = Conv2d(self.hist_len * self.num_channels, 64, 8, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)

        self.conv2 = Conv2d(64, 64, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)

        self.conv3 = Conv2d(64, 128, 4, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)

        self.conv4 = Conv2d(128, 128, 4, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)

        self.out_channels = self.conv4.out_channels
        self.flat_features = self.out_channels * self.h_out * self.w_out  

    def forward(self, x):
        # Normalize to [0,1]
        x = x.float() / 255.0

        x_conv = F.relu(self.bn1(self.conv1(x)))
        x_conv = F.relu(self.bn2(self.conv2(x_conv)))
        x_conv = F.relu(self.bn3(self.conv3(x_conv)))
        x_conv = F.relu(self.bn4(self.conv4(x_conv)))

        flat_features = num_flat_features(x_conv)
        x_conv = x_conv.view(-1, flat_features)

        return x_conv

class StackedChannelCNN(Module):
    """CNN backbone that stacks historical images along channels and concatenates telemetry and past actions.

    If q_net=True, the current action is also concatenated.
    """
    def __init__(self, q_net, action_space_size):
        super(StackedChannelCNN, self).__init__()
        self.q_net = q_net
        self.h_out, self.w_out = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
        self.hist_len = cfg.IMG_HIST_LEN
        self.num_channels = 1 if cfg.GRAYSCALE else 3
        self.action_space_size = action_space_size
        self.act_buf_len = cfg.ACT_BUF_LEN

    # Try to use PreTrainedCNN, fallback to VanillaCNN if it fails
        try:
            self.cnn = PreTrainedCNN(stacked_channels=self.hist_len)
            # Determinar la dimensión de salida de la CNN preentrenada
            with torch.no_grad():
                dummy = torch.zeros(1, self.num_channels*self.hist_len, self.h_out, self.w_out)
                cnn_out = self.cnn(dummy)
                self.flat_features = cnn_out.shape[1]
        except Exception as e:  # TODO: add cfg parameter to toggle pretrained CNN
            print(f"[StackedChannelCNN] Falling back to VanillaCNN by default.")
            self.cnn = VanillaCNNBN()
            self.flat_features = self.cnn.flat_features

        # Calcular las características planas de salida
        #self.flat_features = 256  # Salida de EfficientNet
        self.mlp_input_features =   (self.flat_features + 3 + self.action_space_size*(self.act_buf_len+1)
                                    if self.q_net else
                                    self.flat_features + 3 + self.action_space_size*self.act_buf_len)
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def stack_hist_images(self, images):
        """Reorder images to (batch, hist_len*C, H, W) from (batch, hist_len, H, W, C)."""
        batch_size, hist_len, height, width, channels = images.shape
        images = images.permute(0, 1, 4, 2, 3)  # (batch, hist_len, C, H, W)
        images = images.reshape(batch_size, hist_len * channels, height, width)  # (batch, hist_len*C, H, W)
        return images    

    def forward(self, x):

        #view_input_tensor_matplotlib(x)

        speed, gear, rpm, images, *acts = x
        #images = images.float() / 255.0
        images = self.stack_hist_images(images)
        cnn_out = self.cnn(images)

        if self.q_net:
            prev_acts = acts[:-1]
            act = acts[-1]
            x = torch.cat((speed, gear, rpm, cnn_out, *prev_acts, act), -1)
        else:
            prev_acts = acts
            x = torch.cat((speed, gear, rpm, cnn_out, *prev_acts), -1) 

    # Pass through the MLP
        x = self.mlp(x)
        return x

class StackedChannelCNNActor(TorchActorModule):
    """SAC actor using the StackedChannelCNN as backbone."""
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        self.net = StackedChannelCNN(q_net=False, action_space_size=dim_act)
        
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)

        self.act_limit = act_limit
        self.grayscale = cfg.GRAYSCALE
        self.act_buf_len = cfg.ACT_BUF_LEN

        # Initialize bias so the car starts accelerating and going straight
        desired_action = torch.tensor([0.5, 0.0])  # If action_dim == 2
        unsquashed = torch.atanh(desired_action)  # Inverse of tanh

        with torch.no_grad():
            nn.init.zeros_(self.mu_layer.weight)  # No contribution from the net at init
            self.mu_layer.bias.copy_(unsquashed)

        # Initialize log_std to start with moderate exploration (optional)
        # nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        # nn.init.constant_(self.log_std_layer.bias, -0.5)

        # Debug (can be removed)
        print("mu_layer.bias:", self.mu_layer.bias.data)
        print("log_std_layer.bias:", self.log_std_layer.bias.data)

    def forward(self, obs, test=False, with_logprob=True):
        """Forward pass returning action and optional log-prob."""
        net_out = self.net(obs)           
        
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        #std = F.softplus(log_std) + EPSILON # Alternativa
        std = torch.exp(log_std)
        #std = torch.clamp(std, EPSILON)  # Clamping std to avoid numerical issues
        #std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)  # Reemplaza NaN e infinitos
        
        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # NB: this is from Spinup:
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # FIXME: this formula is mathematically wrong, no idea why it seems to work
            # Whereas SB3 does this:
            #logp_pi -= torch.sum(torch.log(1 - torch.tanh(pi_action) ** 2 + EPSILON), dim=1)  # TODO: double check
            # # log_prob -= th.sum(th.log(1 - actions**2 + self.epsilon), dim=1)

            """ if not test:
                mu_vals = mu.mean(dim=0).detach().cpu().numpy()
                std_vals = std.mean(dim=0).detach().cpu().numpy()
                pi_vals = pi_action.mean(dim=0).detach().cpu().numpy()
                logp_val = logp_pi.mean().item()

                print(
                    f"mu: [{mu_vals[0]:6.3f}, {mu_vals[1]:6.3f}] "
                    f"std: [{std_vals[0]:6.3f}, {std_vals[1]:6.3f}] "
                    f"pi_action: [{pi_vals[0]:6.3f}, {pi_vals[1]:6.3f}] "
                    f"logp_pi: {logp_val:7.4f}"
                ) """

        else:
            logp_pi = None

        #pi_action_old = pi_action
        
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        # pi_action = pi_action.squeeze()

        # Debug print kept for reference
        # print(...)

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class StackedChannelCNNQFunction(nn.Module):
    """Q-function using the StackedChannelCNN backbone."""
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        action_space_size = action_space.shape[0]
        self.net = StackedChannelCNN(q_net=True, action_space_size=action_space_size)
        self.grayscale = cfg.GRAYSCALE
        self.act_buf_len = cfg.ACT_BUF_LEN

    def forward(self, obs, act):
        """Compute Q(s,a)."""
        x = (*obs, act)        
        q = self.net(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class StackedChannelCNNActorCritic(nn.Module):
    """Actor-Critic wrapper with StackedChannelCNN actor and two critics."""
    def __init__(self, observation_space, action_space):
        super().__init__()

        # build policy and value functions
        self.actor = StackedChannelCNNActor(observation_space, action_space)
        self.q1 = StackedChannelCNNQFunction(observation_space, action_space)
        self.q2 = StackedChannelCNNQFunction(observation_space, action_space)

    def act(self, obs, test=False):
        """Return a numpy action from the actor without grad."""
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()
        
class StackedChannelCNNActorCriticREDQ(nn.Module):
    """Actor-Critic wrapper with StackedChannelCNN actor and a set of REDQ critics."""
    def __init__(self, observation_space, action_space, num_critics=10):
        super().__init__()

        self.actor = StackedChannelCNNActor(observation_space, action_space)
        self.qs = nn.ModuleList([StackedChannelCNNQFunction(observation_space, action_space) for _ in range(num_critics)])

    def act(self, obs, test=False):
        """Return a numpy action from the actor without grad."""
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()

# RNN: ==========================================================================================================

from importlib import import_module

class PreTrainedCNN(Module):
    """Wrapper around torchvision CNNs to accept stacked channels and output features."""
    def __init__(self, stacked_channels=1):
        super(PreTrainedCNN, self).__init__()
        self.h_out, self.w_out = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
        self.num_channels = (1 if cfg.GRAYSCALE else 3)*stacked_channels

    # Read CNN class name from cfg
        cnn_name = cfg.TMRL_CONFIG["CNN_CLASS"]
        pretrained = cfg.TMRL_CONFIG["CNN_PRETRAINED"]

        # Importa la clase CNN
        try:
            module = import_module("torchvision.models")
            imagenet_cnn_cls = getattr(module, cnn_name)
            # Pretrained weights only valid for RGB images
            if pretrained:
                assert self.num_channels == 3 and not cfg.GRAYSCALE, "Pretrained weights only available for RGB images (3 channels)."
            # Instancia la CNN
            self.cnn = imagenet_cnn_cls(weights="DEFAULT" if pretrained else None)
            if not replace_first_conv(self.cnn, self.num_channels):
                raise RuntimeError("No se pudo reemplazar la primera capa Conv2d para aceptar más canales.")

            # Remove classifier/fc to obtain features only
            if hasattr(self.cnn, 'classifier'):
                self.cnn_features = nn.Sequential(*(list(self.cnn.children())[:-1]))
            elif hasattr(self.cnn, 'fc'):
                self.cnn_features = nn.Sequential(*(list(self.cnn.children())[:-1]))
            else:
                raise ValueError("No se reconoce la arquitectura de la CNN pasada.")
            self.normalize = True
            print(f"[PreTrainedCNN] Using {cnn_name} with {self.num_channels} input channels and feature output ({self.h_out}, {self.w_out})")
        except Exception as e:
            print(f"[PreTrainedCNN] Could not use pretrained CNN '{cnn_name}': {e}\nFalling back to VanillaCNNBN.")
            # Force hist_len=1 for RNN compatibility
            self.cnn_features = VanillaCNNBN(hist_len=1)
            self.normalize = False

    def forward(self, images):
        images = images.float() / 255.0  # Expect (batch, C, H, W)
        if self.normalize and self.num_channels == 3:  # TODO: try/validate
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
            images = (images - mean) / std
        cnn_out = self.cnn_features(images)
        # If CNN returns (batch, features, 1, 1), flatten
        if cnn_out.ndim == 4:
            cnn_out = cnn_out.view(cnn_out.size(0), -1)
        elif cnn_out.ndim == 2:
            pass  # ya está plano
        else:
            raise RuntimeError("CNN output does not have the expected shape.")
        return cnn_out

class CNNRNNEncoder(nn.Module):
    """Encode an image sequence with a CNN and an RNN (GRU/LSTM/RNN)."""
    def __init__(self, cnn: nn.Module, rnn_hidden_size=128, rnn_layers=1, rnn_type='gru'):
        """Args:
        cnn: PreTrainedCNN instance (or any CNN encoder returning (batch, features)).
        rnn_hidden_size: size of the RNN hidden state.
        rnn_layers: number of RNN layers.
        rnn_type: 'GRU', 'LSTM', or 'RNN'.
        """
        super().__init__()
        self.cnn = cnn
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers

        # Determina el tamaño de salida de la CNN de forma dinámica
        with torch.no_grad():
            dummy = torch.zeros(1, cnn.num_channels, cnn.h_out, cnn.w_out)
            cnn_out_dim = cnn(dummy).shape[1]

        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=cnn_out_dim, hidden_size=rnn_hidden_size, num_layers=rnn_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=cnn_out_dim, hidden_size=rnn_hidden_size, num_layers=rnn_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=cnn_out_dim, hidden_size=rnn_hidden_size, num_layers=rnn_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be one of 'GRU', 'LSTM', or 'RNN'")

    def forward(self, images_seq):
        """Encode image sequence.

        Args:
            images_seq: Tensor of shape (batch, seq_len, H, W, C)
        Returns:
            Tensor of shape (batch, rnn_hidden_size) with last hidden state.
        """
        batch, seq_len, H, W, C = images_seq.shape
        # Reordena a (batch, seq_len, C, H, W)
        images_seq = images_seq.permute(0, 1, 4, 2, 3).contiguous()
        # Procesa cada imagen de la secuencia por la CNN
        images_seq = images_seq.view(batch * seq_len, C, H, W)
        cnn_features = self.cnn(images_seq)  # (batch*seq_len, cnn_feat)
        cnn_features = cnn_features.view(batch, seq_len, -1)  # (batch, seq_len, cnn_feat)
        # Pasa la secuencia por la RNN
        rnn_out, _ = self.rnn(cnn_features)  # (batch, seq_len, rnn_hidden_size)
        # Devuelve el último hidden state de la secuencia
        return rnn_out[:, -1, :]  # (batch, rnn_hidden_size)
        
class MLPHead(nn.Module):
    """MLP head that merges telemetry, RNN features, and action buffers."""
    def __init__(self, rnn_hidden_size, telemetry_dim, act_buf_dim, action_dim, q_net=False, mlp_layers=(256, 256)):
        """Args:
        rnn_hidden_size: size of the RNN output.
        telemetry_dim: total dimension of telemetry data (e.g., speed, rpm, gear).
        act_buf_dim: dimension of past actions buffer (e.g., action_dim * act_buf_len).
        action_dim: dimension of current action (0 for actor head).
        q_net: True for Q-head, False for actor head.
        mlp_layers: hidden layers of the MLP.
        """
        super().__init__()
        self.q_net = q_net
        self.telemetry_dim = telemetry_dim
        if q_net:
            mlp_input_dim = rnn_hidden_size + telemetry_dim + act_buf_dim + action_dim
            mlp_sizes = [mlp_input_dim] + list(mlp_layers) + [1]
        else:
            mlp_input_dim = rnn_hidden_size + telemetry_dim + act_buf_dim
            mlp_sizes = [mlp_input_dim] + list(mlp_layers)
        self.mlp = mlp(mlp_sizes, nn.ReLU)
    
    def forward(self, telemetry, rnn_out, acts):
        """Forward pass merging inputs and applying MLP."""
        telemetry = torch.cat(telemetry, dim=-1)
        if self.q_net:
            prev_acts = acts[:-1]
            act = acts[-1]
            x_cat = torch.cat((telemetry, rnn_out, *prev_acts, act), -1)
        else:
            prev_acts = acts
            x_cat = torch.cat((telemetry, rnn_out, *prev_acts), -1)

        x_cat = self.mlp(x_cat)
        return x_cat

class CNNRNNActor(TorchActorModule):
    """SAC actor using CNNRNNEncoder and MLPHead."""
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

    # Config from cfg
        rnn_hidden_size = cfg.TMRL_CONFIG["RNN_HIDDEN_SIZE"]
        rnn_layers = cfg.TMRL_CONFIG["RNN_LAYERS"]
        rnn_type = cfg.TMRL_CONFIG["RNN_TYPE"]
        telemetry_dim = sum(int(np.prod(space.shape)) for space in observation_space[:3])
        act_buf_dim = act_buf_dim = action_space.shape[0] * cfg.ACT_BUF_LEN
        mlp_layers = cfg.TMRL_CONFIG["MLP_LAYERS"] if hasattr(cfg, "MLP_LAYERS") else (256, 256)

        self.cnn_encoder = CNNRNNEncoder(
            cnn=PreTrainedCNN(),
            rnn_hidden_size=rnn_hidden_size,
            rnn_layers=rnn_layers,
            rnn_type=rnn_type
        )
        self.head = MLPHead(
            rnn_hidden_size=rnn_hidden_size,
            telemetry_dim=telemetry_dim,
            act_buf_dim=act_buf_dim,
            action_dim=0,
            q_net=False,
            mlp_layers=mlp_layers
        )
        self.mu_layer = nn.Linear(mlp_layers[-1], dim_act)
        self.log_std_layer = nn.Linear(mlp_layers[-1], dim_act)
        self.act_limit = act_limit

        # Initialize bias so the car starts accelerating and going straight
        desired_action = torch.tensor([0.5, 0.0])  # If action_dim == 2
        unsquashed = torch.atanh(desired_action)  # Inverse of tanh

        with torch.no_grad():
            nn.init.zeros_(self.mu_layer.weight)  # No contribution from the net at init
            self.mu_layer.bias.copy_(unsquashed)

    def forward(self, obs, test=False, with_logprob=True):
        """Forward pass returning action and optional log-prob.

        obs: (speed, gear, rpm, ..., telemetry_n, images_seq, *acts)
        """
        telemetry_dim = self.head.telemetry_dim
        telemetry = [obs[i] for i in range(telemetry_dim)]
        images_seq = obs[telemetry_dim]
        acts = obs[telemetry_dim + 1:]

        rnn_out = self.cnn_encoder(images_seq)
        mlp_out = self.head(telemetry, rnn_out, acts)
        mu = self.mu_layer(mlp_out)
        log_std = self.log_std_layer(mlp_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class CNNRNNQFunction(nn.Module):
    """Q-function using CNNRNNEncoder and MLPHead, merging action after the encoder."""
    def __init__(self, observation_space, action_space):
        super().__init__()
        action_dim = action_space.shape[0]

        # Configuración desde cfg
        rnn_hidden_size = cfg.TMRL_CONFIG["RNN_HIDDEN_SIZE"]
        rnn_layers = cfg.TMRL_CONFIG["RNN_LAYERS"]
        rnn_type = cfg.TMRL_CONFIG["RNN_TYPE"]
        telemetry_dim = sum(int(np.prod(space.shape)) for space in observation_space[:3])
        act_buf_dim = act_buf_dim = action_space.shape[0] * cfg.ACT_BUF_LEN
        mlp_layers = cfg.TMRL_CONFIG["MLP_LAYERS"] if hasattr(cfg, "MLP_LAYERS") else (256, 256)

        self.cnn_encoder = CNNRNNEncoder(
            cnn=PreTrainedCNN(),
            rnn_hidden_size=rnn_hidden_size,
            rnn_layers=rnn_layers,
            rnn_type=rnn_type
        )
        self.head = MLPHead(
            rnn_hidden_size=rnn_hidden_size,
            telemetry_dim=telemetry_dim,
            act_buf_dim=act_buf_dim,
            action_dim=action_dim,
            q_net=True,
            mlp_layers=mlp_layers
        )

    def forward(self, obs, act):
        # obs: (speed, gear, rpm, ..., telemetry_n, images_seq, *acts)
        telemetry_dim = self.head.telemetry_dim
        telemetry = [obs[i] for i in range(telemetry_dim)]
        images_seq = obs[telemetry_dim]
        acts = obs[telemetry_dim + 1:]
        acts = list(obs[telemetry_dim + 1:]) + [act]

        rnn_out = self.cnn_encoder(images_seq)
        q = self.head(telemetry, rnn_out, acts)
    
        return torch.squeeze(q, -1)


class CNNRNNActorCritic(nn.Module):
    """Actor-Critic wrapper with CNNRNN actor and two critics."""
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.actor = CNNRNNActor(observation_space, action_space)
        self.q1 = CNNRNNQFunction(observation_space, action_space)
        self.q2 = CNNRNNQFunction(observation_space, action_space)

    def act(self, obs, test=False):
        """Return a numpy action from the actor without grad."""
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()

# UNSUPPORTED ==========================================================================================================


# RNN OLD: ==========================================================


def rnn(input_size, rnn_size, rnn_len):
    """Utility creating a GRU with given sizes (legacy helper)."""
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers, bias=True, batch_first=True, dropout=0, bidirectional=False)
    return gru


class SquashedGaussianRNNActor(nn.Module):
    """Legacy RNN-based squashed Gaussian policy over sequences."""
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        act_limit = act_space.high[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size] + list(mlp_sizes), activation, activation)
        self.mu_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.act_limit = act_limit
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, obs_seq, test=False, with_logprob=True, save_hidden=False):
        """Forward pass over a sequence.

        Returns (action, log_prob) and optionally saves hidden state.
        """
        self.rnn.flatten_parameters()

        # sequence_len = obs_seq[0].shape[0]
        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        obs_seq_cat = torch.cat(obs_seq, -1)
        net_out, h = self.rnn(obs_seq_cat, h)
        net_out = net_out[:, -1]
        net_out = self.mlp(net_out)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        if save_hidden:
            self.h = h

        return pi_action, logp_pi

    def act(self, obs, test=False):
        obs_seq = tuple(o.view(1, *o.shape) for o in obs)  # artificially add sequence dimension
        with torch.no_grad():
            a, _ = self.forward(obs_seq=obs_seq, test=test, with_logprob=False, save_hidden=True)
            return a.squeeze().cpu().numpy()


class RNNQFunction(nn.Module):
    """Legacy RNN Q-function merging action after the RNN latent."""
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size + dim_act] + list(mlp_sizes) + [1], activation)
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, obs_seq, act, save_hidden=False):
        """Compute Q over sequences; optionally saves hidden state."""
        self.rnn.flatten_parameters()

        # sequence_len = obs_seq[0].shape[0]
        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        # logging.debug(f"len(obs_seq):{len(obs_seq)}")
        # logging.debug(f"obs_seq[0].shape:{obs_seq[0].shape}")
        # logging.debug(f"obs_seq[1].shape:{obs_seq[1].shape}")
        # logging.debug(f"obs_seq[2].shape:{obs_seq[2].shape}")
        # logging.debug(f"obs_seq[3].shape:{obs_seq[3].shape}")

        obs_seq_cat = torch.cat(obs_seq, -1)

        # logging.debug(f"obs_seq_cat.shape:{obs_seq_cat.shape}")

        net_out, h = self.rnn(obs_seq_cat, h)
        # logging.debug(f"1 net_out.shape:{net_out.shape}")
        net_out = net_out[:, -1]
        # logging.debug(f"2 net_out.shape:{net_out.shape}")
        net_out = torch.cat((net_out, act), -1)
        # logging.debug(f"3 net_out.shape:{net_out.shape}")
        q = self.mlp(net_out)

        if save_hidden:
            self.h = h

        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class RNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()

        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianRNNActor(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
        self.q1 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
        self.q2 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)

