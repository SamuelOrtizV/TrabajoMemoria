import numpy as np

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

    # Verifica si obs[3] existe antes de intentar acceder
    if len(obs) > 3:
        images = (obs[3][-1] * 255.0).astype(np.uint8)  # Convierte las imágenes si existen
    else:
        pass

    # Construye obs_mod dinámicamente
    obs_mod = (obs[0], obs[1], obs[2], images)
    rew_mod = rew
    terminated_mod = terminated
    truncated_mod = truncated
    info_mod = info
    return prev_act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod