from marllib.marl.common import dict_update, get_model_config, check_algo_type, \
    recursive_dict_update
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from copy import deepcopy
from typing import Any, Dict, Tuple
import sys
from model.critic_cc_v3 import CentralizedCritic
from model.actor_IL import actor_il
from model.actor_VD import actor_vd
SYSPARAMs = deepcopy(sys.argv)


def build_model(
        environment: Tuple[MultiAgentEnv, Dict],
        algorithm: str,
        model_preference: Dict,
) -> Tuple[Any, Dict]:
    """
    construct the model
    Args:
        :param environment: name of the environment
        :param algorithm: name of the algorithm
        :param model_preference:  parameters that can be pass to the model for customizing the model

    Returns:
        Tuple[Any, Dict]: model class & model configuration
    """

    # load model
    if algorithm.algo_type == "IL": # Independent Learning
        model_class = actor_il
    elif algorithm.algo_type == "CC": # Centralized Critic
        model_class = CentralizedCritic
    else:  # Value Decomposition
        model_class = actor_vd

    if model_preference["core_arch"] in ["gru", "lstm"]:
        model_config = get_model_config("rnn")
    elif model_preference["core_arch"] in ["mlp"]:
        model_config = get_model_config("mlp")
    elif model_preference["core_arch"] in ["att"]:
        model_config = get_model_config("att")
    elif model_preference["core_arch"] in ["vit", "vit_LM", "vit_crossatt"]:
        model_config = get_model_config("vit")
    else:
        raise NotImplementedError("{} not supported agent model arch".format(model_preference["core_arch"]))

    if algorithm.algo_type == "VD":
        mixer_arch_config = get_model_config("mixer")
        model_config = recursive_dict_update(model_config, mixer_arch_config)

    return model_class, model_config