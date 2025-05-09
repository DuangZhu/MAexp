from marllib.marl.common import dict_update, get_model_config, check_algo_type, \
    recursive_dict_update
from marllib.marl.algos import run_il, run_vd, run_cc
from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl.models import BaseRNN, BaseMLP, CentralizedCriticRNN, CentralizedCriticMLP, ValueDecompRNN, \
    ValueDecompMLP, JointQMLP, JointQRNN, DDPGSeriesRNN, DDPGSeriesMLP
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune import register_env
from copy import deepcopy
from tabulate import tabulate
from typing import Any, Dict, Tuple
import yaml
import os
import sys
from model.critic_cc_v3 import CentralizedCritic
from model.actor_IL import actor_il
from model.actor_VD import actor_vd
# from model_vit.critic_mae import CentralizedCritic_vit
# from model_vit.critic_mae_LM import CentralizedCritic_vit_LM
# from model_vit.critic_mae_crossatt import CentralizedCritic_vit_crossatt
from model_vit.actor_mae_crossatt_IL import Crossat_actor_il
from model_vit.actor_mae_crossatt_VD import Crossat_vd
# from model.critic_cc_maans import CentralizedCritic_maans
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

    if algorithm.name in ["iddpg", "facmac", "maddpg"]:
        if model_preference["core_arch"] in ["gru", "lstm"]:
            model_class = DDPGSeriesRNN
        else:
            model_class = DDPGSeriesMLP

    elif algorithm.name in ["qmix", "vdn", "iql"]:
        if model_preference["core_arch"] in ["gru", "lstm"]:
            model_class = JointQRNN
        else:
            model_class = JointQMLP

    else:
        if algorithm.algo_type == "IL":
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = BaseRNN
            elif model_preference["core_arch"] in ["vit_crossatt"]:
                model_class = Crossat_actor_il
            else:
                model_class = actor_il
        elif algorithm.algo_type == "CC":
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = CentralizedCriticRNN
            elif model_preference["core_arch"] in ["vit"]:
                model_class = CentralizedCritic_vit
            elif model_preference["core_arch"] in ["vit_LM"]:
                model_class = CentralizedCritic_vit_LM
            elif model_preference["core_arch"] in ["vit_crossatt"]:
                model_class = CentralizedCritic_vit_crossatt
            elif model_preference["core_arch"] in ["maans"]:
                model_class = CentralizedCritic_maans
            else:
                model_class = CentralizedCritic # CentralizedCritic or CentralizedCritic_ttbot
        else:  # VD
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = ValueDecompRNN
            elif model_preference["core_arch"] in ["vit_crossatt"]:
                model_class = Crossat_vd
            else:
                model_class = actor_vd

    if model_preference["core_arch"] in ["gru", "lstm"]:
        model_config = get_model_config("rnn")
    elif model_preference["core_arch"] in ["mlp", "maans"]:
        model_config = get_model_config("mlp")
    elif model_preference["core_arch"] in ["att"]:
        model_config = get_model_config("att")
    elif model_preference["core_arch"] in ["vit", "vit_LM", "vit_crossatt"]:
        model_config = get_model_config("vit")
    else:
        raise NotImplementedError("{} not supported agent model arch".format(model_preference["core_arch"]))

    if len(environment[0].observation_space.spaces["obs"].shape) == 1:
        encoder = "fc_encoder"
    else:
        encoder = "cnn_encoder"

    # encoder config
    encoder_arch_config = get_model_config(encoder)
    model_config = recursive_dict_update(model_config, encoder_arch_config)
    model_config = recursive_dict_update(model_config, {"model_arch_args": model_preference})

    if algorithm.algo_type == "VD":
        mixer_arch_config = get_model_config("mixer")
        model_config = recursive_dict_update(model_config, mixer_arch_config)

    return model_class, model_config