from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, get_load_path, class_to_dict
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

import numpy as np
import torch
import copy

def export_policy_as_onnx(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    loaded_dict = torch.load(resume_path)
    actor_critic_class = eval(train_cfg.runner.policy_class_name)
    if env_cfg.env.num_privileged_obs is None:
        env_cfg.env.num_privileged_obs = env_cfg.env.num_propriceptive_obs
    actor_critic = actor_critic_class(
        env_cfg.env.num_propriceptive_obs, env_cfg.env.num_privileged_obs, env_cfg.env.num_actions, **class_to_dict(train_cfg.policy)
    ).to(args.rl_device)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    
    # 将模型移动到 GPU 上
    model = copy.deepcopy(actor_critic.actor).to("cuda")
    model.eval()

    # 生成一个 GPU 版本的 dummy_input
    # dummy_input = torch.randn(env_cfg.env.num_propriceptive_obs).to("cuda")
    dummy_input = torch.randn(1, env_cfg.env.num_propriceptive_obs).to("cuda")
    input_names = ["nn_input"]
    output_names = ["nn_output"]

    # 导出为 GPU 版本的 ONNX 模型
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported GPU version of the policy as ONNX to: ", path)

if __name__ == '__main__':
    args = get_args()
    export_policy_as_onnx(args)