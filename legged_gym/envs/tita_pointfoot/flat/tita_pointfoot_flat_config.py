from legged_gym.envs.tita_pointfoot.mixed_terrain.tita_pointfoot_rough_config import TITAPointFootRoughCfg, TITAPointFootRoughCfgPPO


class TITAPointFootFlatCfg(TITAPointFootRoughCfg):
    class env(TITAPointFootRoughCfg.env):
        num_privileged_obs = 27

    class terrain(TITAPointFootRoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights_critic = False

    class asset(TITAPointFootRoughCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(TITAPointFootRoughCfg.rewards):
        max_contact_force = 350.

        class scales(TITAPointFootRoughCfg.rewards.scales):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 5.
            unbalance_feet_air_time = 1.0
            no_fly = 1.
            # feet_contact_forces = -0.01

    class commands(TITAPointFootRoughCfg.commands):
        num_commands = 3
        heading_command = False
        resampling_time = 4.

        class ranges(TITAPointFootRoughCfg.commands.ranges):
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand(TITAPointFootRoughCfg.domain_rand):
        friction_range = [0.,
                          1.5]  # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.


class TITAPointFootFlatCfgPPO(TITAPointFootRoughCfgPPO):
    class policy(TITAPointFootRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]

    class runner(TITAPointFootRoughCfgPPO.runner):
        experiment_name = 'tita_pointfoot_flat'
        max_iterations = 30000
