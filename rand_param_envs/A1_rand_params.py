from cmath import phase
from logging import root
from turtle import st
import numpy as np
from gym import spaces
from rand_param_envs.base import RandomEnv
from rand_param_envs.gym import utils
from rand_param_envs.utilities import transformations
from rand_param_envs.utilities import pose3d
from rand_param_envs.utilities import motion_util
from rand_param_envs.utilities import imitation_task
from rand_param_envs.utilities import locomotion_gym_config
from rand_param_envs.utilities import minitaur_motor
from rand_param_envs.utilities import robot_config

class A1RandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0):
        self.time_step = 0.01
        self.step_counter = 0
        self.env_step_counter = 0
        self._env_time_step = 0.001 * 33
        self.imitation_task = imitation_task.ImitationTask(ref_motion_filenames=['/home/chenzhiyuan105/oyster/rand_param_envs/rand_param_envs/utilities/pace1.txt'],
                                      enable_cycle_sync=True,
                                      tar_frame_steps=[1, 2, 10, 30],
                                      ref_state_init_prob=0.9,
                                      warmup_time=0.25)
        RandomEnv.__init__(self, log_scale_limit, '/home/chenzhiyuan105/unitree_mujoco/data/a1/xml/a1.xml', 4)
        utils.EzPickle.__init__(self)
        self.hard_reset = True
        self.reset()
        self.hard_reset = False

    def _step(self, a):
        if(self.first_try == True):
            ob = self._get_obs()
            reward = 0
            done = False
            return ob, reward, done, {}

        # if isinstance(self.action_space, spaces.Box):
        action_dim = len(a)
        wrapped_action = np.zeros([action_dim, 1])
        for i in range(action_dim):
            wrapped_action[i] = a[i]
        wrapped_action = wrapped_action.ravel()
        clipped_action = np.clip(wrapped_action, self.action_space.low, self.action_space.high)
        self.do_simulation(clipped_action, self.frame_skip)
        # self.do_simulation(a, self.frame_skip)
        
        reward = self._reward()
        
        s = self.state_vector()

        if self.imitation_task and hasattr(self.imitation_task, 'done'):
            done = self.imitation_task.done(self)
        else:
            done = False

        # done = False
        
        ob = self._get_obs()
        
        # 相当于robot.step()
        self.step_counter += self.frame_skip
        
        if self.imitation_task and hasattr(self.imitation_task, 'update'):
            self.imitation_task.update(self)      
        
        self.env_step_counter += 1
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        self.env_step_counter = 0
        self.step_counter = 0
        
        # qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        NUM_LEGS = 4
        INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * NUM_LEGS)
        HIP_JOINT_OFFSET = 0.0
        UPPER_LEG_JOINT_OFFSET = 0.0
        KNEE_JOINT_OFFSET = 0.0
        JOINT_OFFSETS = np.array(
        [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
        JOINT_DIRECTIONS = np.ones(12) 
        root_pos = np.array([0., 0., 0.32])
        root_rot = np.array([0., 0., 0., 1.])
        joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
        qpos = np.zeros(19)
        qpos[0:3] = root_pos
        qpos[3:7] = root_rot
        qpos[7:19] = joint_pose
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)        
        self.set_state(qpos, qvel)
        if self.imitation_task and hasattr(self.imitation_task, 'reset'):
            self.imitation_task.reset(self)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def _reward(self):
        if self.imitation_task:
            return self.imitation_task(self)
        return 0

    def get_time_since_reset(self):
        return self.time_step * self.step_counter

if __name__ == "__main__":
    env = A1RandParamsEnv()
    tasks = env.sample_tasks(40)
    # print(env.model.data.qpos)
    
    # while True:
    #     env.render()

    # while True:
    #     for i in range(100):
    #         qpos = env.model.data.qpos.ravel().copy()
    #         qpos[7] = 5
    #         qvel = np.zeros(18)
    #         # qvel = env.init_qvel + env.np_random.uniform(low=-.005, high=.005, size=env.model.nv)
    #         env.set_state(qpos, qvel)
    #         env.render()
    #     for i in range(100):
    #         env.reset()
    #         env.render()

    # while True:
    #     env.reset()
    #     env.set_task(np.random.choice(tasks))
    #     for _ in range(100):
    #         env.step(env.action_space.sample())  # take a random action
    #         env.render()
    
    # pace = np.loadtxt("/home/chenzhiyuan105/oyster/rand_param_envs/rand_param_envs/utilities/pace_action1.txt")
    # pace = pace.ravel()
    # action = np.zeros(12)
    # while True:
    #     for i in range(442):
    #         for j in range(12):
    #             action[j] = pace[i * 12 + j]
    #             env.step(action)
    #             env.render()

