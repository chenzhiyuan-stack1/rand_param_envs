from cmath import phase
from logging import root
from turtle import st
import numpy as np
from rand_param_envs.base import RandomEnv
from rand_param_envs.gym import utils
from rand_param_envs.utilities import transformations
from rand_param_envs.utilities import pose3d
from rand_param_envs.utilities import motion_util

class A1RandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0):
        # information about reference pose
        self.time_step = 0.01
        self.step_counter = 0
        self.frame_duration = 0.01667
        # hyperparameters of reward function
        self._weight = 1.0
        self._pose_weight = 0.5
        self._velocity_weight=0.05
        self._end_effector_weight=0.2
        self._root_pose_weight=0.15
        self._root_velocity_weight=0.1
        self._pose_err_scale = 5.0
        self._velocity_err_scale=0.1
        self._end_effector_err_scale=40
        self._end_effector_height_err_scale=3.0
        self._root_pose_err_scale=20
        self._root_velocity_err_scale=2
        RandomEnv.__init__(self, log_scale_limit, '/home/chenzhiyuan105/unitree_mujoco/data/a1/xml/a1.xml', 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        # posbefore = self.model.data.qpos[0, 0]
        if (hasattr(self, 'num_frames') == False):
            self.load_ref() # set self.frame_dimension & self.num_frames
        self.do_simulation(a, self.frame_skip)
        # posafter, height, ang = self.model.data.qpos[0:3, 0]
        # alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        reward = self.reward()
        s = self.state_vector()
        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
        #             (height > .7) and (abs(ang) < .2))
        done = False
        ob = self._get_obs()
        self.step_counter += self.frame_skip # +1 or +self.frame_skip
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        self.step_counter = 0
        
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        
        self.last_qpos = qpos
        
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def get_motion_time(self):
        return self.time_step * self.step_counter

    def calc_frame(self):
        time = self.get_motion_time()
        motion_duration = (self.num_frames - 1) * self.frame_duration
        phase = time / motion_duration
        phase -= np.floor(phase) # enable_loop = True
        
        f0 = int(phase * (self.num_frames - 1))
        f1 = min(f0 + 1, self.num_frames - 1)

        norm_time = phase * motion_duration
        time0 = f0 * self.frame_duration
        time1 = f1 * self.frame_duration
        assert (norm_time >= time0 - 1e-5) and (norm_time <= time1 + 1e-5)
        
        blend = (norm_time - time0) / (time1 - time0) # help us decide to choose f0 or f1
        frame0 = self.ref_poses[f0].ravel() # 数据格式与model.data.qpos.ravel()相同
        frame1 = self.ref_poses[f1].ravel()
        
        blend_frame = np.zeros(self.frame_dim)
        blend_frame[0:3] = (1.0 - blend) * frame0[0:3] + blend * frame1[0:3] # root pos
        blend_frame[7:19] = (1.0 - blend) * frame0[7:19] + blend * frame1[7:19] # joints
        root_rot0 = frame0[3:7]
        root_rot1 = frame1[3:7]
        blend_root_rot = transformations.quaternion_slerp(root_rot0, root_rot1, blend)
        if blend_root_rot[-1] < 0:
            blend_root_rot = -blend_root_rot # standardize_quaternion
        blend_frame[3:7] = blend_root_rot
        # blend_frame[3:7] = frame0[3:7]
        return blend_frame
        
    def reward(self):
        pose_reward = self._calc_reward_pose()
        velocity_reward = self._calc_reward_velocity()
        end_effector_reward = self._calc_reward_end_effector()
        root_pose_reward = self._calc_reward_root_pose()
        root_velocity_reward = self._calc_reward_root_velocity()
        reward = self._pose_weight * pose_reward \
                + self._velocity_weight * velocity_reward \
                + self._end_effector_weight * end_effector_reward \
                + self._root_pose_weight * root_pose_reward \
                + self._root_velocity_weight * root_velocity_reward
        return reward * self._weight

    def _calc_reward_pose(self):
        pose_err = 0.0
        state_ref = self.calc_frame()
        for j in range(self.frame_dim - 7):
            j_pose_ref = state_ref[j + 7]
            j_pose_sim = self.model.data.qpos.ravel()[j + 7]
            j_pose_diff = j_pose_ref - j_pose_sim
            j_pose_err = j_pose_diff * j_pose_diff
            pose_err += j_pose_err
        pose_reward = np.exp(-self._pose_err_scale * pose_err)

        return pose_reward

    def _calc_reward_velocity(self):
        return 0.0

    def _calc_reward_end_effector(self):
        return 0.0
    
    def _calc_reward_root_pose(self):
        root_pose_err = 0.0
        state_ref = self.calc_frame()
        state_sim = self.model.data.qpos.ravel()
        root_pos_ref = state_ref[0:3]
        root_rot_ref = state_ref[3:7]
        root_pos_sim = state_sim[0:3]
        root_rot_sim = state_sim[3:7]

        root_pos_diff = root_pos_ref - root_pos_sim
        root_pos_err = root_pos_diff.dot(root_pos_diff)

        root_rot_diff = transformations.quaternion_multiply(
            root_rot_ref, transformations.quaternion_conjugate(root_rot_sim))
        _, root_rot_diff_angle = pose3d.QuaternionToAxisAngle(root_rot_diff)
        root_rot_diff_angle = motion_util.normalize_rotation_angle(
            root_rot_diff_angle)
        root_rot_err = root_rot_diff_angle * root_rot_diff_angle

        root_pose_err = root_pos_err + 0.5 * root_rot_err
        root_pose_reward = np.exp(-self._root_pose_err_scale * root_pose_err)
        
        return root_pose_reward

    def _calc_reward_root_velocity(self):
        return 0.0

    def _termination(self):
        # if not self._robot.is_safe:
        #     return True

        # if self._task and hasattr(self._task, 'done'):
        #     return self._task.done(self)

        # for s in self.all_sensors():
        #     s.on_terminate(self)

        # return False
        pass

    def load_ref(self):
        ref_poses = np.loadtxt("/home/chenzhiyuan105/oyster/ref/pace.txt")
        self.num_frames = ref_poses.shape[0]
        self.frame_dim = ref_poses.shape[1]
        self.ref_poses = np.zeros([self.num_frames, self.frame_dim, 1], dtype=float)
        for i in range(self.num_frames):
            cur_ref_pose = ref_poses[i]
            tmp = np.zeros([self.frame_dim, 1], dtype=float)
            for j in range(self.frame_dim):
                tmp[j] = np.array(cur_ref_pose[j])
            self.ref_poses[i] = tmp

    def show_ref(self):
        for i in range(self.num_frames):
            qpos = self.ref_poses[i].ravel()
            qpos[0] = self.last_qpos.ravel()[0] - 0.0145
            qpos[1] = self.last_qpos.ravel()[1] - 0.0005
            self.last_qpos = qpos
            qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
            self.set_state(qpos, qvel)
            self.render()

if __name__ == "__main__":

    env = A1RandParamsEnv()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        for _ in range(100):
            env.render()
            env.step(env.action_space.sample())  # take a random action
            print("env.reward")
            print(env.reward())

