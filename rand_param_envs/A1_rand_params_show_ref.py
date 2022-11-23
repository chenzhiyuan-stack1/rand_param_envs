import numpy as np
from rand_param_envs.base import RandomEnv
from rand_param_envs.gym import utils

class A1RandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0):
        RandomEnv.__init__(self, log_scale_limit, '/home/chenzhiyuan105/unitree_mujoco/data/a1/xml/a1.xml', 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        # print("pos before")
        # print(self.model.data.qpos)
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
        #             (height > .7) and (abs(ang) < .2))
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        # print("init pos")
        # print(self.init_qpos)
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.last_qpos = qpos
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def _reward(self):
        # if self._task:
        #     return self._task(self)
        # return 0
        pass

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
        # ref_poses = np.loadtxt("/home/chenzhiyuan105/oyster/ref/canter.txt")
        # ref_poses = np.loadtxt("/home/chenzhiyuan105/oyster/ref/left_turn0.txt")
        # ref_poses = np.loadtxt("/home/chenzhiyuan105/oyster/ref/right_turn0.txt")
        # ref_poses = np.loadtxt("/home/chenzhiyuan105/oyster/ref/trot.txt")
        # ref_poses = np.loadtxt("/home/chenzhiyuan105/oyster/ref/trot2.txt")
        self.dim1 = ref_poses.shape[0]
        self.dim2 = ref_poses.shape[1]
        self.ref_poses = np.zeros([self.dim1, self.dim2, 1], dtype=float)
        for i in range(self.dim1):
            cur_ref_pose = ref_poses[i]
            tmp = np.zeros([self.dim2, 1], dtype=float)
            for j in range(self.dim2):
                tmp[j] = np.array(cur_ref_pose[j])
            self.ref_poses[i] = tmp

    def show_ref(self):
        for i in range(self.dim1):
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
    env.reset()
    env.load_ref()
    while True:
        env.show_ref()

