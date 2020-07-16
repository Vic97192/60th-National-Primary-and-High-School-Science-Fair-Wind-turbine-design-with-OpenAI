import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from xfoil import XFoil
from xfoil.test import naca0012
from xfoil.generator import nurbs
from xfoil.model import Airfoil

''' 由機翼控制參數傳入 airfoil'''
def construct_airfoil(*pts):
	k = {}
	k['ta_u'] = pts[0]
	k['ta_l'] = pts[1]
	k['tb_u'] = pts[2]
	k['tb_l'] = pts[3]
	k['alpha_b'] = pts[4]
	k['alpha_c'] = pts[5]
	return nurbs.NURBS(k)

''' 將機翼之座標拆成 x, y'''
def get_coords_plain(argv):
	x_l = argv[0]
	y_l = argv[1]
	x_u = argv[2]
	y_u = argv[3]
	ycoords = np.append(y_l[::-1], y_u[1:])
	xcoords = np.append(x_l[::-1], x_u[1:])
	return ycoords, xcoords

class XFoilEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.xf = XFoil()
        self.min_action = np.array([0.1, 0.05, 1.0, 0.05, 0.4,  1.0,  0.0])
        self.max_action = np.array([0.4,  0.4, 3.0,  3.0, 8.0, 10.0, 10.0])
        ''' 測試開始, 可省略測試'''
#        (.01, .4), (.05, .4), (1, 3), (0.05, 3), (0.4, 8), (1, 10)

        #        self.xf.airfoil = naca0012
#        k = [0.08813299, 0.28250898, 2.80168427, 2.56204214, 1.48703742, 8.53824561]
#        k = [0.34422, 0.38976, 1.1, 2.9989, 1.6071, 9.9649]
        k = [0.1584, 0.1565, 2.1241, 1.8255, 11.6983, 3.827]  # org
#        k = [0.1784, 0.1365, 2.1201, 1.8057, 3.8071, 11.7009]
#        k = [0.1472, 0.1638, 2.1041, 1.8156, 3.8141, 11.6808]
#        k = [0.1784, 0.1365, 2.1201, 1.8057, 3.8071, 11.7009]
#        k = [0.1783,  0.1366,  2.1283,  1.8073,  3.8325, 11.7176]
#        k = [0.25840,  0.14474,  2.22410,  1.92550, 11.59984,  3.92623]
        airfoil = construct_airfoil(*k)
        x, y = get_coords_plain(airfoil._spline(100))
        self.xf.airfoil = Airfoil(x=x, y=y)
        #        test_airfoil = NACA4(2, 3, 15)
#        a = test_airfoil.max_thickness()
#        self.xf.airfoil = test_airfoil.get_coords()
        self.xf.Re = 1e6
        self.xf.M = 0.04
        self.xf.print = False
        cl, cd, cm, cp = self.xf.a(9)
        x = np.array(x, dtype='float32')
        y = np.array(y,dtype='float32')
#        reward = cl/cd
        reward = cl
        ''' 測試結束, 可省略測試'''


#        cl, cd, cm, cp = self.xf.a(12.2357)
#        self.action_space = spaces.Discrete(30)
        self.action_space = spaces.Box(self.min_action, self.max_action, dtype=np.float32)

#        high = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        high = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32) #創建state大小(25,f22機翼,b3cl.cd.aoa)
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
#        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        ''' action 包含 6 個機翼形狀控制參數以及 1 個攻角'''
        angular = action[6]
        k = action[0:6]
        '''將機翼控制座標傳入 airfoil, 並計算翼面之座標'''
        airfoil = construct_airfoil(*k)
        x, y = get_coords_plain(airfoil._spline(100))

        '''將座標傳入 xfoil '''
        self.xf.airfoil = Airfoil(x=x, y=y)
#        state = self.state
        self.xf.Re = 1e6
        self.xf.M = 0.04
        '''計算 cl, cd, cm, cp 當角度為 angular 時'''
        cl, cd, cm, cp = self.xf.a(angular)

        '''如果結果不穩定, 有無限大之值, 重設 state'''
        if np.isnan(cl) or np.isnan(cd) or np.isnan(cm) or np.isnan(cp):
            reward = -10.0
#            self.state = self.reset()
            done = 0
        else:
            '''如果結果穩定, 結束這個 weight 的計算'''

            '''升力最佳或升阻比最佳在此設定'''
#            reward = cl/cd
            reward = cl
            '''從機翼座標裡抽取 11 點當作 state'''
            x1, y1 = get_coords_plain(airfoil._spline(6))
            '''state : 22 個機翼形狀值, 1 個角度, 1 個 cl, cd'''
#            self.state = np.append(np.append(np.append(np.append(x1, y1),angular), cl), cd)
#            self.state = np.append(np.append(x1, y1),angular)
            self.state = np.append(x1, y1)
            done = 1

        return np.array(self.state), reward, done, {}


    def step1(self, action):
        #        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        angular = action[6]
        k = action[0:6]
        airfoil = construct_airfoil(*k)
        x, y = get_coords_plain(airfoil._spline(100))
        x1, y1 = get_coords_plain(airfoil._spline(6))
        self.xf.airfoil = Airfoil(x=x, y=y)
        #        state = self.state
        self.xf.Re = 1e6
        self.xf.M = 0.04
        cl, cd, cm, cp = self.xf.a(angular)
        if np.isnan(cl) or np.isnan(cd) or np.isnan(cm) or np.isnan(cp):
#            reward = np.nan
            reward = -10
#            self.state = self.reset()
            done = 0
        else:
#            reward = cl/cd
            reward = cl
            done = 1
#            self.state = np.append(np.append(np.append(np.append(x1, y1),angular), cl), cd)
#            self.state = np.append(np.append(x1, y1),angular)
            self.state = np.append(x1, y1)

        return np.array(self.state), reward, done, x, y

    def reset(self):
#        (.01, .4), (.05, .4), (1, 3), (0.05, 3), (0.4, 8), (1, 10)
#        0.08813299, 0.28250898, 2.80168427, 2.56204214, 1.48703742, 8.53824561
        '''為了避免 state 有無窮大的值, 所以設定decays範圍'''
#        decays = [1.0, 0.5, 0.25, 0.125, 0.0625]
        decays = [1.0, 0.999, 0.995, 0.99, 0.95, 0.9, 0.5]
        for decay in decays:
            tmp1 = self.np_random.uniform(low=-1.0, high=1.0, size=(7,))
#        tmp = tmp * [0.39, 0.35, 2.0, 2.95, 7.6, 9, 10] + [0.01, 0.05, 1, 0.05, 0.4, 1, 0]
#        tmp = tmp * [0.01, 0.1, 0.5, 0.5, 0.1, 1.0, 5] + [0.08813299, 0.28250898, 2.50168427, 2.56, 1.487, 8.54, 0]

        # k = [0.1584, 0.1565, 2.1241, 1.8255, 3.827, 11.6983]
            '''從標準NACA5410開始找'''
            tmp1 = tmp1 * decay
            tmp = tmp1 * [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 5] + [0.1584, 0.1565, 2.1241, 1.8255, 11.6983, 3.827, 6]
            airfoil = construct_airfoil(*tmp)
            x, y = get_coords_plain(airfoil._spline(100))
            self.xf.airfoil = Airfoil(x=x, y=y)
            self.xf.Re = 1e6
            self.xf.M = 0.04
            cl, cd, cm, cp = self.xf.a(tmp[6])
            if not np.isnan(cl):
                break

        x, y = get_coords_plain(airfoil._spline(6))
        self.xf.Re = 1e6
        self.xf.M = 0.04
#        self.state = np.append(np.append(np.append(np.append(x, y), tmp1[6]), cl), cd)
#        self.state = np.append(np.append(x, y), tmp1[6])
        self.state = np.append(x, y)
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):

        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
