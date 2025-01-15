import os
import gym
# import torch

from utils import CliffWalkingWapper, save_results, make_dir
from agent import QLearning, Sarsa
from plot import plot_rewards
from typing import Union
from algorithm import Algorithm

curr_path = os.path.dirname(__file__)
algo_dict = {
    'QLearning': QLearning,
    'Sarsa': Sarsa
}

class Trainer(object):
    '''训练相关参数'''
    def __init__(
            self, 
            cfg,
            algo:Union[str, Algorithm] ='QLearning',
            env='CliffWalking-v0',# 0 up, 1 right, 2 down, 3 left
    ):
        self.seed = 0
        self.algo = algo
        self.env = env  
        self.agent = algo_dict[algo] if isinstance(algo,str) else algo,
        self.cfg = cfg,
        self.eval_eps = 
        self.result_path = curr_path + "/outputs/" + self.env + '/' + '/results/'  # path to save results
        self.model_path = curr_path + "/outputs/" + self.env + '/' + '/models/'  # path to save models
        self.train_eps = 200  # 训练的episode数目
        self.eval_eps = 30
        self.gamma = 0.9  # reward的衰减率
        self.lr = 0.1  # learning rate
        self.render_frqc = 30 # 仿真渲染频率
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")  # check gpu