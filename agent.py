import numpy as np
from algorithm import Algorithm

class QLearning(Algorithm):
    def __init__(self, state_dim, action_dim, cfg):  
        super.__init__(state_dim, action_dim, cfg)
        
    def choose_action(self, state):
        self.sample_count += 1
        # 根据V值选取动作，a=argmax Q V=max Q
        # epsilon-Greedy的
        if np.random.uniform(0, 1)<self.epsilon:
            action = np.random.choice(self.action_dim)  #随机探索选取一个动作
        else:
            action = self.predict(state)
        return action
    
    def predict(self, state): 
        return np.argmax(self.Q_table[state])
    
    def update(self, state, action, reward, next_state, done):  
        best_next_action = self.predict(next_state)
        V_next_state = self.Q_table[next_state, best_next_action]
        target = (1-done) *(reward+self.gamma*V_next_state) 
        self.Q_table[state, action] = self.lr*target + (1-self.lr)*self.Q_table[state, action]



class Sarsa(object):
    def __init__(self, state_dim, action_dim, cfg): 
        super.__init__(state_dim, action_dim, cfg)

    def choose_action(self, state): 
        self.sample_count += 1
        # 根据V值选取动作，a=argmax Q V=max Q
        # epsilon-Greedy的
        if np.random.uniform(0, 1)<self.epsilon:
            action = np.random.choice(self.action_dim)  #随机探索选取一个动作
        else:
            action = self.predict(state)
        return action
    
    def predict(self, state): 
        return np.argmax(self.Q_table[state])
    
    def update(self, state, action, reward, next_state, done): 
        next_action = self.choose_action(next_state) # 先通过ε-greedy策略执行动作，然后根据所执行的动作，更新值函数。
        V_next_state = self.Q_table[next_state, next_action]
        target = (1-done) * (reward+self.gamma*V_next_state) 
        self.Q_table[state, action] = self.lr*target + (1-self.lr)*self.Q_table[state, action]