from game.players import BasePokerPlayer
from game.game import setup_config, start_poker
from game.engine.card import Card
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from src.mcm import estimate_win_rate
from torch.distributions import Categorical
from collections import namedtuple, deque
from tqdm.notebook import tqdm
from agents.random_player import setup_ai as random_ai
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai

buffer_size = int(1e4)
BATCH_SIZE = 32         
GAMMA = 0.99            
TAU = 1e-3              
EPS_END = 0.05
EPS_START = 0.9
EPS_DECAY = 200
EPISODE_PER_BATCH = 20  # update the  agent every 5 episode
NUM_BATCH = 200        # totally update the agent for 500 time
LAST_ACTION_NUM = 2
STATE_LEN = 1 + 4*LAST_ACTION_NUM
INIT_STEP = 0
INPUT_DIM = 64
OUTPUT_DIM = 3
eps = 0.9
eps_decay = 0.9
steps_done = 0
avg_total_rewards, avg_final_rewards = [], []

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            # nn.BatchNorm1d(STATE_LEN),
            nn.Linear(STATE_LEN, INPUT_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(INPUT_DIM),
            nn.Linear(INPUT_DIM, INPUT_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(INPUT_DIM),
            nn.Linear(INPUT_DIM, OUTPUT_DIM)
        )
        # fold, call, raise, amount

    def forward(self, state):
        return F.softmax(self.network(state), dim=-1)
class ReplayBuffer:
    def __init__(self, batch_size):
        self.index = 0
        self.memory = []  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        if len(self.memory) < buffer_size:
          self.memory.append(None)
        self.memory[self.index] = self.experience(state, action, reward, next_state, done)
        self.index = (self.index + 1) % buffer_size
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
class DQNAgent(BasePokerPlayer):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Q-Network
        self.qnetwork_local = QNetwork()
        self.qnetwork_target = QNetwork()
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=1e-4)
        # Replay memory
        self.memory = ReplayBuffer(BATCH_SIZE)
        self.action_list = []
        self.opp_action_list = []
        self.my_action_list = []
        self.total_action_list = []
        self.state_list = []
        self.mask = []
        self.steps_done = INIT_STEP
        self.seq_rewards = []
        self.action_map = {'fold':0, 'call':1, 'raise':2}

    def save_model(self):
        torch.save(self.qnetwork_local, 'qnetwork_local.pt')
        torch.save(self.qnetwork_target, 'qnetwork_target.pt')

    def load_model(self):
        torch.load('qnetwork_local.pt')
        torch.load('qnetwork_target.pt')
    
    def memorize(self, state, action, reward, next_state, dones):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, dones)

    def step(self):
        if len(self.memory) > BATCH_SIZE:
            self.learn(GAMMA)

    def declare_action(self, valid_actions, hole_card, round_state):
        my_hole = [Card.from_str(card) for card in hole_card]
        community_card = [Card.from_str(card) for card in round_state['community_card']]
        w = estimate_win_rate(num_simulation=1000, hole_card=my_hole, community_card=community_card)
        
        state = np.zeros(STATE_LEN)
        idx = 0
        state[idx] = w * 3
        idx += 1
        for i in range(0, len(self.my_action_list)):
            state[idx], state[idx+1] = self.my_action_list[i][0], self.my_action_list[i][1]
            idx += 2
        for i in range(0, len(self.opp_action_list)):
            state[idx], state[idx+1] = self.opp_action_list[i][0], self.opp_action_list[i][1]
            idx += 2
            
        # print(state)
        action = self.sample(state).item()

        action_info = valid_actions[action]
        action_take = action_info['action']
        amount = 0
        if action_take == 'raise':
            amount = action_info['amount']['min']
            if amount == -1:
                action_take, amount = valid_actions[1]['action'], valid_actions[1]['amount']
                action = 1
            print(f'raise with win rate:{w}')
        else:
            amount = action_info['amount']

        self.action_list.append(action)
        self.state_list.append(state)
        self.seq_rewards.append(0)
        # print(w, action_take, amount)
        return action_take, amount

    def sample(self, state, eps=0., test=False):
        action = 0
        if test:
            self.qnetwork_local.eval()
            with torch.no_grad():
                action = self.qnetwork_local(torch.from_numpy(state).unsqueeze(0).to(torch.float32)).max(1)[1].view(1, 1)
                print(f'output:{self.qnetwork_local(torch.from_numpy(state).unsqueeze(0).to(torch.float32))}')
            return action
        
        # state = torch.from_numpy(state).unsqueeze(0)
        state = torch.from_numpy(state).unsqueeze(0).to(torch.float32)
        # Epsilon-greedy policy
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if epsilon <= np.random.uniform(0, 1):
            # print(epsilon)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action = self.qnetwork_local(state).max(1)[1].view(1, 1)
                print(f'output:{self.qnetwork_local(state)}')
        else:
            action = torch.LongTensor([[random.choice(np.arange(3))]])  
        return action

    def get_expected_state_action_values(self):
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()
        self.Q_expected = self.qnetwork_local(self.state_batch).gather(1, self.action_batch)
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.qnetwork_target(self.non_final_next_states).max(1)[0].detach()
        self.expected_state_action_values = self.reward_batch + GAMMA * next_state_values

    def learn(self, gamma):
        states, actions, rewards, next_states, dones = self.memory.sample()
        # # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute loss
        self.qnetwork_local.train()
        loss = F.mse_loss(Q_expected.to(self.device), Q_targets.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()             

    def update(self, local_model, target_model, tau):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())  

    def receive_game_start_message(self, game_info):
        self.seq_rewards = []
        self.state_list = []
        self.action_list = []
        self.opp_action_list = [(3, 0), (3, 0)]
        self.my_action_list = [(3, 0), (3, 0)]
        self.total_action_list = []
        self.mask = []
        self.uid = game_info['seats'][0]['uuid']
        self.max_round = game_info['rule']['max_round']
        self.small_blind = game_info['rule']['small_blind_amount']
        self.round_start_stack = self.stack = self.initial_stack = game_info['rule']['initial_stack']
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.opp_action_list = [(3, 0), (3, 0)]
        self.my_action_list = [(3, 0), (3, 0)]
        self.round_count = round_count
        self.last_action = None
        self.last_bet = 0
        self.last_self_action = None
        self.last_bet = 0
        self.total_bet = 0
        self.bluff = False
        self.is_small_blind_pos = False
        self.round_start_stack = seats[0]['stack']
        pass

    def receive_street_start_message(self, street, round_state):
        self.total_bet = self.stack - round_state['seats'][0]['stack']
        pass

    def receive_game_update_message(self, action, round_state):
        if action['player_uuid'] != self.uid:
            self.last_action = action['action']
            self.last_bet = action['amount']
            for i in range(1, len(self.opp_action_list)):
                self.opp_action_list[i] = self.opp_action_list[i-1]
            self.opp_action_list[0] = (self.action_map[self.last_action], 3*self.last_bet/self.initial_stack)
        else:
            for i in range(1, len(self.my_action_list)):
                self.my_action_list[i] = self.my_action_list[i-1]
            self.my_action_list[0] = (self.action_map[action['action']], 3*action['amount']/self.initial_stack)
        
        if action['player_uuid'] != self.uid:
            self.total_action_list.append(1) #opp's turn
            self.total_action_list.append(self.action_map[action['action']])
            self.total_action_list.append(3*action['amount']/self.initial_stack)
        else:
            self.total_action_list.append(0) #model's turn
            self.total_action_list.append(self.action_map[action['action']])
            self.total_action_list.append(3*action['amount']/self.initial_stack)
        

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.stack = round_state['seats'][0]['stack']
        if len(self.action_list) > 0:
            self.mask.append(len(self.action_list)-1)
        if len(self.seq_rewards) > 0:
            self.seq_rewards[-1] = 3*(self.stack - self.round_start_stack) / self.initial_stack
            if self.action_list[-1] == self.action_map['fold']:
                if self.state_list[-1][0] < 3*0.45:
                    self.seq_rewards[-1] += 0.015
                else:
                    self.seq_rewards[-1] += 0.005
                    
            i = len(self.seq_rewards) - 2
            while i >= 0 and self.seq_rewards[i] == 0:
                self.seq_rewards[i] = self.seq_rewards[i+1] * eps_decay
                if self.action_list[i] == self.action_map['fold']:
                    if self.state_list[i][0] < 3*0.45:
                        self.seq_rewards[i] += 0.015
                    else:
                        self.seq_rewards[i] += 0.005
                i -= 1
        pass

def train(preTrained=False):
    agent = DQNAgent()
    if preTrained:
        agent.load_model()
    agent.qnetwork_target.train()  # Switch network into training mode 
    agent.qnetwork_local.train()  # Switch network into training mode 
    

    # prg_bar = tqdm(range(NUM_BATCH))
    game_results = []
    win_times, win_stack = 0, 0
    for i in (range(NUM_BATCH)):
        print(f'epoch : {i}')
        agent.qnetwork_target.train()  # Switch network into training mode 
        agent.qnetwork_local.train()  # Switch network into training mode 
        mask = []
        log_probs, rewards = [], []
        # total_rewards, final_rewards = [], []
        total_rewards = 0
        # collect trajectory
        config = setup_config(max_round=EPISODE_PER_BATCH, initial_stack=1000, small_blind_amount=5)
        config.register_player(name="p1", algorithm=agent)
        config.register_player(name="p2", algorithm=baseline5_ai())
        game_result = start_poker(config, verbose=0)
        final_reward = game_result['players'][0]['stack'] - game_result['rule']['initial_stack']
        game_results.append(final_reward)
        if final_reward > 0:
            win_times += 1
        win_stack += final_reward
        print('final_reward:', final_reward)
        action_num = len(agent.state_list)
        if action_num == 0:
            continue
        seq_rewards = [agent.seq_rewards[i] for i in range(action_num)]
        # rewards = (seq_rewards - np.mean(seq_rewards)) / (np.std(seq_rewards) + 1e-9)
        rewards = (np.array(seq_rewards) + final_reward/game_result['rule']['initial_stack']/20) * 100
        # seq_rewards[-1] = final_reward
        mask = [1] * len(seq_rewards)
        for i in agent.mask:
            mask[i] = 0
        agent.state_list.append([0]*STATE_LEN)
        for i in range(action_num):
            agent.memorize(agent.state_list[i], agent.action_list[i], torch.FloatTensor([rewards[i]]), agent.state_list[i+1], mask[i])
        total_rewards += final_reward
        print(agent.action_list)
        print(mask)
        print(seq_rewards)
        print(rewards)
        for i in range(5):
            agent.step()
            agent.update(agent.qnetwork_local, agent.qnetwork_target, TAU)
        agent.save_model()
    print('-------------------', f'win rate aganist random ai : {win_times / NUM_BATCH}')
    print(f'win stack : {win_stack}')
    print(game_results)
