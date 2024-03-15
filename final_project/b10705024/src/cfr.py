from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator
import random
import copy
import numpy as np
import json
import pickle
class infoset():
    def __init__(self, sb=5, money=1000, player=1):
        self.state_table = {'terminal':0, 'chance':1, 'fold':2, 'call':3, 'raise':4, 'bet':5}
        self.history = []
        self.hole_card = []
        self.community = []
        self.pot = sb * 3
        self.p1 = money
        self.p2 = money
        self.init_player = player

    def add_history(self, state, num=0, new_card=None, player=0):
        self.history += [[player, self.state_table[state], num]]
        if new_card is not None:
            self.community += new_card
        if player == 1:
            self.p1 -= num
        else:
            self.p2 -= num
        self.pot += num
    def last_player(self):
        return self.history[-1][0]

    def last_state(self):
        return self.history[-1][1]
    
    def last_num(self):
        return self.history[-1][2]

    def get_info(self):
        all_cards = ([c for c in self.hole_card[0]] + self.community)
        all_cards.sort()
        # print(tuple(all_cards))
        return tuple(all_cards)

class cfrAgent():
    def __init__(self, iter, sb, test=False):
        self.sb = sb
        self.iter = iter
        self.suit = "CDHS"
        self.rank = "23456789TJQKA"
        # fold, call, raise(10 scales)
        self.total_actions = 3 
        self.cum_regret_table = {}
        self.cum_strategy_table = {}
        self.profile = {}
        with open('strategy.json','r') as jsonFile:
            self.cum_strategy_table = json.load(jsonFile)
            self.cum_strategy_table = {self._str2tuple(k): self.cum_strategy_table[k] for k in self.cum_strategy_table}
        if not test:
            with open('profile.json','r') as jsonFile:
                self.profile = json.load(jsonFile)
                self.profile = {self._str2tuple(k): self.profile[k] for k in self.profile}
            with open('regret.json','r') as jsonFile:
                self.cum_regret_table = json.load(jsonFile)
                self.cum_regret_table = {self._str2tuple(k): self.cum_regret_table[k] for k in self.cum_regret_table}
            self.train()

    def _str2tuple(self, key):
        a = []
        i = 2
        while i + 2 < len(key):
            a.append(key[i:i+2])
            i = i + 6
        # print(tuple(a))
        return tuple(a)

    def is_terminal(self, h):
        if h.last_state() == h.state_table['terminal']:
            return True
        return False

    def is_chance(self, h):
        if h.last_state() == h.state_table['chance']:
            return True
        return False

    def is_fold(self, h):
        if h.last_state() == h.state_table['fold']:
            return True
        return False

    def sample_action(self, valid_actions, ratios):
        ratios = np.array(ratios)
        total_raitos = np.sum(ratios)
        if total_raitos != 0:
            ratios = ratios / total_raitos
        else:
            ratios = [1/len(valid_actions)]*len(valid_actions)
        print(ratios)
        if abs(ratios[0] - ratios[1]) <= 1e-6 and abs(ratios[1] - ratios[2]) <= 1e-6:
            return valid_actions[1], ratios[1]
            
        # eps = random.random()
        # for i in range(len(valid_actions)):
        #     # print(eps, ratios[i])
        #     if eps <= ratios[i]:
        #         return valid_actions[i]
        #     eps -= ratios[i]
        # return valid_actions[2]
        print(np.argmax(ratios))
        return valid_actions[np.argmax(ratios)], ratios[np.argmax(ratios)]
    
    def sample_card(self, hole_card, community, num):
        # Format: hole_card = [my_cards, opp_cards]
        cards = []
        for i in range(num):
            card = random.choice(self.suit) + random.choice(self.rank)
            while card in hole_card[0] or card in hole_card[1] or card in community or card in cards:
                card = random.choice(self.suit) + random.choice(self.rank)
            cards += [card]
        cards.sort()
        return cards

    def cfr(self, h, i, t, pi_1, pi_2, stage, depth):
        # print(h.history)
        if self.is_fold(h):
            # print('fold')
            if h.last_player == h.init_player:
                return -h.pot
            return h.pot
        if self.is_terminal(h):
            # print(f'depth : {depth}')
            # print('terminal')
            my_hole = [Card.from_str(card) for card in h.hole_card[0]]
            opp_hole = [Card.from_str(card) for card in h.hole_card[1]]

            cards = self.sample_card(h.hole_card, h.community, 5-len(h.community))

            community = [Card.from_str(card) for card in h.community+cards]
            my_score = HandEvaluator.eval_hand(my_hole, community)
            opp_score = HandEvaluator.eval_hand(opp_hole, community)
            if my_score >= opp_score:
                return h.pot
            else:
                return -h.pot
        if self.is_chance(h):
            # print('chance')
            card = self.sample_card(h.hole_card, h.community, 3 if stage == 'preflop' else 1)
            h.add_history('bet', 0, card, i)
            return self.cfr(h, i, t, pi_1, pi_2, 'flop' if stage == 'preflop' else ('turn' if stage == 'flop' else 'river'), depth+1)
        v_profile = 0
        v_p2a = [0]*self.total_actions
        for act in range(self.total_actions):
            _h = copy.deepcopy(h)
            if (i == 1 and h.p1-500 < 0) or (i == 2 and h.p2-500 < 0):
                _h.add_history('terminal', 0, None, i)
                stage = 'river'
            else:
                if act == 0:
                    _h.add_history('fold', 0, None, i)
                elif act == 1:
                    _h.add_history('call', _h.last_num(), None, i)
                    if _h.last_state() == _h.state_table['call']:
                        _h.add_history('chance' if stage != 'river' else 'terminal', 0, None, i)
                elif act == 2:
                    _h.add_history('raise', min(_h.p1, 500), None, i)
            if tuple(_h.get_info()) not in self.profile:
                    self.profile[tuple(_h.get_info())] = [1/self.total_actions] * self.total_actions 
            if i == 1:
                v_p2a[act] = self.cfr(_h, 2, t, self.profile[tuple(_h.get_info())][act] * pi_1, pi_2, stage, depth+1)
            else:
                v_p2a[act] = self.cfr(_h, 1, t, pi_1, self.profile[tuple(_h.get_info())][act] * pi_2, stage, depth+1)
            v_profile = v_profile + self.profile[tuple(h.get_info())][act] * v_p2a[act]
        
        for act in range(self.total_actions):
            if tuple(h.get_info()) not in self.cum_regret_table:
                self.cum_regret_table[tuple(h.get_info())] = [0] * self.total_actions 
            if tuple(h.get_info()) not in self.cum_strategy_table:
                self.cum_strategy_table[tuple(h.get_info())] = [0] * self.total_actions 
            if i == 1:
                self.cum_regret_table[tuple(h.get_info())][act] += pi_2 * (v_p2a[act] - v_profile)
                self.cum_strategy_table[tuple(h.get_info())][act] += pi_1 * self.profile[tuple(h.get_info())][act]
            else:
                self.cum_regret_table[tuple(h.get_info())][act] += pi_1 * (v_p2a[act] - v_profile)
                self.cum_strategy_table[tuple(h.get_info())][act] += pi_2 * self.profile[tuple(h.get_info())][act]
        total_regret = 0
        for act in range(self.total_actions):
            total_regret += max(0, self.cum_regret_table[tuple(h.get_info())][act])
        for act in range(self.total_actions):
            if total_regret <= 0:
                self.profile[tuple(h.get_info())][act] = 1/self.total_actions 
            else:
                self.profile[tuple(h.get_info())][act] = max(0, self.cum_regret_table[tuple(h.get_info())][act]) / total_regret
        return v_profile

    def train(self):
        for t in range(self.iter):
            print(f'epoch : {t}')
            for i in [1, 2]:
                my_cards = self.sample_card([[], []], [], 2)
                opp_cards = self.sample_card([my_cards, []], [], 2)
                if tuple(my_cards) not in self.profile:
                    self.profile[tuple(my_cards)] = [1/self.total_actions] * self.total_actions 
                # community = self.sample_card([my_cards, opp_cards], [], 3)
                h = infoset(self.sb, 1000, i)
                h.hole_card = [my_cards, opp_cards]
                if i == 1:
                    h.p1 -= self.sb
                    h.p2 -= 2 * self.sb
                else:
                    h.p1 -= 2 * self.sb
                    h.p2 -= self.sb
                h.add_history('bet', 0, None, i)
                self.cfr(h, i, t, 1, 1, 'preflop', 0)
            print(f'info len : {len(self.profile)}')
        with open('./profile.json','w') as jsonFile:
            json.dump({str(k): self.profile[k] for k in self.profile}, jsonFile)
        with open('./regret.json','w') as jsonFile:
            json.dump({str(k): self.cum_regret_table[k] for k in self.cum_regret_table}, jsonFile)
        with open('./strategy.json','w') as jsonFile:
            json.dump({str(k): self.cum_strategy_table[k] for k in self.cum_strategy_table}, jsonFile)