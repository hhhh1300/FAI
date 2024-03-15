from game.players import BasePokerPlayer
from game.engine.card import Card
from src.cfr import cfrAgent
import random
from src.mcm import estimate_win_rate

class CallPlayer(
    BasePokerPlayer
):  # Do not forget to make parent class as "BasePokerPlayer"

    def __init__(self):
        super(BasePokerPlayer, self)
        self.uid = 0
        self.last_action = None
        self.last_bet = 0
        self.last_self_action = None
        self.last_self_bet = 0
        self.opp_action_list = []
        self.total_bet = 0
        self.bluff = False
        self.is_small_blind_pos = False
        self.small_blind = 5
        # self.agent = cfrAgent(100000, 5, test)
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        print(round_state['street'])
        fold_info = valid_actions[0]
        call_info = valid_actions[1]
        raise_info = valid_actions[2]
        if round_state['small_blind_pos'] == 0:
            self.is_small_blind_pos = True

        if self.stack - (self.max_round - self.round_count + 1) * (self.small_blind * 2) > self.initial_stack:
            action, amount = fold_info['action'], fold_info['amount']
            return action, amount

        my_hole = [Card.from_str(card) for card in hole_card]
        community_card = [Card.from_str(card) for card in round_state['community_card']]
        w = estimate_win_rate(num_simulation=5000, hole_card=my_hole, community_card=community_card)
        action, amount = fold_info['action'], fold_info['amount']


        if round_state['street'] == 'turn' and w <= 0.6:
            self.bluff = True
            for i in range(len(self.opp_action_list)):
                print(self.opp_action_list[i])
                if self.opp_action_list[i][0] != 'call' or self.opp_action_list[i][1] > 10:
                    self.bluff = False
                    break

    
        if w > 0.60 or (self.bluff):
            action = raise_info['action']
            amount = max(raise_info['amount']['min'], (raise_info['amount']['max'] - raise_info['amount']['min']) // 40)
            if self.bluff:
                amount = max(raise_info['amount']['min'], raise_info['amount']['max'] // 20)
            elif w > 0.80:
                amount = max(raise_info['amount']['min'], raise_info['amount']['max'] // 10)
                if (self.last_action == 'raise' and (self.last_bet >= 50 or self.stack <= 500)):
                    amount = max(raise_info['amount']['min'], raise_info['amount']['max'] // 2)
            elif self.last_action == 'raise':
                action, amount = call_info['action'], call_info['amount']
            elif w > 0.7:
                amount = max(raise_info['amount']['min'], (raise_info['amount']['max'] - raise_info['amount']['min']) // 20)


            if call_info['amount'] >= 75:
                if w <= 0.7:
                    action, amount = fold_info['action'], fold_info['amount']
            elif call_info['amount'] >= 40:
                if w < 0.75 and not self.bluff:
                    action, amount = call_info['action'], call_info['amount']

            if raise_info['amount']['min'] == -1:
                action, amount = call_info['action'], call_info['amount']

        elif w >= 0.5:
            action, amount = call_info['action'], call_info['amount']

            if (call_info['amount'] >= 20 and not self.bluff and self.last_self_action != 'raise' and self.total_bet <= 50) or (self.last_action == 'raise' and call_info['amount'] >= 50):
                action, amount = fold_info['action'], fold_info['amount']

        elif call_info['amount'] == 0:
            action, amount = call_info['action'], call_info['amount']
        elif self.last_self_action != None and (self.last_self_action == 'raise' or self.last_bet <= 40) and call_info['amount'] <= 50:
            action, amount = call_info['action'], call_info['amount']


        self.last_self_action, self.last_self_bet = action, amount
        return action, amount
        # # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        # return action, amount  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):

        self.seat_num = 0
        if game_info['seats'][1]['name'] == 'b10705024':
            self.seat_num = 1
        self.uid = game_info['seats'][self.seat_num]['uuid']
        self.max_round = game_info['rule']['max_round']
        self.small_blind = game_info['rule']['small_blind_amount']
        self.stack = self.initial_stack = game_info['rule']['initial_stack']
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        self.last_action = None
        self.last_bet = 0
        self.last_self_action = None
        self.last_bet = 0
        self.total_bet = 0
        self.bluff = False
        self.is_small_blind_pos = False
        self.opp_action_list = []

    def receive_street_start_message(self, street, round_state):
        self.total_bet = self.stack - round_state['seats'][self.seat_num]['stack']

    def receive_game_update_message(self, action, round_state):
        if action['player_uuid'] != self.uid:
            self.last_action = action['action']
            self.last_bet = action['amount']
            self.opp_action_list.append((action['action'], action['amount']))

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.stack = round_state['seats'][self.seat_num]['stack']


def setup_ai():
    return CallPlayer()
