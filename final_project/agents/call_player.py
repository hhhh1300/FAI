from game.players import BasePokerPlayer


class CallPlayer(
    BasePokerPlayer
):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        print('valid actions')
        print(valid_actions)
        print('hole cards')
        print(hole_card)
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        # print('game info')
        # print(game_info)
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        # print("round_count")
        # print(round_count)
        # print("hole card")
        # print(hole_card)
        # print("seats")
        # print(seats)
        pass

    def receive_street_start_message(self, street, round_state):
        # print('street')
        # print(street)
        # print("round state")
        # print(round_state)
        pass

    def receive_game_update_message(self, action, round_state):
        # print('action')
        # print(action)
        # print('round state')
        # print(round_state)
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return CallPlayer()
