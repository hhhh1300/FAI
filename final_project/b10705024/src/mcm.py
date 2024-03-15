from game.engine.card import Card
from game.engine.deck import Deck
from game.engine.hand_evaluator import HandEvaluator
import random

#Reference:https://github.com/ishikota/PyPokerEngine/blob/master/pypokerengine/utils/card_utils.py

def estimate_win_rate(num_simulation, hole_card, community_card=None):
    win_count = sum([monte_carlo_simulation(hole_card, community_card) for _ in range(num_simulation)])
    return 1.0 * win_count / num_simulation

def monte_carlo_simulation(hole_card, community_card):
    community_card = fill_community_card(community_card, used_card=hole_card+community_card)
    unused_cards = sample_card(2, hole_card + community_card)
    opp_hole = unused_cards[0:2]
    opp_score = HandEvaluator.eval_hand(opp_hole, community_card)
    my_score = HandEvaluator.eval_hand(hole_card, community_card)
    if my_score > opp_score:
        return 1
    return 0

def fill_community_card(base_cards, used_card):
    return base_cards + sample_card(5-len(base_cards), used_card)

def sample_card(card_num, used_card):
    used = [card.to_id() for card in used_card]
    left_cards = [card_id for card_id in range(1, 53) if card_id not in used]
    sampled = random.sample(left_cards, card_num)
    return [Card.from_id(card_id) for card_id in sampled]
