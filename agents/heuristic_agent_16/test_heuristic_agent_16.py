"""Standalone contract/rules tests. Does not import or validate the user's engine.

Run: python -m unittest -v test_heuristic_agent_16.py
The miniature reference environment below is independent of RoutePlanner.
"""
import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations, product
import json
import importlib.util
from pathlib import Path
import random
import sys
import types
import unittest


class GemColor(Enum):
    WHITE = auto()
    BLUE = auto()
    GREEN = auto()
    RED = auto()
    BLACK = auto()
    GOLD = auto()


class NodeType(Enum):
    MAIN_DECISION = auto()
    OVERFLOW_DISCARD = auto()
    NOBLE_CLAIM = auto()


class ActionType(Enum):
    BUY_VISIBLE = auto()
    BUY_RESERVED = auto()
    TAKE_GEMS = auto()
    RESERVE_VISIBLE = auto()
    RESERVE_TOP_DECK = auto()
    DISCARD_GEMS = auto()
    TAKE_NOBLE = auto()


C = tuple(GemColor)[:5]
AC = tuple(GemColor)
module_names = ('splendor_v1', 'splendor_v1.env', 'splendor_v1.env.core',
             'splendor_v1.env.core.constants', 'splendor_v1.env.core.enums',
             'splendor_v1.env.core.actions')
saved_modules = {name: sys.modules.get(name) for name in module_names}
for name in module_names:
    sys.modules[name] = types.ModuleType(name)
sys.modules['splendor_v1.env.core.constants'].COLOR_ORDER = C
sys.modules['splendor_v1.env.core.enums'].GemColor = GemColor
sys.modules['splendor_v1.env.core.enums'].NodeType = NodeType
sys.modules['splendor_v1.env.core.actions'].ActionType = ActionType

try:
    spec = importlib.util.spec_from_file_location('_heuristic16_contract_under_test',
                                                Path(__file__).with_name('heuristic_agent_16.py'))
    agent_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent_module
    spec.loader.exec_module(agent_module)
    for name in ('HeuristicAgent16', 'RoutePlanner', 'Position', 'CardInfo', 'NobleInfo',
                 'card_info', 'card_type', 'noble_bundle_requirements'):
        globals()[name] = getattr(agent_module, name)
finally:
    for name, old in saved_modules.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


@dataclass
class Card:
    id: int
    tier: int
    points: int
    bonus_color: GemColor
    cost: dict


@dataclass
class Noble:
    id: int
    points: int
    requirement: dict


@dataclass
class Player:
    gems: dict = field(default_factory=lambda: dict.fromkeys(AC, 0))
    bonuses: dict = field(default_factory=lambda: dict.fromkeys(C, 0))
    reserved_cards: list = field(default_factory=list)
    purchased_cards: list = field(default_factory=list)
    points: int = 0


@dataclass
class State:
    players: list = field(default_factory=lambda: [Player(), Player()])
    bank: dict = field(default_factory=lambda: {c: (5 if c == GemColor.GOLD else 4) for c in AC})
    visible_cards: dict = field(default_factory=lambda: {i: [] for i in (1, 2, 3)})
    decks: dict = field(default_factory=lambda: {i: [] for i in (1, 2, 3)})
    nobles: list = field(default_factory=list)
    current_player: int = 0
    node_type: NodeType = NodeType.MAIN_DECISION
    max_gems: int = 10
    winners: object = None

    def clone(self):
        return copy.deepcopy(self)


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    tier: int = 0
    slot: int = 0
    reserved_index: int = 0
    gem_colors: tuple = ()
    payment: tuple = ()


def definitions():
    data = json.loads((Path(__file__).parent / 'test_card_data.json').read_text())
    cards = []
    for tier in (1, 2, 3):
        for row in data['BASE_TIER_' + str(tier)]:
            cards.append(Card(row['id'], tier, row['points'], GemColor[row['bonus_color']],
                              {GemColor[k]: v for k, v in row['cost'].items()}))
    nobles = [Noble(n['id'], n['points'], {GemColor[k]: v for k, v in n['requirement'].items()})
              for n in data['NOBLES']]
    return cards, nobles


class MiniEnv:
    def __init__(self):
        self.ids = {}

    def clone(self):
        return copy.deepcopy(self)

    def action_to_id(self, action):
        if action not in self.ids:
            self.ids[action] = len(self.ids)
        return self.ids[action]

    @staticmethod
    def _check_terminated(state):
        return state.winners is not None

    @staticmethod
    def _eligible(state):
        p = state.players[state.current_player]
        return [i for i, n in enumerate(state.nobles) if n is not None
                and all(p.bonuses[c] >= n.requirement[c] for c in C)]

    @staticmethod
    def _payments(player, card):
        req = [max(0, card.cost[c] - player.bonuses[c]) for c in C]
        # Independently enumerate how much GOLD substitutes each color.
        for gold_parts in product(*(range(min(n, player.gems[GemColor.GOLD]) + 1) for n in req)):
            if sum(gold_parts) > player.gems[GemColor.GOLD]:
                continue
            colored = tuple(n - g for n, g in zip(req, gold_parts))
            if all(n <= player.gems[c] for c, n in zip(C, colored)):
                yield colored + (sum(gold_parts),)

    def _legal_actions(self, state):
        if self._check_terminated(state):
            return []
        p = state.players[state.current_player]
        if state.node_type == NodeType.OVERFLOW_DISCARD:
            return [Action(ActionType.DISCARD_GEMS, gem_colors=(c,)) for c in AC if p.gems[c]]
        if state.node_type == NodeType.NOBLE_CLAIM:
            return [Action(ActionType.TAKE_NOBLE, slot=i) for i in self._eligible(state)]
        out = []
        available = [c for c in C if state.bank[c]]
        if available:
            out.extend(Action(ActionType.TAKE_GEMS, gem_colors=x)
                       for x in combinations(available, min(3, len(available))))
        out.extend(Action(ActionType.TAKE_GEMS, gem_colors=(c, c)) for c in C if state.bank[c] >= 4)
        for tier, row in state.visible_cards.items():
            for slot, card in enumerate(row):
                if card is None:
                    continue
                out.extend(Action(ActionType.BUY_VISIBLE, tier, slot, payment=pay)
                           for pay in self._payments(p, card))
                if len(p.reserved_cards) < 3:
                    out.append(Action(ActionType.RESERVE_VISIBLE, tier, slot))
        for index, card in enumerate(p.reserved_cards):
            out.extend(Action(ActionType.BUY_RESERVED, reserved_index=index, payment=pay)
                       for pay in self._payments(p, card))
        if len(p.reserved_cards) < 3:
            out.extend(Action(ActionType.RESERVE_TOP_DECK, tier=tier)
                       for tier, deck in state.decks.items() if deck)
        return out

    @staticmethod
    def _end(state):
        owner = state.current_player
        if owner == 1 and max(p.points for p in state.players) >= 15:
            score = max(p.points for p in state.players)
            tied = [i for i, p in enumerate(state.players) if p.points == score]
            fewest = min(len(state.players[i].purchased_cards) for i in tied)
            state.winners = [i for i in tied if len(state.players[i].purchased_cards) == fewest]
        state.current_player = 1 - owner
        state.node_type = NodeType.MAIN_DECISION

    def _post(self, state):
        if sum(state.players[state.current_player].gems.values()) > state.max_gems:
            state.node_type = NodeType.OVERFLOW_DISCARD
        elif self._eligible(state):
            state.node_type = NodeType.NOBLE_CLAIM
        else:
            self._end(state)

    def step(self, action, state):
        if action not in self._legal_actions(state):
            raise AssertionError('Illegal fixture action: ' + repr(action))
        p = state.players[state.current_player]
        kind = action.action_type
        if kind == ActionType.TAKE_NOBLE:
            p.points += state.nobles[action.slot].points
            state.nobles[action.slot] = None
            self._end(state)
            return state
        if kind == ActionType.DISCARD_GEMS:
            c = action.gem_colors[0]
            p.gems[c] -= 1
            state.bank[c] += 1
        elif kind == ActionType.TAKE_GEMS:
            for c in action.gem_colors:
                p.gems[c] += 1
                state.bank[c] -= 1
        else:
            if kind == ActionType.BUY_RESERVED:
                card = p.reserved_cards.pop(action.reserved_index)
            elif kind == ActionType.RESERVE_TOP_DECK:
                card = state.decks[action.tier].pop()
            else:
                card = state.visible_cards[action.tier][action.slot]
                deck = state.decks[action.tier]
                state.visible_cards[action.tier][action.slot] = deck.pop() if deck else None
            if kind in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
                for c, n in zip(AC, action.payment):
                    p.gems[c] -= n
                    state.bank[c] += n
                p.points += card.points
                p.bonuses[card.bonus_color] += 1
                p.purchased_cards.append(card)
            else:
                p.reserved_cards.append(card)
                if state.bank[GemColor.GOLD]:
                    state.bank[GemColor.GOLD] -= 1
                    p.gems[GemColor.GOLD] += 1
        self._post(state)
        return state


class FixturePolicy:
    def __init__(self):
        self.rng = random.Random(1)

    def select_action(self, env, state):
        legal = env._legal_actions(state)
        buys = [a for a in legal if a.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED)]
        if buys:
            return max(buys, key=lambda a: HeuristicAgent16._action_card(state, a).points)
        return self.rng.choice(legal) if legal else None


def position(gems=(0, 0, 0, 0, 0, 0), bonuses=(0, 0, 0, 0, 0), count=0, nobles=()):
    return Position(gems, bonuses, tuple(n - g for n, g in zip((4, 4, 4, 4, 4, 5), gems)),
                    tuple(range(count)), (), tuple(range(len(nobles))), 0)


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards, cls.nobles = definitions()

    def test_eighteen_families_and_feedback(self):
        self.assertEqual(len({card_type(c)['family'] for c in self.cards}), 18)
        for i in (45, 51, 57, 63, 69):
            self.assertEqual(card_type(self.cards[i])['feedback'], 'main_color')
        self.assertEqual(card_type(self.cards[83])['feedback'], 'different_color')

    def test_corrected_noble_bundle(self):
        nobles = [NobleInfo(tuple(self.nobles[i].requirement[c] for c in C)) for i in (0, 8, 9)]
        bundles = noble_bundle_requirements(nobles, (0,) * 5)
        all_three = next(b for b in bundles if len(b['nobles']) == 3)
        self.assertEqual(all_three['requirement'], (4, 4, 4, 0, 0))
        self.assertEqual((all_three['missing_purchases'], all_three['points']), (12, 9))

    def test_payment_options_match_reference_including_optional_gold(self):
        rng = random.Random(31)
        for _ in range(80):
            card = rng.choice(self.cards)
            gems = tuple(rng.randrange(5) for _ in C) + (rng.randrange(4),)
            bonuses = tuple(rng.randrange(5) for _ in C)
            p = position(gems, bonuses, 1)
            player = Player(dict(zip(AC, gems)), dict(zip(C, bonuses)))
            planner = RoutePlanner([card_info(card)], [], payment_width=1000)
            self.assertEqual(set(planner.payments(p, card_info(card))), set(MiniEnv._payments(player, card)))

    def test_all_zero_point_tier_one_reachable_by_third_turn(self):
        # Exhaustive local state expansion, independent of heuristic beam scores.
        front = [position()]
        collector = RoutePlanner([], [], reserve_width=0)
        for _ in range(2):
            front = list({collector.state_key(s): s for p in front for s in collector.successors(p)}.values())
        for card in self.cards[:40]:
            affordable = any(collector.payments(p, card_info(card)) for p in front)
            self.assertEqual(affordable, card.points == 0, card.id)

    def test_bonus_chain_keeps_bonuses_but_spends_tokens(self):
        cards = [card_info(self.cards[i]) for i in (57, 83, 85)]
        planner = RoutePlanner(cards, [])
        p = position((0, 0, 3, 0, 0, 0), (0, 0, 3, 0, 0), 3)
        first = next(s for s in planner.successors(p) if s.path[-1].card_key == cards[0].key
                     and s.path[-1].kind == 'buy')
        self.assertEqual(first.gems[2], 0)
        self.assertEqual(first.bank[2], 4)
        self.assertEqual(first.bonuses[2], 4)
        self.assertEqual(planner.payments(first, cards[1]), [])
        self.assertEqual(cards[1].cost[2] - first.bonuses[2], 3)

    def test_one_noble_per_turn(self):
        nobles = [NobleInfo((1, 0, 0, 0, 0)), NobleInfo((0, 1, 0, 0, 0))]
        planner = RoutePlanner([], nobles)
        p = position(bonuses=(1, 1, 0, 0, 0), nobles=nobles)
        for nxt in planner.successors(p):
            self.assertEqual(nxt.points, 3)
            self.assertEqual(len(nxt.nobles), 1)

    def test_overflow_conserves_bank_and_respects_cap(self):
        planner = RoutePlanner([], [])
        p = position((2, 2, 2, 2, 2, 0))
        for nxt in planner.successors(p):
            self.assertEqual(sum(nxt.gems), 10)
            self.assertEqual(tuple(a + b for a, b in zip(nxt.bank, nxt.gems)), (4, 4, 4, 4, 4, 5))

    def test_reservations_do_not_award_points_or_bonus(self):
        card = card_info(self.cards[83])
        planner = RoutePlanner([card], [])
        p = position(count=1)
        reserved = next(s for s in planner.successors(p) if s.path[-1].kind == 'reserve')
        self.assertEqual((reserved.points, reserved.bonuses), (0, (0,) * 5))
        self.assertEqual((reserved.gems[5], reserved.bank[5]), (1, 4))
        self.assertEqual(reserved.visible, ())
        self.assertEqual(reserved.reserved, (0,))

    def test_reserve_limit_counts_unknown_cards(self):
        from dataclasses import replace
        planner = RoutePlanner([card_info(self.cards[0])], [])
        p = replace(position(count=1), unknown_reserves=3)
        self.assertFalse(any(s.path[-1].kind == 'reserve' for s in planner.successors(p)))

    def test_plan_replays_in_reference_environment(self):
        rng = random.Random(41)
        for _ in range(5):
            state = State()
            state.visible_cards[1] = rng.sample(self.cards[:40], 4)
            state.visible_cards[2] = rng.sample(self.cards[40:70], 2)
            state.nobles = rng.sample(self.nobles, 3)
            agent = HeuristicAgent16(num_rollouts=0, planning_horizon=4, beam_width=4)
            planner, p = agent._planner(state, 0, 0)
            plan = planner.search(p)
            env = MiniEnv()
            for move in plan['path']:
                legal = env._legal_actions(state)
                matching = [a for a in legal if agent._matches(state, a, move)]
                if move.kind == 'buy':
                    matching = [a for a in matching if a.payment == move.tokens]
                self.assertTrue(matching, move)
                env.step(matching[0], state)
                while state.node_type != NodeType.MAIN_DECISION:
                    env.step(env._legal_actions(state)[0], state)
                state.current_player = 0  # Planner's stated no-opponent approximation.


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.cards, self.nobles = definitions()
        self.env = MiniEnv()

    def agent(self, **kwargs):
        defaults = dict(num_rollouts=0, planning_horizon=2, beam_width=3,
                        planning_worlds=1, random_seed=12, num_calc_moves=5)
        defaults.update(kwargs)
        return HeuristicAgent16(**defaults)

    def state(self):
        s = State()
        for tier, lo, hi in ((1, 0, 40), (2, 40, 70), (3, 70, 90)):
            s.visible_cards[tier] = copy.deepcopy(self.cards[lo:lo + 4])
            s.decks[tier] = copy.deepcopy(self.cards[lo + 4:hi])
        s.nobles = copy.deepcopy(self.nobles[:3])
        return s

    def test_select_returns_legal_action_and_does_not_mutate_inputs(self):
        state = self.state()
        before = copy.deepcopy(state)
        action = self.agent().select_action(self.env, state)
        self.assertIn(action, self.env._legal_actions(state))
        self.assertEqual(state, before)
        self.assertEqual(self.env.ids, {})

    def test_reaching_fifteen_is_not_an_immediate_win(self):
        state = State()
        state.players[0].points = 14
        state.players[1].points = 14
        a = Card(100, 1, 1, C[0], dict.fromkeys(C, 0))
        b = Card(101, 2, 2, C[1], dict.fromkeys(C, 0))
        state.visible_cards[1] = [a]
        state.visible_cards[2] = [b]
        action = next(a for a in self.env._legal_actions(state) if a.action_type == ActionType.BUY_VISIBLE and a.tier == 1)
        value, certain = self.agent()._response_value(self.env, state, action, 0)
        self.assertEqual((value, certain), (-1000, True))

    def test_last_seat_takes_confirmed_win(self):
        state = State(current_player=1)
        state.players[1].points = 14
        state.visible_cards[1] = [Card(100, 1, 1, C[0], dict.fromkeys(C, 0))]
        action = self.agent().select_action(self.env, state)
        self.assertEqual(action.action_type, ActionType.BUY_VISIBLE)
        self.env.step(action, state)
        self.assertEqual(state.winners, [1])

    def test_hidden_reservations_resampled_not_used_as_public_targets(self):
        state = self.state()
        state.players[1].reserved_cards.append(state.decks[3].pop())
        original = card_info(state.players[1].reserved_cards[0]).key
        agent = self.agent()
        planner, p = agent._planner(state, 1, 0)
        self.assertEqual(p.reserved, ())
        self.assertEqual(p.unknown_reserves, 1)
        worlds = [agent._sample_world(state, 0) for _ in range(12)]
        self.assertTrue(any(card_info(w.players[1].reserved_cards[0]).key != original for w in worlds))
        for w in worlds:
            all_keys = sorted(card_info(c).key for c in w.decks[3] + w.players[1].reserved_cards)
            self.assertEqual(all_keys, sorted(card_info(c).key for c in state.decks[3] + state.players[1].reserved_cards))

    def test_known_public_reservation_preserved(self):
        state = self.state()
        card = state.decks[3].pop()
        state.players[1].reserved_cards.append(card)
        agent = self.agent(known_reserved_cards=lambda s, observer, owner: [card])
        for _ in range(4):
            self.assertEqual(agent._sample_world(state, 0).players[1].reserved_cards[0], card)

    def test_hidden_winning_card_is_only_a_sampled_threat(self):
        state = State()
        state.players[1].points = 14
        state.players[1].reserved_cards = [Card(100, 2, 2, C[0], dict.fromkeys(C, 0))]
        action = self.env._legal_actions(state)[0]
        value, certain = self.agent()._response_value(self.env, state, action, 0)
        self.assertEqual(value, -1000)
        self.assertFalse(certain)

    def test_noble_node_awards_only_one_noble(self):
        state = State(node_type=NodeType.NOBLE_CLAIM)
        state.players[0].bonuses = dict.fromkeys(C, 4)
        state.nobles = copy.deepcopy(self.nobles[:2])
        action = self.agent().select_action(self.env, state)
        self.assertEqual(action.action_type, ActionType.TAKE_NOBLE)
        self.env.step(action, state)
        self.assertEqual(state.players[0].points, 3)
        self.assertEqual(sum(n is not None for n in state.nobles), 1)

    def test_forced_discard_returns_one_action_and_does_not_switch_owner_early(self):
        state = self.state()
        state.players[0].gems = dict(zip(AC, (3, 2, 2, 2, 2, 0)))
        state.bank = {c: (5 if c == GemColor.GOLD else 4) - state.players[0].gems[c] for c in AC}
        state.node_type = NodeType.OVERFLOW_DISCARD
        action = self.agent().select_action(self.env, state)
        self.assertEqual(action.action_type, ActionType.DISCARD_GEMS)
        self.assertEqual(state.current_player, 0)
        self.env.step(action, state)
        self.assertEqual(sum(state.players[0].gems.values()), 10)

    def test_rollout_cutoff_is_reported(self):
        agent = self.agent(num_rollouts=1, max_rollout_steps=1, rollout_policy=FixturePolicy())
        action = agent.select_action(self.env, self.state())
        self.assertIsNotNone(action)
        self.assertGreater(agent.get_diagnostic_stats()['truncated_rollouts'], 0)

    def test_full_rollout_framework_with_injected_policy(self):
        state = self.state()
        agent = self.agent(num_rollouts=2, max_rollout_steps=200, num_calc_moves=2,
                           rollout_policy=FixturePolicy())
        action = agent.select_action(self.env, state)
        self.assertIn(action, self.env._legal_actions(state))
        self.assertGreater(agent.get_diagnostic_stats()['rollouts'], 0)
        records = agent.get_rollout_debug()['evaluated']
        self.assertTrue(all(-1 <= r['value'] <= 1 for r in records))
        self.assertTrue(any(r['truncated'] < r['rollout_count'] for r in records))

    def test_seeded_decisions_repeat(self):
        state = self.state()
        one, two = self.agent(), self.agent()
        self.assertEqual(one.select_action(self.env, state), two.select_action(self.env, state))

    def test_get_policy_legal_normalized(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy unavailable in this interpreter')
        state = self.state()
        legal_ids = {self.env.action_to_id(a) for a in self.env._legal_actions(state)}
        policy = self.agent().get_policy(self.env, state)
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
        self.assertTrue(set(np.flatnonzero(policy)).issubset(legal_ids))

    def test_terminal_state_and_multiplayer_guard(self):
        state = State(winners=[0])
        self.assertIsNone(self.agent().select_action(self.env, state))
        state = State(players=[Player(), Player(), Player()])
        with self.assertRaises(ValueError):
            self.agent().select_action(self.env, state)

    def test_empty_opponent_actions_cut_off_response(self):
        class DeadOpponent(MiniEnv):
            def _legal_actions(self, state):
                return [] if state.current_player == 1 else super()._legal_actions(state)
        env, state, agent = DeadOpponent(), State(), self.agent()
        action = env._legal_actions(state)[0]
        value, certain = agent._response_value(env, state, action, 0)
        self.assertFalse(certain)
        self.assertLessEqual(abs(value), 5 * agent.rollout_cutoff_weight)
        self.assertEqual(agent.get_diagnostic_stats()['no_legal_response'], 1)
        self.assertIsNotNone(agent.select_action(env, state))

    def test_empty_rollout_actions_do_not_call_continuation(self):
        class DeadOpponent(MiniEnv):
            def _legal_actions(self, state):
                return [] if state.current_player == 1 else super()._legal_actions(state)
        class NeverCalled:
            def select_action(self, env, state):
                raise AssertionError('Policy should not be called without legal actions')
        env, state = DeadOpponent(), State()
        agent = self.agent(rollout_policy=NeverCalled())
        value, cutoff = agent._rollout(env, state, env._legal_actions(state)[0], 0, 1)
        self.assertTrue(cutoff)
        self.assertLessEqual(abs(value), agent.rollout_cutoff_weight)
        self.assertEqual(agent.get_diagnostic_stats()['no_legal_rollout'], 1)

    def test_empty_forced_and_root_actions_do_not_invent_a_turn(self):
        class EmptyEnv(MiniEnv):
            def _legal_actions(self, state):
                return []
        env, agent = EmptyEnv(), self.agent()
        state = State(node_type=NodeType.NOBLE_CLAIM)
        self.assertIs(agent._complete_turn(env, state, 0), state)
        self.assertEqual(state.current_player, 0)
        self.assertEqual(state.node_type, NodeType.NOBLE_CLAIM)
        self.assertIsNone(agent.select_action(env, state))
        self.assertEqual(agent.get_diagnostic_stats()['no_legal_root'], 1)


if __name__ == '__main__':
    unittest.main()
