"""HeuristicAgent16: resource-conserving routes plus two-player response search.

Drop into splendor_v1/agents/heuristic_agent_16.py. Requires only the engine
interfaces used by H15; H15 itself is not a dependency. H3 is imported lazily
only when terminal rollouts are enabled without a supplied rollout policy.

Planning is bounded and approximate: future market refills and opponent turns
are omitted inside each player's local route search. Root/response transitions
use the real environment and sampled hidden worlds. No full-deck color symmetry
is assumed. See README.md for controls, integration, and known limitations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations, product
from math import exp, isfinite, tanh
import copy
import random

from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import GemColor, NodeType
from splendor_v1.env.core.actions import ActionType


COLORS = tuple(c for c in COLOR_ORDER if c != GemColor.GOLD)
ALL_COLORS = COLORS + (GemColor.GOLD,)
if len(COLORS) != 5:
    raise ValueError("H16 supports the five-color base game only")


@dataclass(frozen=True)
class CardInfo:
    key: tuple
    tier: int
    points: int
    bonus: int
    cost: tuple


@dataclass(frozen=True)
class NobleInfo:
    requirement: tuple
    points: int = 3


@dataclass(frozen=True)
class PlanMove:
    kind: str
    card_key: tuple = ()
    tokens: tuple = ()  # Actual payment for buy; retained gain is NOT assumed.


@dataclass(frozen=True)
class Position:
    gems: tuple
    bonuses: tuple
    bank: tuple
    visible: tuple  # Indices into the planner's card list.
    reserved: tuple
    nobles: tuple  # Indices into the planner's noble list.
    points: int
    bought: int = 0
    unknown_reserves: int = 0
    path: tuple = ()


def card_info(card):
    cost = tuple(int(card.cost.get(c, 0)) for c in COLORS)
    tier = int(getattr(card, "tier", getattr(card, "level", 0)))
    bonus = COLORS.index(card.bonus_color)
    # Base cards are unique by these game properties. IDs/names do not affect value.
    return CardInfo((tier, int(card.points), bonus, cost), tier,
                    int(card.points), bonus, cost)


def card_type(card):
    """Return an exact family plus structural tags; no strategic tier weights."""
    c = card if isinstance(card, CardInfo) else card_info(card)
    shape = tuple(sorted((n for n in c.cost if n), reverse=True))
    self_cost = c.cost[c.bonus]
    if not self_cost:
        feedback = "different_color"
    elif self_cost == max(c.cost):
        feedback = "main_color"
    else:
        feedback = "support_color"
    return {"family": (c.tier, c.points, shape), "support": len(shape),
            "peak": max(c.cost), "self_cost": self_cost, "feedback": feedback}


def noble_bundle_requirements(nobles, bonuses):
    """All nonempty bundles; missing counts are purchase lower bounds, not turns."""
    result = []
    for size in range(1, len(nobles) + 1):
        for ids in combinations(range(len(nobles)), size):
            req = tuple(max(nobles[i].requirement[k] for i in ids) for k in range(5))
            missing = tuple(max(0, req[k] - bonuses[k]) for k in range(5))
            result.append({"nobles": ids, "requirement": req,
                           "missing": missing, "missing_purchases": sum(missing),
                           "points": sum(nobles[i].points for i in ids)})
    return result


class RoutePlanner:
    """Own-turn beam search over actual token collections, buys and reservations.

    Every generated route obeys token supply, cap, payments, the reserve limit,
    reusable bonuses and one noble per turn. It is executable only conditional
    on no opponent interference and ignoring unknown replacement cards.
    Width/payment/discard pruning sacrifices optimality, never legality.
    """

    def __init__(self, cards, nobles, horizon=4, beam_width=8, payment_width=3,
                 reserve_width=2, discard_width=2, max_gems=10):
        self.cards = tuple(cards)
        self.nobles = tuple(nobles)
        self.horizon = horizon
        self.beam_width = beam_width
        self.payment_width = payment_width
        self.reserve_width = reserve_width
        self.discard_width = discard_width
        self.max_gems = max_gems
        self._eval_cache = {}

    @staticmethod
    def state_key(p):
        return (p.gems, p.bonuses, p.bank, p.visible, p.reserved, p.nobles,
                p.points, p.bought, p.unknown_reserves)

    def deficit(self, p, card):
        return max(0, sum(max(0, card.cost[k] - p.bonuses[k] - p.gems[k])
                          for k in range(5)) - p.gems[5])

    def _stock_quality(self, p):
        """Alternative options, not a sum of fictional simultaneous purchases."""
        values = []
        for i in p.visible + p.reserved:
            c = self.cards[i]
            deficit = self.deficit(p, c)
            # Approximate tie-breaker only. Exact action sequences determine routes.
            value = (c.points + 0.6) / (1.0 + deficit)
            values.append(value)
        values.sort(reverse=True)
        return (values[0] if values else 0.0) + (0.12 * values[1] if len(values) > 1 else 0.0)

    def potential(self, p):
        key = self.state_key(p)
        if key in self._eval_cache:
            return self._eval_cache[key]
        live = [self.cards[i] for i in p.visible + p.reserved]
        # Bonus demand from remaining useful purchases, with saturation at cost.
        # Best few alternatives are weak residual potential, not booked points.
        coverage = sorted((sum(min(p.bonuses[k], c.cost[k]) for k in range(5))
                           / max(1, sum(c.cost)) * (0.5 + 0.15 * c.points)
                           for c in live), reverse=True)
        engine = sum(coverage[:3]) / 3.0
        remaining = [self.nobles[i] for i in p.nobles]
        bundle_value = 0.0
        for bundle in noble_bundle_requirements(remaining, p.bonuses):
            # Missing cards must be obtainable from this frozen market. A distant
            # impossible bundle must not manufacture progress value.
            supply = Counter(c.bonus for c in live)
            if any(bundle["missing"][k] > supply[k] for k in range(5)):
                continue
            distance = bundle["missing_purchases"]
            # Bundles are alternatives, not additive noble-progress rewards.
            claim_turns = len(bundle["nobles"])
            candidate = bundle["points"] / (1.0 + max(distance, claim_turns))
            bundle_value = max(bundle_value, candidate)
        # Engine value tapers as the player approaches the finish. Opponent urgency
        # is handled by the shorter planning horizon and root response search.
        phase = min(1.0, max(0.0, (15 - p.points) / 8.0))
        value = (p.points + 0.65 * self._stock_quality(p)
                 + phase * (0.65 * engine + 0.65 * bundle_value))
        self._eval_cache[key] = value
        return value

    def payments(self, p, c):
        required = tuple(max(0, c.cost[k] - p.bonuses[k]) for k in range(5))
        options = []

        def visit(k, payment, gold_left):
            if k == 5:
                options.append(tuple(payment) + (p.gems[5] - gold_left,))
                return
            lo = max(0, required[k] - gold_left)
            hi = min(required[k], p.gems[k])
            for ordinary in range(hi, lo - 1, -1):
                visit(k + 1, payment + [ordinary], gold_left - required[k] + ordinary)

        visit(0, [], p.gems[5])
        if len(options) <= self.payment_width:
            return options
        # Keep the gold-preserving payment and alternatives with useful leftovers.
        ranked = sorted(options, key=lambda pay: self._stock_quality(replace(
            p, gems=tuple(g - x for g, x in zip(p.gems, pay)))), reverse=True)
        best_gold = min(options, key=lambda pay: pay[5])
        return [best_gold] + [pay for pay in ranked if pay != best_gold][:self.payment_width - 1]

    def _trim(self, p):
        excess = sum(p.gems) - self.max_gems
        if excess <= 0:
            return [p]
        candidates = []
        # Excess is <= 3 in base-game main actions, so enumeration is small.
        for discarded in product(*(range(min(g, excess) + 1) for g in p.gems)):
            if sum(discarded) != excess:
                continue
            candidates.append(replace(p,
                gems=tuple(g - d for g, d in zip(p.gems, discarded)),
                bank=tuple(b + d for b, d in zip(p.bank, discarded))))
        candidates.sort(key=lambda x: (self._stock_quality(x), x.gems[5]), reverse=True)
        return candidates[:self.discard_width]

    def _claim(self, p):
        eligible = [i for i in p.nobles if all(p.bonuses[k] >= self.nobles[i].requirement[k]
                                              for k in range(5))]
        if not eligible:
            return [p]
        return [replace(p, points=p.points + self.nobles[i].points,
                        nobles=tuple(j for j in p.nobles if j != i)) for i in eligible]

    def successors(self, p):
        raw = []
        for i in p.visible + p.reserved:
            card = self.cards[i]
            for payment in self.payments(p, card):
                bonuses = list(p.bonuses)
                bonuses[card.bonus] += 1
                raw.append(replace(p,
                    gems=tuple(g - n for g, n in zip(p.gems, payment)),
                    bank=tuple(b + n for b, n in zip(p.bank, payment)),
                    bonuses=tuple(bonuses), points=p.points + card.points,
                    visible=tuple(j for j in p.visible if j != i),
                    reserved=tuple(j for j in p.reserved if j != i), bought=p.bought + 1,
                    path=p.path + (PlanMove("buy", card.key, payment),)))
        available = [k for k in range(5) if p.bank[k] > 0]
        take_sets = list(combinations(available, min(3, len(available)))) if available else []
        take_sets += [(k, k) for k in range(5) if p.bank[k] >= 4]
        for taken in take_sets:
            gain = tuple(taken.count(k) for k in range(6))
            raw.append(replace(p,
                gems=tuple(g + n for g, n in zip(p.gems, gain)),
                bank=tuple(b - n for b, n in zip(p.bank, gain)),
                path=p.path + (PlanMove("take", tokens=gain),)))
        if len(p.reserved) + p.unknown_reserves < 3:
            targets = sorted(p.visible, key=lambda i: (
                (self.cards[i].points + 0.5) / (1.0 + self.deficit(p, self.cards[i]))), reverse=True)
            for i in targets[:self.reserve_width]:
                gold = int(p.bank[5] > 0)
                raw.append(replace(p, gems=p.gems[:5] + (p.gems[5] + gold,),
                    bank=p.bank[:5] + (p.bank[5] - gold,),
                    visible=tuple(j for j in p.visible if j != i), reserved=p.reserved + (i,),
                    path=p.path + (PlanMove("reserve", self.cards[i].key),)))
        for candidate in raw:
            for trimmed in self._trim(candidate):
                yield from self._claim(trimmed)

    def search(self, start):
        beam = [start]
        best = start
        best_score = self.potential(start)
        frontier = []
        finish_turn = None
        expanded = 0
        # Keep several strategic directions alive instead of pruning every
        # infrastructure/noble route behind a short-term token accumulation route.
        bundles = noble_bundle_requirements([self.nobles[i] for i in start.nobles], start.bonuses)
        goals = []
        for size in range(1, min(3, len(start.nobles)) + 1):
            group = [b for b in bundles if len(b["nobles"]) == size]
            if group:
                goals.append(max(group, key=lambda b: b["points"] / (1 + b["missing_purchases"]))["requirement"])
        for turn in range(1, self.horizon + 1):
            states = {}
            for p in beam:
                for nxt in self.successors(p):
                    expanded += 1
                    key = self.state_key(nxt)
                    states.setdefault(key, nxt)
            if not states:
                break
            ranked = sorted(states.values(), key=lambda p: (
                p.points >= 15, self.potential(p), -p.bought), reverse=True)
            frontier.append({"turn": turn, "points": max(p.points for p in ranked)})
            # Do not continue a local route past its first finish trigger. The real
            # environment alone decides who wins after the equal-turn final round.
            for p in ranked:
                score = start.points + (self.potential(p) - start.points) * (0.94 ** turn)
                if p.points >= 15:
                    finish_turn = turn if finish_turn is None else min(finish_turn, turn)
                    score += 1.5 / turn
                if score > best_score:
                    best_score, best = score, p
            unfinished = [p for p in ranked if p.points < 15]
            beam = unfinished[:max(1, self.beam_width // 2)]
            if unfinished:
                seeds = [max(unfinished, key=lambda p: (p.points, self.potential(p))),
                         max(unfinished, key=lambda p: (sum(p.bonuses), self.potential(p)))]
                for req in goals:
                    seeds.append(max(unfinished, key=lambda p: (
                        -sum(max(0, req[k] - p.bonuses[k]) for k in range(5)), self.potential(p))))
                kept = {self.state_key(p) for p in beam}
                for p in seeds + unfinished:
                    if len(beam) >= self.beam_width:
                        break
                    if self.state_key(p) not in kept:
                        kept.add(self.state_key(p))
                        beam.append(p)
            if not beam:
                break
        return {"value": best_score, "path": best.path, "points": best.points,
                "bonuses": best.bonuses, "gems": best.gems, "finish_turn": finish_turn,
                "frontier": frontier, "expanded": expanded}


class HeuristicAgent16:
    """H15-compatible agent, optimized for two-player base Splendor.

    num_rollouts=0 runs route/response search only. With positive rollouts, H3
    supplies continuations unless rollout_policy is provided. Their mean result
    is combined with a small route-value prior; neither mode is proven stronger.

    known_reserved_cards(state, observer, owner) may return opponent cards that
    observer legitimately knows. Default: own reservations known, opponent
    reservations unknown. This conservative default resamples the latter with
    same-tier hidden deck cards rather than reading private identities.
    """

    def __init__(self, num_rollouts=16, num_calc_moves=8, max_rollout_steps=200,
                 random_seed=None, planning_horizon=4, beam_width=8,
                 planning_worlds=2, reply_width=1, payment_width=3,
                 prior_weight=2.0, rollout_policy=None, known_reserved_cards=None,
                 rollout_cutoff_weight=0.25, name=None):
        self.name = name
        for name, value in (("num_calc_moves", num_calc_moves),
                            ("max_rollout_steps", max_rollout_steps),
                            ("planning_horizon", planning_horizon), ("beam_width", beam_width),
                            ("planning_worlds", planning_worlds), ("reply_width", reply_width),
                            ("payment_width", payment_width)):
            if int(value) < 1:
                raise ValueError(name + " must be >= 1")
            setattr(self, name, int(value))
        if num_rollouts < 0 or prior_weight < 0 or not 0 <= rollout_cutoff_weight <= 1:
            raise ValueError("Invalid rollout/prior settings")
        self.num_rollouts = int(num_rollouts)
        self.prior_weight = float(prior_weight)
        self.rollout_cutoff_weight = float(rollout_cutoff_weight)
        self.rng = random.Random(random_seed)
        self.rollout_policy = rollout_policy
        self.known_reserved_cards = known_reserved_cards
        if self.num_rollouts and self.rollout_policy is None:
            from splendor_v1.agents.heuristic_agent_3 import HeuristicAgent3
            self.rollout_policy = HeuristicAgent3()
        self.last_board_model = None
        self.last_candidate_debug = None
        self.last_rollout_debug = None
        self.reset_diagnostic_stats()

    def reset_diagnostic_stats(self):
        self._stats = Counter()
        self._plan_cache = {}

    @staticmethod
    def _clone(obj):
        return obj.clone() if hasattr(obj, "clone") else copy.deepcopy(obj)

    @staticmethod
    def _step(env, state, action):
        result = env.step(action, state)
        if hasattr(result, "players"):
            return result
        if isinstance(result, tuple):
            for item in result:
                if hasattr(item, "players"):
                    return item
        return state

    @staticmethod
    def _winner_value(state, player):
        winners = getattr(state, "winners", None)
        if winners is None:
            winner = getattr(state, "winner", None)
            winners = [] if winner is None else [winner]
        if not winners or len(winners) > 1:
            return 0.0
        return 1.0 if player in winners else -1.0

    def _known(self, state, observer, owner):
        cards = [c for c in state.players[owner].reserved_cards if c is not None]
        if observer == owner:
            return cards
        if self.known_reserved_cards is None:
            return []
        return list(self.known_reserved_cards(state, observer, owner))

    def _planner(self, state, player_index, observer):
        player = state.players[player_index]
        visible = [c for tier in (1, 2, 3) for c in state.visible_cards[tier] if c is not None]
        reserved = self._known(state, observer, player_index)
        cards = [card_info(c) for c in visible + reserved]
        nobles = [NobleInfo(tuple(n.requirement.get(c, 0) for c in COLORS), int(n.points))
                  for n in state.nobles if n is not None]
        p = Position(tuple(int(player.gems.get(c, 0)) for c in ALL_COLORS),
                     tuple(int(player.bonuses.get(c, 0)) for c in COLORS),
                     tuple(int(state.bank.get(c, 0)) for c in ALL_COLORS),
                     tuple(range(len(visible))), tuple(range(len(visible), len(cards))),
                     tuple(range(len(nobles))), int(player.points),
                     unknown_reserves=len(player.reserved_cards) - len(reserved))
        horizon = self.planning_horizon
        # A near-finishing opponent makes long investments less attractive.
        if max(q.points for i, q in enumerate(state.players) if i != player_index) >= 12:
            horizon = min(horizon, 3)
        planner = RoutePlanner(cards, nobles, horizon, self.beam_width, self.payment_width,
                               max_gems=int(getattr(state, "max_gems", 10)))
        return planner, p

    def _player_model(self, state, owner, observer, detailed):
        planner, p = self._planner(state, owner, observer)
        if not detailed:
            return {"value": planner.potential(p), "path": ()}
        key = (planner.cards, planner.nobles, planner.state_key(p), planner.horizon)
        if key not in self._plan_cache:
            self._plan_cache[key] = planner.search(p)
        return self._plan_cache[key]

    def _evaluate(self, env, state, root, detailed=False):
        if env._check_terminated(state):
            return 1000.0 * self._winner_value(state, root)
        if detailed and not env._legal_actions(state):
            return self._no_legal_cutoff(env, state, root, "leaf")
        ours = self._player_model(state, root, root, detailed)["value"]
        theirs = self._player_model(state, 1 - root, root, detailed)["value"]
        return ours - theirs

    def _no_legal_cutoff(self, env, state, root, context):
        """Stop a dead-end simulation without inventing a pass, winner or draw."""
        self._stats["no_legal_" + context] += 1
        return 5.0 * self.rollout_cutoff_weight * tanh(
            self._evaluate(env, state, root) / 5.0)

    def _complete_turn(self, env, state, owner):
        """Resolve discard/noble nodes without treating them as opponent turns."""
        for _ in range(32):
            if env._check_terminated(state) or state.node_type == NodeType.MAIN_DECISION:
                return state
            legal = list(env._legal_actions(state))
            if not legal:
                self._stats["no_legal_forced"] += 1
                return state  # Leave this unresolved; the caller cuts off the branch.
            choices = []
            for action in legal:
                child = self._step(env, self._clone(state), action)
                choices.append((self._evaluate(env, child, owner), child))
            state = max(choices, key=lambda x: x[0])[1]
        raise RuntimeError("Forced-node chain exceeded 32 transitions")

    def _after(self, env, state, action, owner):
        return self._complete_turn(env, self._step(env, self._clone(state), action), owner)

    def _sample_world(self, state, observer):
        sampled = self._clone(state)
        hidden_by_tier = {tier: [] for tier in (1, 2, 3)}
        for owner, player in enumerate(sampled.players):
            known = Counter(card_info(c).key for c in self._known(state, observer, owner))
            for slot, card in enumerate(player.reserved_cards):
                if card is None:
                    continue
                key = card_info(card).key
                if known[key]:
                    known[key] -= 1
                else:
                    hidden_by_tier[card_info(card).tier].append((owner, slot, card))
        # H15 accepts dict decks or a sequence. Tier order for sequences is 1,2,3.
        for tier in (1, 2, 3):
            deck_index = tier if isinstance(sampled.decks, dict) else tier - 1
            pool = list(sampled.decks[deck_index]) + [x[2] for x in hidden_by_tier[tier]]
            self.rng.shuffle(pool)
            for owner, slot, _ in hidden_by_tier[tier]:
                sampled.players[owner].reserved_cards[slot] = pool.pop()
            sampled.decks[deck_index] = pool
        return sampled

    @staticmethod
    def _action_card(state, action):
        if action.action_type in (ActionType.BUY_VISIBLE, ActionType.RESERVE_VISIBLE):
            return state.visible_cards[action.tier][action.slot]
        if action.action_type == ActionType.BUY_RESERVED:
            return state.players[state.current_player].reserved_cards[action.reserved_index]
        return None

    @staticmethod
    def _group(action):
        if action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
            return "buy"
        if action.action_type == ActionType.TAKE_GEMS:
            return "take"
        if action.action_type in (ActionType.RESERVE_VISIBLE, ActionType.RESERVE_TOP_DECK):
            return "reserve"
        return "forced"

    def _matches(self, state, action, move):
        if self._group(action) != move.kind:
            return False
        if move.kind == "take":
            taken = Counter(action.gem_colors)
            return tuple(taken[c] for c in ALL_COLORS) == move.tokens
        card = self._action_card(state, action)
        return card is not None and card_info(card).key == move.card_key

    def _root_candidates(self, env, state, worlds):
        root = state.current_player
        legal = list(env._legal_actions(state))
        if state.node_type != NodeType.MAIN_DECISION:
            return legal
        ours = self._player_model(state, root, root, True)
        theirs = self._player_model(state, 1 - root, root, True)
        structure = []
        for owner in (root, 1 - root):
            planner, p = self._planner(state, owner, root)
            structure.append({"player": owner,
                "card_types": [{"card_key": c.key, **card_type(c)} for c in planner.cards],
                "noble_bundles": noble_bundle_requirements(planner.nobles, p.bonuses)})
        self.last_board_model = {"us": ours, "opponent": theirs, "structure": structure}
        records = []
        for action in legal:
            children = [self._after(env, world, action, root) for world in worlds]
            score = sum(self._evaluate(env, s, root) for s in children) / len(children)
            tags = []
            if all(env._check_terminated(s) and self._winner_value(s, root) == 1 for s in children):
                tags.append("terminal_win")
            if any(s.players[root].points >= 15 for s in children):
                tags.append("finish_trigger")  # Never confused with a confirmed win.
            if ours["path"] and self._matches(state, action, ours["path"][0]):
                tags.append("route_start")
            card = self._action_card(state, action)
            if card is not None and theirs["path"]:
                targets = {m.card_key for m in theirs["path"] if m.card_key}
                if card_info(card).key in targets:
                    tags.append("contested_route")
            records.append({"action": action, "score": score, "tags": tags,
                            "group": self._group(action)})
        records.sort(key=lambda r: r["score"], reverse=True)
        selected = []
        seen = set()

        def add(record):
            key = env.action_to_id(record["action"])
            if key not in seen:
                seen.add(key)
                selected.append(record)

        # Confirmed wins can exceed the budget. A mere finish trigger receives
        # one seed, so losing ways to reach 15 cannot crowd out defensive moves.
        for r in records:
            if "terminal_win" in r["tags"]:
                add(r)
        for tag in ("finish_trigger", "route_start", "contested_route"):
            if len(selected) < self.num_calc_moves:
                candidate = next((r for r in records if tag in r["tags"]), None)
                if candidate:
                    add(candidate)
        for group in ("buy", "take", "reserve"):
            if len(selected) < self.num_calc_moves:
                candidate = next((r for r in records if r["group"] == group), None)
                if candidate:
                    add(candidate)
        for r in records:
            if len(selected) >= self.num_calc_moves:
                break
            add(r)
        self.last_candidate_debug = {"legal_count": len(legal), "selected_records": selected}
        return [r["action"] for r in selected]

    def _response_value(self, env, state, action, root):
        child = self._after(env, state, action, root)
        if env._check_terminated(child):
            return self._evaluate(env, child, root), True
        if child.node_type != NodeType.MAIN_DECISION:
            return self._no_legal_cutoff(env, child, root, "response"), False
        if child.current_player == root:
            return self._evaluate(env, child, root, True), False
        replies = []
        for reply in env._legal_actions(child):
            after = self._after(env, child, reply, child.current_player)
            score = self._evaluate(env, after, root)
            replies.append((score, after, reply))
        if not replies:
            return self._no_legal_cutoff(env, child, root, "response"), False
        replies.sort(key=lambda x: x[0])
        # A terminal loss available to the opponent must never be pruned.
        if replies[0][0] == -1000.0:
            # A winning purchase of an unknown refill/private reserve is only
            # a sampled threat. Do not turn it into a proven tactical result.
            public_cards = {card_info(c).key for tier in (1, 2, 3)
                            for c in state.visible_cards[tier] if c is not None}
            public_cards.update(card_info(c).key for c in self._known(state, root, 1 - root))
            for score, _, reply in replies:
                if score != -1000.0:
                    break
                target = self._action_card(child, reply)
                if target is None or card_info(target).key in public_cards:
                    return -1000.0, True
            return -1000.0, False
        values = [self._evaluate(env, s, root, True) for _, s, _ in replies[:self.reply_width]]
        return min(values), False

    def _rollout(self, env, sampled, first, root, seed):
        # Clone the continuation policy so candidate order cannot change its state.
        policy = copy.deepcopy(self.rollout_policy)
        if isinstance(getattr(policy, "rng", None), random.Random):
            policy.rng.seed(seed)
        state = self._step(env, self._clone(sampled), first)
        for _ in range(self.max_rollout_steps - 1):
            if env._check_terminated(state):
                return self._winner_value(state, root), False
            if not env._legal_actions(state):
                return self._no_legal_cutoff(env, state, root, "rollout") / 5.0, True
            action = policy.select_action(env, state)
            if action is None:
                raise RuntimeError("Continuation policy returned None in a nonterminal state")
            state = self._step(env, state, action)
        if env._check_terminated(state):
            return self._winner_value(state, root), False
        # A truncated game is not falsely reported as a terminal draw.
        return self.rollout_cutoff_weight * tanh(self._evaluate(env, state, root) / 5.0), True

    def _get_rollout_evaluated_candidates(self, env, state):
        if len(state.players) != 2:
            raise ValueError("H16 currently supports exactly two players")
        self._plan_cache = {}
        self.last_board_model = self.last_candidate_debug = self.last_rollout_debug = None
        if env._check_terminated(state):
            return []
        root = state.current_player
        if not env._legal_actions(state):
            self._stats["no_legal_root"] += 1
            return []  # select_action -> None; get_policy -> zero distribution.
        count = max(self.planning_worlds, self.num_rollouts)
        worlds = [self._sample_world(state, root) for _ in range(count)]
        seeds = [self.rng.randrange(2**32) for _ in range(count)]
        sim_env = self._clone(env)
        if state.node_type != NodeType.MAIN_DECISION:
            # Return one actual forced action, not an entire completed turn.
            evaluated = []
            for action in sim_env._legal_actions(state):
                value = sum(self._evaluate(sim_env,
                    self._complete_turn(sim_env, self._step(sim_env, self._clone(w), action), root),
                    root, True) for w in worlds[:self.planning_worlds]) / self.planning_worlds
                evaluated.append((action, tanh(value / 5.0), value))
            self._stats["forced_decisions"] += 1
            return evaluated
        candidates = self._root_candidates(sim_env, state, worlds[:self.planning_worlds])
        evaluated, details = [], []
        for action in candidates:
            forecasts = [self._response_value(sim_env, w, action, root)
                         for w in worlds[:self.planning_worlds]]
            route_value = sum(v for v, _ in forecasts) / self.planning_worlds
            route_prior = sum(tanh(v / 5.0) for v, _ in forecasts) / self.planning_worlds
            # All-world immediate wins/losses found by the explicit response layer
            # outrank stochastic continuation estimates on this sampled belief.
            forced = all(certain and v == forecasts[0][0] for v, certain in forecasts)
            samples = []
            truncated = 0
            if not forced:
                for i in range(self.num_rollouts):
                    value, cutoff = self._rollout(self._clone(env), worlds[i], action, root, seeds[i])
                    samples.append(value)
                    truncated += int(cutoff)
            if forced:
                value = 1.0 if forecasts[0][0] > 0 else (-1.0 if forecasts[0][0] < 0 else 0.0)
            elif samples:
                value = (sum(samples) + self.prior_weight * route_prior) / (len(samples) + self.prior_weight)
            else:
                value = route_prior
            evaluated.append((action, value, route_value))
            details.append({"action_id": sim_env.action_to_id(action), "value": value,
                            "route_value": route_value, "rollout_count": len(samples),
                            "rollout_mean": sum(samples) / len(samples) if samples else None,
                            "truncated": truncated, "sampled_forced_result": forced})
            self._stats["rollouts"] += len(samples)
            self._stats["truncated_rollouts"] += truncated
        self.last_rollout_debug = {"evaluated": details, "planning_worlds": self.planning_worlds}
        self._stats["decisions"] += 1
        self._stats["candidates"] += len(candidates)
        return evaluated

    def select_action(self, env, state):
        evaluated = self._get_rollout_evaluated_candidates(env, state)
        if not evaluated:
            return None
        action = max(evaluated, key=lambda item: (item[1], item[2]))[0]
        self._stats["selected_" + self._group(action)] += 1
        return action

    def get_policy(self, env, state, action_size=1139, temperature=0.25):
        import numpy as np
        if temperature <= 0 or not isfinite(temperature):
            raise ValueError("temperature must be finite and > 0")
        evaluated = self._get_rollout_evaluated_candidates(env, state)
        policy = np.zeros(action_size, dtype=np.float32)
        if not evaluated:
            return policy
        peak = max(v for _, v, _ in evaluated)
        weights = [exp((v - peak) / temperature) for _, v, _ in evaluated]
        total = sum(weights)
        for (action, _, _), weight in zip(evaluated, weights):
            index = env.action_to_id(action)
            if not 0 <= index < action_size:
                raise ValueError("action_size does not cover engine action IDs")
            policy[index] += weight / total
        return policy

    def get_board_model_debug(self):
        return self.last_board_model

    def get_candidate_debug(self):
        return self.last_candidate_debug

    def get_rollout_debug(self):
        return self.last_rollout_debug

    def get_diagnostic_stats(self):
        return dict(self._stats)

    def format_diagnostic_summary(self):
        return "HeuristicAgent16\n" + "\n".join(f"{k}: {v}" for k, v in sorted(self._stats.items()))


class HeuristicAgent16Diagnostics(HeuristicAgent16):
    """Compatibility alias: diagnostics are already enabled on the base class."""
