# HeuristicAgent16

A standalone successor to the supplied H15, using executable short routes,
shared noble requirements, and explicit opponent replies. Designed for the
two-player base game. This is an experimental agent, not a demonstrated strength
improvement over H15.

## Install

Copy `heuristic_agent_16.py` into `splendor_v1/agents/` in your project.
It does not inherit from H15 and does not alter any game definitions.

```python
from splendor_v1.agents.heuristic_agent_16 import HeuristicAgent16

agent = HeuristicAgent16(
    num_rollouts=16,
    num_calc_moves=8,
    planning_horizon=4,
    beam_width=8,
    planning_worlds=2,
    random_seed=42,
)
action = agent.select_action(env, state)
policy = agent.get_policy(env, state, action_size=1139, temperature=0.25)
```

`get_policy` performs a fresh decision/search; it does not reuse the preceding
`select_action` computation. It returns a normalized NumPy float32 distribution
over shortlisted legal actions, or all zeros on a terminal state. NumPy is only
needed for this API.

For an initial fast integration check:

```python
agent = HeuristicAgent16(
    num_rollouts=0,       # Pure route + opponent-response evaluation; no H3 import.
    num_calc_moves=5,
    planning_horizon=3,
    beam_width=5,
    planning_worlds=1,
    random_seed=42,
)
```

The default positive rollout count imports the existing `HeuristicAgent3` as a
continuation policy, following H15's framework. Pass `rollout_policy=your_agent`
to substitute another cheap policy supporting `select_action(env, state)`.
Do not pass H16 itself as its own rollout policy: recursive planning is costly.

## What changed

1. **Route search conserves resources.** Purchases consume tokens, return them
   to the bank, and keep permanent bonuses. Different payment choices can leave
   different future options, including spending gold despite having the color.
2. **Every tier can contribute.** No mandatory primary Tier-3 strategy and no
   tier-specific multiplier on prestige points. Actual costs and produced
   bonuses determine sequential relationships. `card_type` identifies all 18
   families and main-color/support-color/different-color bonus feedback.
3. **Nobles are evaluated in groups.** Combined requirements use per-color
   maxima. The planner branches over eligible noble choices but claims only
   one per turn. Scoring, infrastructure, and noble-directed beam seeds preserve
   different potential routes.
4. **The same evaluator runs for both players.** Root actions change the market,
   bank, payments and target availability before both positions are evaluated.
   The opponent's best cheaply ranked replies receive detailed route analysis.
5. **Legal engine transitions decide tactics.** Reaching 15 is a finish trigger,
   not an automatic win. End-round resolution and purchased-card tiebreaks come
   from the real environment. An immediate public-information winning reply is
   not pruned by the response shortlist.
6. **H15's rollout scaffold remains optional.** Candidates share sampled hidden
   worlds. Continuation estimates get a small route-value prior, controlled by
   `prior_weight`. Cutoffs use a discounted heuristic estimate and are reported
   explicitly rather than silently labeled draws.

## Decision pipeline

- Build an own-turn route forecast for each player.
- Simulate all legal root actions with the engine, resolving forced choices.
- Shortlist finish triggers, route starts, contested targets, and diverse action
  categories, then fill by the cheap position score. Confirmed terminal winning
  actions may exceed the nominal candidate budget; mere finish triggers get
  only one seed.
- For each candidate and planning world, consider the opponent's legal replies.
  All immediate losses are inspected; `reply_width` limits detailed evaluation
  of nonterminal responses.
- Evaluate the resulting positions with both players' route planners.
- If enabled, run terminal continuations and combine their mean with the route
  prior. Select by combined value, breaking ties with the route evaluation.

## Important approximations

The local planner is an **own-turn, frozen-market beam search**, not a full game
solver. It omits opponent interference and unknown refills *inside* its forecast.
Root and reply transitions use the actual environment, so they do include
refills, bank returns, and blocking. Local route feasibility is conditional on
the frozen-market assumption. Further alternating search would be an extension.

The horizon counts the player's main turns, not forced discard/noble substeps.
The local planner stops extending a route after it first reaches 15. It does not
declare that route a win. Long investments beyond the horizon are represented
only by weak residual potential, and may be undervalued. Noble residuals do not
assume that currently unavailable bonus cards will appear.

Payment alternatives, discard choices, reservations, beam states, and root
actions are pruned. Therefore a valuable move can still be missed. The local
planner models visible reservations, not blind-reservation outcomes. The real
root action generator and rollouts still support blind reservations.

Points are the main planning objective; small affordability, bonus-coverage,
and noble-potential terms provide horizon estimates. These constants and the
0.94 turn discount are untuned heuristic choices. No calibrated win-probability
or theoretical optimality claim is made. H3 continuations can still undervalue
plans that H3 fails to execute.

The code uses the actual cards/nobles supplied by `state`. It never assumes all
120 color permutations preserve the deck. The test fixture incorporates your
corrected noble ID 1: blue-green-red. Update your production definitions
separately if they still contain the older requirement.

## Hidden information

By default, own reserved cards are known; opponent reservations are treated as
unknown. Their identities are pooled with hidden cards of the same tier and
resampled for each world. Opponent route forecasts do not directly read private
reservation identities. This is intentionally conservative if a reservation
was originally taken from the visible market.

To use legitimately remembered public reservations, supply:

```python
def known_reserved_cards(state, observer, owner):
    # Return only cards known from observer's observation/history.
    # The agent handles observer == owner itself.
    return observation_history.known_reservations(owner)

agent = HeuristicAgent16(known_reserved_cards=known_reserved_cards)
```

This callback must return only still-reserved, genuinely known cards, and must
work on cloned simulation states. It must not simply return every opponent
reservation from the internal state. Its results also constrain determinization.

Determinization plus a continuation policy is an approximation to hidden-
information play, not an information-set solver. A supplied continuation policy
must itself respect information boundaries. The default H3 implementation was
not available for inspection in this workspace. Worlds are shared across root
candidates; private `random.Random` policy RNGs are reseeded consistently.
Policies using global/random NumPy RNGs need their own reproducibility handling.

## Engine contract

Matches the supplied H15 conventions:

- `env._legal_actions(state)`, `env.step(action, state)`,
  `env._check_terminated(state)`, `env.action_to_id(action)`.
- `state.players`, `bank`, `visible_cards[tier]`, `decks`, `nobles`,
  `current_player`, `node_type`, optional `max_gems`.
- Player `gems`, `bonuses`, `points`, and `reserved_cards`.
- Card `tier` (or `level`), `points`, `bonus_color`, and `cost`.
- Noble `points` and `requirement`.
- Visible actions `tier`/`slot`, reserved buys `reserved_index`,
  token takes `gem_colors`.
- Terminal state `winners`, or scalar `winner`.

State objects and environments must support `clone()` or `deepcopy`. Clone
implementations must isolate mutable data. Simulation transitions use explicit
states as in H15; environments must not rely on unrelated mutable hidden state
outside those states. `step` can return a state, a tuple containing a state, or
mutate the supplied state. Deck dictionaries use keys 1/2/3; sequence decks must
be ordered tier 1/2/3. Reservation and deck collections must be mutable lists.

`HeuristicAgent16Diagnostics` is a compatibility subclass; base H16 already
collects diagnostics. `get_board_model_debug()` includes both forecasts, paths,
card families, and all remaining noble bundles. `get_candidate_debug()` and
`get_rollout_debug()` explain shortlists and results. Debug data includes action
objects and dataclasses; it is not directly JSON serializable.

## Validation and benchmarking

Run the included independent rules/contract tests from this directory:

```bash
python -m unittest -v test_heuristic_agent_16.py
```

The test module loads H16 with temporary enum/module adapters and restores those
module registrations afterward. It does not replace production modules or
validate the production engine. The JSON fixture contains your 90 supplied cards
and ten nobles with the corrected ID 1. Tests cover resource conservation,
payments, cap/discards, route replay, noble groups and single claims, final-round
responses, hidden reservations, full/cutoff continuations, reproducibility,
non-mutation, and legal policy output.

Validation performed for this delivery: all **26 tests passed**, including the
NumPy policy test; both Python files also passed bytecode compilation.

Nonterminal simulations with no legal actions now stop with a bounded heuristic
cutoff rather than raising an exception or inventing a pass, draw, or winner.
Diagnostics count `no_legal_response`, `no_legal_rollout`, `no_legal_forced`,
`no_legal_leaf`, and `no_legal_root` events. At the actual root, an empty action
list returns `None` from `select_action` (or a zero policy); the game runner must
handle that result instead of passing `None` to `env.step`. This does not repair
an underlying engine transition or legality bug if one causes the empty list.

A complete legal game against the included simple fixture policy was also used
as a smoke check. These checks establish mechanics under the fixture, not Elo.
No H16-vs-H15 strength benchmark or real-engine integration test was possible
without the rest of your repository and H3.

For a meaningful comparison in your engine, run paired deals with seats swapped,
keep the opponent pool fixed, and report match outcomes plus wall-clock time per
decision. Compare both equal rollout counts and similar thinking-time budgets:
H16 spends extra time on route/response analysis. Start with route-only mode to
measure overhead, then enable H3 continuations. Keep development/test deal seeds
separate from the final evaluation set.
