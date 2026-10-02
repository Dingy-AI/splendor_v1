//! Two-player rules engine. State copies contain no Python objects or heap allocations.
use crate::tables::*;
use std::sync::OnceLock;

pub const EMPTY: u8 = 255;
pub const OBSERVATION_SIZE: usize = 258;

#[derive(Clone, Copy)]
pub struct Card {
    pub tier: u8,
    pub points: u8,
    pub bonus: usize,
    pub cost: [u8; 5],
    pub mapping: [usize; 5],
}

#[derive(Clone, Copy)]
pub struct Noble {
    pub points: u8,
    pub requirement: [u8; 5],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Player {
    pub gems: [u8; 6],
    pub bonuses: [u8; 5],
    pub reserved: [u8; 3],
    pub hidden: [bool; 3],
    pub reserved_len: usize,
    pub purchased: [u8; 90],
    pub purchased_len: usize,
    pub nobles: [u8; 10],
    pub nobles_len: usize,
    pub points: u16,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            gems: [0; 6],
            bonuses: [0; 5],
            reserved: [EMPTY; 3],
            hidden: [false; 3],
            reserved_len: 0,
            purchased: [EMPTY; 90],
            purchased_len: 0,
            nobles: [EMPTY; 10],
            nobles_len: 0,
            points: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    // Python NodeType values: main=0, noble=1, discard=2.
    pub node_type: u8,
    pub players: [Player; 2],
    pub bank: [u8; 6],
    pub nobles: [u8; 3],
    pub visible: [[u8; 4]; 3],
    pub decks: [[u8; 40]; 3],
    pub deck_lens: [usize; 3],
    pub current_player: usize,
    pub turn_number: u32,
    pub winners: [bool; 2],
    pub game_over: bool,
    pub end_triggered: bool,
    pub noble_taken: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            node_type: 0,
            players: [Player::default(); 2],
            bank: [4, 4, 4, 4, 4, 5],
            nobles: [EMPTY; 3],
            visible: [[EMPTY; 4]; 3],
            decks: [[EMPTY; 40]; 3],
            deck_lens: [0; 3],
            current_player: 0,
            turn_number: 0,
            winners: [false; 2],
            game_over: false,
            end_triggered: false,
            noble_taken: false,
        }
    }
}

pub fn payment_table(tier: usize) -> &'static [[u8; 5]] {
    match tier {
        0 => &PAYMENTS_1,
        1 => &PAYMENTS_2,
        2 => &PAYMENTS_3,
        _ => unreachable!(),
    }
}

fn payment_key(payment: &[u8; 5]) -> usize {
    payment
        .iter()
        .enumerate()
        .map(|(i, &x)| (x as usize) << (3 * i))
        .sum()
}

// Direct lookup preserves the canonical Python IDs without per-action hashing.
fn payment_ids() -> &'static [[i16; 32768]; 3] {
    static LOOKUP: OnceLock<[[i16; 32768]; 3]> = OnceLock::new();
    LOOKUP.get_or_init(|| {
        let mut lookup = [[-1; 32768]; 3];
        for (tier, row) in lookup.iter_mut().enumerate() {
            for (id, payment) in payment_table(tier).iter().enumerate() {
                row[payment_key(payment)] = id as i16;
            }
        }
        lookup
    })
}

fn emit_payment(card: &Card, table: usize, base: usize, payment: [u8; 5], out: &mut Vec<u16>) {
    let canonical = std::array::from_fn(|i| payment[card.mapping[i]]);
    let id = payment_ids()[table][payment_key(&canonical)];
    assert!(id >= 0, "Python canonical payment table has no entry");
    out.push((base + id as usize) as u16);
}

struct PaymentEnumeration<'a> {
    card: &'a Card,
    table: usize,
    base: usize,
    low: &'a [u8; 5],
    high: &'a [u8; 5],
}

impl PaymentEnumeration<'_> {
    fn enumerate(&self, payment: &mut [u8; 5], color: usize, remaining: u8, out: &mut Vec<u16>) {
        if color == 5 {
            emit_payment(self.card, self.table, self.base, *payment, out);
            return;
        }
        for gold in self.low[color]..=self.high[color].min(remaining) {
            payment[color] = gold;
            self.enumerate(payment, color + 1, remaining - gold, out);
        }
    }
}

fn buy_actions(player: &Player, card_id: u8, table: usize, base: usize, out: &mut Vec<u16>) {
    let card = &CARDS[card_id as usize];
    let required: [u8; 5] = std::array::from_fn(|i| card.cost[i].saturating_sub(player.bonuses[i]));
    let minimum: [u8; 5] = std::array::from_fn(|i| required[i].saturating_sub(player.gems[i]));
    let min_total: u16 = minimum.iter().map(|&v| v as u16).sum();
    let gold = player.gems[5];
    if min_total > gold as u16 {
        return;
    }
    let extra = gold - min_total as u8;
    // Preserve Python's specialized payment enumeration order (including extra=2).
    if extra <= 2 {
        emit_payment(card, table, base, minimum, out);
        if extra == 0 {
            return;
        }
        for i in (0..5).rev() {
            if required[i] <= minimum[i] {
                continue;
            }
            let mut payment = minimum;
            payment[i] += 1;
            emit_payment(card, table, base, payment, out);
            if extra == 2 {
                for j in ((i + 1)..5).rev() {
                    if required[j] <= minimum[j] {
                        continue;
                    }
                    payment[j] += 1;
                    emit_payment(card, table, base, payment, out);
                    payment[j] -= 1;
                }
                if required[i] - minimum[i] >= 2 {
                    payment[i] += 1;
                    emit_payment(card, table, base, payment, out);
                }
            }
        }
    } else {
        PaymentEnumeration {
            card,
            table,
            base,
            low: &minimum,
            high: &required,
        }
        .enumerate(&mut [0; 5], 0, gold, out);
    }
}

impl State {
    fn qualifies(&self, player: usize, noble: u8) -> bool {
        noble != EMPTY
            && (0..5)
                .all(|c| self.players[player].bonuses[c] >= NOBLES[noble as usize].requirement[c])
    }

    pub fn legal_action_ids(&self) -> Vec<u16> {
        let mut out = Vec::with_capacity(32);
        self.write_legal_actions(&mut out);
        out
    }

    /// Reuse the caller's action buffer in the future native MCTS loop.
    pub fn write_legal_actions(&self, out: &mut Vec<u16>) {
        out.clear();
        let player = &self.players[self.current_player];
        match self.node_type {
            2 => {
                if player.gems.iter().map(|&v| v as u16).sum::<u16>() > 10 {
                    for c in 0..6 {
                        if player.gems[c] > 0 {
                            out.push((DISCARD_START + c) as u16);
                        }
                    }
                }
            }
            1 => {
                for slot in 0..3 {
                    if self.qualifies(self.current_player, self.nobles[slot]) {
                        out.push((NOBLE_START + slot) as u16);
                    }
                }
            }
            0 => {
                let starts = [BUY_T1_START, BUY_T2_START, BUY_T3_START];
                for (tier, &start) in starts.iter().enumerate() {
                    for slot in 0..4 {
                        let card = self.visible[tier][slot];
                        if card != EMPTY {
                            buy_actions(
                                player,
                                card,
                                tier,
                                start + slot * payment_table(tier).len(),
                                out,
                            );
                        }
                    }
                }
                for slot in 0..player.reserved_len {
                    buy_actions(
                        player,
                        player.reserved[slot],
                        2,
                        BUY_RESERVED_START + slot * PAYMENTS_3.len(),
                        out,
                    );
                }
                if player.reserved_len < 3 {
                    for tier in 0..3 {
                        for slot in 0..4 {
                            if self.visible[tier][slot] != EMPTY {
                                out.push((30 + tier * 4 + slot) as u16);
                            }
                        }
                    }
                    for tier in 0..3 {
                        if self.deck_lens[tier] > 0 {
                            out.push((42 + tier) as u16);
                        }
                    }
                }
                let available = (0..5).filter(|&c| self.bank[c] > 0).count();
                let count = available.min(3);
                for (id, colors) in GEM_ACTIONS.iter().enumerate() {
                    if count > 0
                        && colors.len() == count
                        && colors.windows(2).all(|w| w[0] != w[1])
                        && colors.iter().all(|&c| self.bank[c] > 0)
                    {
                        out.push(id as u16);
                    }
                }
                for c in 0..5 {
                    if self.bank[c] >= 4 {
                        out.push((5 + c) as u16);
                    }
                }
            }
            _ => unreachable!("invalid node type"),
        }
    }

    fn draw(&mut self, tier: usize) -> u8 {
        if self.deck_lens[tier] == 0 {
            return EMPTY;
        }
        self.deck_lens[tier] -= 1;
        let card = self.decks[tier][self.deck_lens[tier]];
        self.decks[tier][self.deck_lens[tier]] = EMPTY;
        card
    }

    fn reserve(&mut self, card: u8, hidden: bool) {
        let player = &mut self.players[self.current_player];
        let slot = player.reserved_len;
        player.reserved[slot] = card;
        player.hidden[slot] = hidden;
        player.reserved_len += 1;
        if self.bank[5] > 0 {
            self.bank[5] -= 1;
            player.gems[5] += 1;
        }
    }

    fn buy(&mut self, card_id: u8, canonical: &[u8; 5]) {
        let card = &CARDS[card_id as usize];
        let player = &mut self.players[self.current_player];
        let mut payment = [0; 5];
        for i in 0..5 {
            payment[card.mapping[i]] = canonical[i];
        }
        for (c, &gold) in payment.iter().enumerate() {
            let paid = card.cost[c].saturating_sub(player.bonuses[c]) - gold;
            player.gems[c] -= paid;
            self.bank[c] += paid;
        }
        let gold: u8 = payment.iter().sum();
        player.gems[5] -= gold;
        self.bank[5] += gold;
        player.purchased[player.purchased_len] = card_id;
        player.purchased_len += 1;
        player.points += card.points as u16;
        player.bonuses[card.bonus] += 1;
        self.end_triggered |= player.points >= 15;
    }

    /// Only native callers that already selected a legal action may use this.
    pub(crate) fn apply_legal(&mut self, id: usize) -> (u16, bool) {
        let actor = self.current_player;
        let previous = self.players[actor].points;
        if id < 30 {
            for &c in GEM_ACTIONS[id] {
                self.players[actor].gems[c] += 1;
                self.bank[c] -= 1;
            }
        } else if id < 42 {
            let tier = (id - 30) / 4;
            let slot = (id - 30) % 4;
            let card = self.visible[tier][slot];
            self.visible[tier][slot] = self.draw(tier);
            self.reserve(card, false);
        } else if id < 45 {
            let card = self.draw(id - 42);
            self.reserve(card, true);
        } else if id < BUY_RESERVED_START {
            let (tier, start) = if id < BUY_T2_START {
                (0, BUY_T1_START)
            } else if id < BUY_T3_START {
                (1, BUY_T2_START)
            } else {
                (2, BUY_T3_START)
            };
            let table = payment_table(tier);
            let slot = (id - start) / table.len();
            self.buy(self.visible[tier][slot], &table[(id - start) % table.len()]);
            self.visible[tier][slot] = self.draw(tier);
        } else if id < DISCARD_START {
            let slot = (id - BUY_RESERVED_START) / PAYMENTS_3.len();
            self.buy(
                self.players[actor].reserved[slot],
                &PAYMENTS_3[(id - BUY_RESERVED_START) % PAYMENTS_3.len()],
            );
            let player = &mut self.players[actor];
            for i in slot..(player.reserved_len - 1) {
                player.reserved[i] = player.reserved[i + 1];
                player.hidden[i] = player.hidden[i + 1];
            }
            player.reserved_len -= 1;
            player.reserved[player.reserved_len] = EMPTY;
            player.hidden[player.reserved_len] = false;
        } else if id < NOBLE_START {
            let c = id - DISCARD_START;
            self.players[actor].gems[c] -= 1;
            self.bank[c] += 1;
        } else {
            let slot = id - NOBLE_START;
            let noble = self.nobles[slot];
            let player = &mut self.players[actor];
            player.nobles[player.nobles_len] = noble;
            player.nobles_len += 1;
            player.points += NOBLES[noble as usize].points as u16;
            self.nobles[slot] = EMPTY;
            self.noble_taken = true;
            self.end_triggered |= player.points >= 15;
        }
        if self.players[actor]
            .gems
            .iter()
            .map(|&v| v as u16)
            .sum::<u16>()
            > 10
        {
            self.node_type = 2;
        } else if !self.noble_taken && self.nobles.iter().any(|&n| self.qualifies(actor, n)) {
            self.node_type = 1;
        } else {
            self.node_type = 0;
            self.noble_taken = false;
            self.current_player = 1 - actor;
            self.turn_number += 1;
        }
        let reward = self.players[actor].points - previous;
        self.check_terminated();
        (reward, self.game_over)
    }

    /// Match Python's _check_terminated, including winner recomputation.
    pub(crate) fn check_terminated(&mut self) -> bool {
        if self.end_triggered && self.node_type == 0 && self.current_player == 0 {
            let best_points = self.players.iter().map(|p| p.points).max().unwrap();
            let fewest = self
                .players
                .iter()
                .filter(|p| p.points == best_points)
                .map(|p| p.purchased_len)
                .min()
                .unwrap();
            self.winners = std::array::from_fn(|i| {
                self.players[i].points == best_points && self.players[i].purchased_len == fewest
            });
            self.game_over = true;
            return true;
        }
        false
    }

    pub fn step(&mut self, id: usize) -> Result<(u16, bool), String> {
        if self.game_over {
            return Err("Cannot step a finished game".into());
        }
        if id >= ACTION_SPACE_SIZE || !self.legal_action_ids().contains(&(id as u16)) {
            return Err(format!("Illegal action ID {id}"));
        }
        // Apply transactionally: malformed imported states cannot partially mutate.
        let mut next = *self;
        if next.players[next.current_player].purchased_len >= 90
            && (45..DISCARD_START).contains(&id)
        {
            return Err("Purchased-card storage is full".into());
        }
        if next.players[next.current_player].nobles_len >= 10 && id >= NOBLE_START {
            return Err("Noble storage is full".into());
        }
        let result = next.apply_legal(id);
        *self = next;
        Ok(result)
    }

    pub fn observation(&self) -> [f32; OBSERVATION_SIZE] {
        let mut out = [0.; OBSERVATION_SIZE];
        let mut cursor = 0;
        let mut push = |value: f32| {
            out[cursor] = value;
            cursor += 1;
        };
        for index in [self.current_player, 1 - self.current_player] {
            let player = &self.players[index];
            for v in player.gems {
                push(v as f32 / 4.);
            }
            for v in player.bonuses {
                push(v as f32 / 5.);
            }
            for slot in 0..3 {
                if slot >= player.reserved_len {
                    for _ in 0..12 {
                        push(0.);
                    }
                } else if index != self.current_player && player.hidden[slot] {
                    for _ in 0..11 {
                        push(0.);
                    }
                    push(1.);
                } else {
                    for v in encode_card(player.reserved[slot]) {
                        push(v);
                    }
                    push(0.);
                }
            }
            push(player.points as f32 / 20.);
        }
        for c in 0..6 {
            push(self.bank[c] as f32 / if c == 5 { 5. } else { 4. });
        }
        for (tier, norm) in [36., 26., 16.].iter().enumerate() {
            push(self.deck_lens[tier] as f32 / norm);
        }
        for noble in self.nobles {
            if noble == EMPTY {
                for _ in 0..6 {
                    push(0.);
                }
            } else {
                for req in NOBLES[noble as usize].requirement {
                    push(req as f32 / 4.);
                }
                push(NOBLES[noble as usize].points as f32 / 3.);
            }
        }
        for tier in self.visible {
            for card in tier {
                for v in encode_card(card) {
                    push(v);
                }
            }
        }
        // Encoding order differs from enum values: main, overflow, noble.
        for node in [0, 2, 1] {
            push(if self.node_type == node { 1. } else { 0. });
        }
        debug_assert_eq!(cursor, OBSERVATION_SIZE);
        out
    }
}

fn encode_card(id: u8) -> [f32; 11] {
    let mut out = [0.; 11];
    if id != EMPTY {
        let card = &CARDS[id as usize];
        for (c, &cost) in card.cost.iter().enumerate() {
            out[c] = cost as f32 / 7.;
        }
        out[5 + card.bonus] = 1.;
        out[10] = card.points as f32 / 5.;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_payment_id_contract() {
        assert_eq!(ACTION_SPACE_SIZE, 1139);
        for tier in 0..3 {
            for (id, payment) in payment_table(tier).iter().enumerate() {
                assert_eq!(payment_ids()[tier][payment_key(payment)], id as i16);
            }
        }
    }

    #[test]
    fn illegal_steps_leave_state_unchanged() {
        let mut state = State::default();
        let before = state;
        assert!(state.step(ACTION_SPACE_SIZE).is_err());
        assert!(state.step(30).is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn observation_uses_current_player_and_node_encoding_order() {
        let mut state = State {
            current_player: 1,
            node_type: 2,
            ..State::default()
        };
        state.players[1].gems[0] = 2;
        let obs = state.observation();
        assert_eq!(obs[0], 0.5);
        assert_eq!(&obs[255..], &[0., 1., 0.]);
    }
}
