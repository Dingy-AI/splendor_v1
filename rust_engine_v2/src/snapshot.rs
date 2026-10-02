//! JSON is a boundary format only; rules and future search operate on native State.
use crate::engine::{Player, State, EMPTY};
use crate::tables::CARDS;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const COLORS: [&str; 6] = ["WHITE", "BLUE", "GREEN", "RED", "BLACK", "GOLD"];

#[derive(Serialize, Deserialize)]
struct PlayerSnapshot {
    id: usize,
    points: u16,
    gems: BTreeMap<String, u8>,
    bonuses: BTreeMap<String, u8>,
    reserved_card_ids: Vec<u8>,
    reserved_card_hidden: Vec<bool>,
    purchased_card_ids: Vec<u8>,
    noble_ids: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct Snapshot {
    state_schema_version: u8,
    node_type: String,
    players: Vec<PlayerSnapshot>,
    bank: BTreeMap<String, u8>,
    noble_ids: Vec<Option<u8>>,
    visible_card_ids: BTreeMap<usize, Vec<Option<u8>>>,
    deck_card_ids: BTreeMap<usize, Vec<u8>>,
    current_player: usize,
    turn_number: u32,
    winner_ids: Vec<usize>,
    game_over: bool,
    end_triggered: bool,
    noble_taken: bool,
}

fn resources<const N: usize>(data: &BTreeMap<String, u8>, max: u8) -> Result<[u8; N], String> {
    if data.len() != N {
        return Err(format!("Expected {N} gem-color entries"));
    }
    let mut out = [0; N];
    for i in 0..N {
        out[i] = *data
            .get(COLORS[i])
            .ok_or_else(|| format!("Missing color {}", COLORS[i]))?;
        if out[i] > max {
            return Err(format!("{} exceeds supported limit {max}", COLORS[i]));
        }
    }
    Ok(out)
}

fn copy_cards<const N: usize>(ids: &[u8], out: &mut [u8; N]) -> Result<(), String> {
    if ids.len() > N || ids.iter().any(|&id| id >= 90) {
        return Err("Invalid card ID or card-list length".into());
    }
    out[..ids.len()].copy_from_slice(ids);
    Ok(())
}

impl State {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let snap: Snapshot = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if snap.state_schema_version != 1 {
            return Err("Only state schema version 1 is supported".into());
        }
        if snap.players.len() != 2 || snap.current_player >= 2 {
            return Err("This port supports exactly two players".into());
        }
        if snap.noble_ids.len() != 3
            || snap.visible_card_ids.len() != 3
            || snap.deck_card_ids.len() != 3
        {
            return Err("Expected three noble slots and three card tiers".into());
        }
        // Guard arithmetic at the public boundary, rather than truncating imported counters.
        if snap.turn_number == u32::MAX {
            return Err("Turn counter is out of range".into());
        }
        let mut state = Self {
            node_type: match snap.node_type.as_str() {
                "MAIN_DECISION" => 0,
                "NOBLE_CLAIM" => 1,
                "OVERFLOW_DISCARD" => 2,
                _ => return Err("Unknown node type".into()),
            },
            bank: resources(&snap.bank, 5)?,
            current_player: snap.current_player,
            turn_number: snap.turn_number,
            game_over: snap.game_over,
            end_triggered: snap.end_triggered,
            noble_taken: snap.noble_taken,
            ..Self::default()
        };
        for (i, src) in snap.players.iter().enumerate() {
            if src.id != i
                || src.reserved_card_ids.len() != src.reserved_card_hidden.len()
                || src.noble_ids.len() > 10
                || src.noble_ids.iter().any(|&id| id >= 10)
                || src.points > 255
            {
                return Err("Invalid player fields".into());
            }
            let player = &mut state.players[i];
            player.gems = resources(&src.gems, 13)?;
            player.bonuses = resources(&src.bonuses, 90)?;
            player.points = src.points;
            copy_cards(&src.reserved_card_ids, &mut player.reserved)?;
            copy_cards(&src.purchased_card_ids, &mut player.purchased)?;
            player.reserved_len = src.reserved_card_ids.len();
            player.hidden[..player.reserved_len].copy_from_slice(&src.reserved_card_hidden);
            player.purchased_len = src.purchased_card_ids.len();
            player.nobles_len = src.noble_ids.len();
            player.nobles[..player.nobles_len].copy_from_slice(&src.noble_ids);
        }
        for (slot, id) in snap.noble_ids.iter().enumerate() {
            if id.is_some_and(|id| id >= 10) {
                return Err("Invalid noble ID".into());
            }
            state.nobles[slot] = id.unwrap_or(EMPTY);
        }
        for tier in 0..3 {
            let visible = snap
                .visible_card_ids
                .get(&(tier + 1))
                .ok_or("Missing visible tier")?;
            let deck = snap
                .deck_card_ids
                .get(&(tier + 1))
                .ok_or("Missing deck tier")?;
            if visible.len() != 4 || deck.len() > [40, 30, 20][tier] {
                return Err("Invalid visible-slot count or deck size".into());
            }
            for (slot, id) in visible.iter().enumerate() {
                if let Some(id) = id {
                    if *id >= 90 || CARDS[*id as usize].tier as usize != tier + 1 {
                        return Err("Invalid visible card or tier".into());
                    }
                }
                state.visible[tier][slot] = id.unwrap_or(EMPTY);
            }
            if deck
                .iter()
                .any(|&id| id >= 90 || CARDS[id as usize].tier as usize != tier + 1)
            {
                return Err("Invalid deck card or tier".into());
            }
            copy_cards(deck, &mut state.decks[tier])?;
            state.deck_lens[tier] = deck.len();
        }
        for id in snap.winner_ids {
            if id >= 2 {
                return Err("Invalid winner ID".into());
            }
            state.winners[id] = true;
        }
        // All real game states obey these totals; reject data that could overflow payment arithmetic.
        for c in 0..6 {
            let total = state.bank[c] as u16
                + state.players[0].gems[c] as u16
                + state.players[1].gems[c] as u16;
            if total != if c == 5 { 5 } else { 4 } {
                return Err("Gem conservation violation".into());
            }
        }
        Ok(state)
    }

    pub fn snapshot_json(&self) -> String {
        let map = |values: &[u8]| -> BTreeMap<String, u8> {
            values
                .iter()
                .enumerate()
                .map(|(i, &v)| (COLORS[i].into(), v))
                .collect()
        };
        let player = |id: usize, src: &Player| PlayerSnapshot {
            id,
            points: src.points,
            gems: map(&src.gems),
            bonuses: map(&src.bonuses),
            reserved_card_ids: src.reserved[..src.reserved_len].to_vec(),
            reserved_card_hidden: src.hidden[..src.reserved_len].to_vec(),
            purchased_card_ids: src.purchased[..src.purchased_len].to_vec(),
            noble_ids: src.nobles[..src.nobles_len].to_vec(),
        };
        let snap = Snapshot {
            state_schema_version: 1,
            node_type: match self.node_type {
                0 => "MAIN_DECISION",
                1 => "NOBLE_CLAIM",
                _ => "OVERFLOW_DISCARD",
            }
            .into(),
            players: self
                .players
                .iter()
                .enumerate()
                .map(|(i, p)| player(i, p))
                .collect(),
            bank: map(&self.bank),
            noble_ids: self
                .nobles
                .iter()
                .map(|&id| if id == EMPTY { None } else { Some(id) })
                .collect(),
            visible_card_ids: self
                .visible
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    (
                        i + 1,
                        row.iter()
                            .map(|&id| if id == EMPTY { None } else { Some(id) })
                            .collect(),
                    )
                })
                .collect(),
            deck_card_ids: (0..3)
                .map(|i| (i + 1, self.decks[i][..self.deck_lens[i]].to_vec()))
                .collect(),
            current_player: self.current_player,
            turn_number: self.turn_number,
            winner_ids: (0..2).filter(|&i| self.winners[i]).collect(),
            game_over: self.game_over,
            end_triggered: self.end_triggered,
            noble_taken: self.noble_taken,
        };
        serde_json::to_string(&snap).expect("snapshot contains only serializable primitives")
    }
}
