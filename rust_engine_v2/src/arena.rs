//! Many independent search trees, one contiguous neural batch per Python call.
use crate::{
    engine::State,
    search::{Config, Search},
};

pub struct Batch {
    pub token: u64,
    pub slots: Vec<usize>,
    pub ready: Vec<usize>,
    pub observations: Vec<u8>,
    pub actions: Vec<u8>,
    pub mask: Vec<u8>,
    pub width: usize,
}

struct Pending {
    token: u64,
    slots: Vec<usize>,
    lengths: Vec<usize>,
    width: usize,
}

pub struct Arena {
    trees: Vec<Option<Search>>,
    running: Vec<bool>,
    cursor: usize,
    token: u64,
    pending: Option<Pending>,
}

impl Arena {
    pub fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("Arena capacity must be positive".into());
        }
        Ok(Self {
            trees: (0..capacity).map(|_| None).collect(),
            running: vec![false; capacity],
            cursor: 0,
            token: 0,
            pending: None,
        })
    }
    pub fn tree(&self, slot: usize) -> Result<&Search, String> {
        self.trees
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| "Unknown arena slot".into())
    }
    pub fn tree_mut(&mut self, slot: usize) -> Result<&mut Search, String> {
        self.trees
            .get_mut(slot)
            .and_then(Option::as_mut)
            .ok_or_else(|| "Unknown arena slot".into())
    }
    fn mutable_slot(&self, slot: usize) -> Result<(), String> {
        if slot >= self.trees.len() {
            return Err("Unknown arena slot".into());
        }
        if self
            .pending
            .as_ref()
            .is_some_and(|p| p.slots.contains(&slot))
        {
            return Err("Respond to this slot's pending batch first".into());
        }
        Ok(())
    }
    pub fn add(&mut self, slot: usize, state: State, config: Config) -> Result<(), String> {
        self.mutable_slot(slot)?;
        if self.trees[slot].is_some() {
            return Err("Remove the existing game before replacing its slot".into());
        }
        self.trees[slot] = Some(Search::new(state, config));
        Ok(())
    }
    pub fn remove(&mut self, slot: usize) -> Result<(), String> {
        self.mutable_slot(slot)?;
        if self.tree(slot)?.is_active() {
            return Err("Cannot remove an active search".into());
        }
        self.trees[slot] = None;
        self.running[slot] = false;
        Ok(())
    }
    pub fn begin(&mut self, slot: usize, noise: Option<Vec<f64>>) -> Result<(), String> {
        self.mutable_slot(slot)?;
        self.tree_mut(slot)?.begin(noise)?;
        self.running[slot] = true;
        Ok(())
    }
    pub fn advance(&mut self, slot: usize, action: u16) -> Result<(u16, bool), String> {
        self.mutable_slot(slot)?;
        self.tree_mut(slot)?.advance(action)
    }
    pub fn gather(&mut self, max_batch: usize) -> Result<Batch, String> {
        if max_batch == 0 {
            return Err("Batch size must be positive".into());
        }
        if self.pending.is_some() {
            return Err("Respond to the pending batch before gathering again".into());
        }
        let mut slots = Vec::new();
        let mut ready = Vec::new();
        let mut requests = Vec::new();
        // Round robin also permits max_batch < number of games without starvation.
        for _ in 0..self.trees.len() {
            let slot = self.cursor;
            self.cursor = (self.cursor + 1) % self.trees.len();
            let Some(tree) = self.trees[slot].as_mut() else {
                continue;
            };
            if !self.running[slot] {
                continue;
            }
            if let Some(request) = tree.next_request() {
                slots.push(slot);
                requests.push(request);
                if slots.len() == max_batch {
                    break;
                }
            } else {
                ready.push(slot);
                self.running[slot] = false;
            }
        }
        let width = requests.iter().map(|r| r.1.len()).max().unwrap_or(0);
        let mut observations = Vec::with_capacity(slots.len() * 258 * 4);
        let mut actions = Vec::with_capacity(slots.len() * width * 8);
        let mut mask = Vec::with_capacity(slots.len() * width);
        let mut lengths = Vec::with_capacity(slots.len());
        for (obs, ids) in requests {
            lengths.push(ids.len());
            for value in obs {
                observations.extend_from_slice(&value.to_le_bytes());
            }
            for column in 0..width {
                let id = ids.get(column).copied().unwrap_or(0);
                actions.extend_from_slice(&(id as i64).to_le_bytes());
                mask.push(u8::from(column < ids.len()));
            }
        }
        if !slots.is_empty() {
            self.token = self.token.checked_add(1).ok_or("Batch token exhausted")?;
            self.pending = Some(Pending {
                token: self.token,
                slots: slots.clone(),
                lengths,
                width,
            });
        }
        Ok(Batch {
            token: self.token,
            slots,
            ready,
            observations,
            actions,
            mask,
            width,
        })
    }
    pub fn respond(&mut self, token: u64, payload: &[u8], double: bool) -> Result<(), String> {
        let pending = self.pending.as_ref().ok_or("No arena batch is pending")?;
        if token != pending.token {
            return Err("Stale or mismatched batch token".into());
        }
        let bytes = if double { 8 } else { 4 };
        let stride = (pending.width + 1) * bytes;
        if payload.len() != pending.slots.len() * stride {
            return Err(
                "Response must contain padded policy rows followed by one value per row".into(),
            );
        }
        let read = |offset: usize| -> f64 {
            if double {
                f64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap())
            } else {
                f32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as f64
            }
        };
        // Validate the entire batch before changing any tree. A rejected response is retryable.
        for (row, &length) in pending.lengths.iter().enumerate() {
            let value = read(row * stride + pending.width * bytes);
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err("Expected a finite value in [-1, 1]".into());
            }
            for column in 0..length {
                let prior = read(row * stride + column * bytes);
                if !prior.is_finite() || prior < 0.0 {
                    return Err("Expected finite nonnegative priors".into());
                }
            }
        }
        let pending = self.pending.take().unwrap();
        for (row, (&slot, &length)) in pending.slots.iter().zip(&pending.lengths).enumerate() {
            let priors = (0..length)
                .map(|column| read(row * stride + column * bytes))
                .collect();
            let value = read(row * stride + pending.width * bytes);
            self.tree_mut(slot)?.respond(priors, value)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_capacity_and_slot_lifecycle() {
        assert!(Arena::new(0).is_err());
        let mut arena = Arena::new(2).unwrap();
        arena.add(0, State::default(), Config::default()).unwrap();
        assert!(arena.add(0, State::default(), Config::default()).is_err());
        assert!(arena.begin(3, None).is_err());
        arena.begin(0, None).unwrap();
        assert!(arena.remove(0).is_err());
        assert!(arena.gather(0).is_err());
        let batch = arena.gather(2).unwrap();
        assert_eq!(batch.slots, vec![0]);
        assert_eq!(batch.observations.len(), 258 * 4);
        assert!(arena.gather(2).is_err());
        assert!(arena.respond(batch.token + 1, &[], false).is_err());
        assert!(arena.respond(batch.token, &[], false).is_err());
        assert!(arena.remove(0).is_err());
    }
    #[test]
    fn rejected_batch_does_not_mutate_any_tree() {
        let mut arena = Arena::new(2).unwrap();
        for slot in 0..2 {
            arena
                .add(slot, State::default(), Config::default())
                .unwrap();
            arena.begin(slot, None).unwrap();
        }
        let batch = arena.gather(2).unwrap();
        let before = arena.tree(0).unwrap().tree();
        let mut values = vec![1.0_f32 / batch.width as f32; 2 * (batch.width + 1)];
        values[batch.width] = 0.0;
        values[2 * (batch.width + 1) - 1] = f32::NAN;
        let payload: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(arena.respond(batch.token, &payload, false).is_err());
        assert_eq!(before, arena.tree(0).unwrap().tree());
        *values.last_mut().unwrap() = 0.0;
        let payload: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        arena.respond(batch.token, &payload, false).unwrap();
        assert_ne!(before, arena.tree(0).unwrap().tree());
    }

    #[test]
    fn final_neural_response_reports_completion_once() {
        let mut arena = Arena::new(1).unwrap();
        let config = Config {
            simulations: 1,
            adaptive_simulations: false,
            ..Config::default()
        };
        arena.add(0, State::default(), config).unwrap();
        arena.begin(0, None).unwrap();
        let batch = arena.gather(1).unwrap();
        let mut values = vec![1.0_f32 / batch.width as f32; batch.width + 1];
        values[batch.width] = 0.0;
        let payload: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        arena.respond(batch.token, &payload, false).unwrap();
        assert!(!arena.tree(0).unwrap().is_active());
        assert_eq!(arena.gather(1).unwrap().ready, vec![0]);
        assert!(arena.gather(1).unwrap().ready.is_empty());
    }
}
