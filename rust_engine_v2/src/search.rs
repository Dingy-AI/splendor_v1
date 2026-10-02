//! Resumable two-player neural PUCT, matching mcts_v5_direct.py.
//! No Python objects or callbacks are held in the tree.
use crate::engine::State;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub simulations: usize,
    pub c_puct: f64,
    pub dirichlet_epsilon: f64,
    pub adaptive_simulations: bool,
    pub min_simulations: usize,
    pub check_interval: usize,
    pub target_visits_per_action: f64,
    pub single_action_simulations: usize,
    pub stability_checks: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            simulations: 400,
            c_puct: 3.0,
            dirichlet_epsilon: 0.25,
            adaptive_simulations: true,
            min_simulations: 80,
            check_interval: 20,
            target_visits_per_action: 20.0,
            single_action_simulations: 4,
            stability_checks: 3,
        }
    }
}
impl Config {
    pub fn from_json(input: &str) -> Result<Self, String> {
        let cfg: Self = serde_json::from_str(input).map_err(|e| e.to_string())?;
        if cfg.simulations == 0
            || cfg.min_simulations == 0
            || cfg.check_interval == 0
            || cfg.single_action_simulations == 0
            || cfg.stability_checks == 0
        {
            return Err("Simulation counts and intervals must be positive".into());
        }
        if !cfg.c_puct.is_finite()
            || cfg.c_puct < 0.0
            || !cfg.target_visits_per_action.is_finite()
            || cfg.target_visits_per_action < 0.0
            || !cfg.dirichlet_epsilon.is_finite()
            || !(0.0..=1.0).contains(&cfg.dirichlet_epsilon)
        {
            return Err("Invalid PUCT, target visits, or noise mixing coefficient".into());
        }
        Ok(cfg)
    }
}

#[derive(Clone)]
struct Node {
    state: Option<Box<State>>,
    parent: Option<usize>,
    action: Option<u16>,
    children: Vec<usize>,
    legal: Option<Vec<u16>>,
    visits: u64,
    value: f64,
    prior: f64,
    network_prior: f64,
    expanded: bool,
}
impl Node {
    fn new(state: Option<State>, parent: Option<usize>, action: Option<u16>, prior: f64) -> Self {
        Self {
            state: state.map(Box::new),
            parent,
            action,
            children: Vec::new(),
            legal: None,
            visits: 0,
            value: 0.0,
            prior,
            network_prior: prior,
            expanded: false,
        }
    }
    fn q(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value / self.visits as f64
        }
    }
}

pub struct Search {
    cfg: Config,
    nodes: Vec<Node>,
    root_player: usize,
    active: bool,
    pending: Option<usize>,
    noise: Option<Vec<f64>>,
    noise_enabled: bool,
    initial_visits: u64,
    num_legal: usize,
    hard_limit: usize,
    soft_budget: usize,
    target_total: u64,
    simulations_run: usize,
    history: Vec<u16>,
    pub metadata: Option<Value>,
}
pub type Request = (Vec<f32>, Vec<u16>);

impl Search {
    pub fn new(state: State, cfg: Config) -> Self {
        Self {
            root_player: state.current_player,
            cfg,
            nodes: vec![Node::new(Some(state), None, None, 0.0)],
            active: false,
            pending: None,
            noise: None,
            noise_enabled: false,
            initial_visits: 0,
            num_legal: 0,
            hard_limit: 0,
            soft_budget: 0,
            target_total: 0,
            simulations_run: 0,
            history: Vec::new(),
            metadata: None,
        }
    }
    pub fn root_state(&self) -> State {
        **self.nodes[0].state.as_ref().unwrap()
    }
    pub fn is_active(&self) -> bool {
        self.active
    }
    pub fn root_actions(&mut self) -> Vec<u16> {
        self.legal(0).to_vec()
    }
    fn legal(&mut self, id: usize) -> &[u16] {
        if self.nodes[id].legal.is_none() {
            self.nodes[id].legal = Some(self.nodes[id].state.as_ref().unwrap().legal_action_ids());
        }
        self.nodes[id].legal.as_ref().unwrap()
    }
    pub fn begin(&mut self, noise: Option<Vec<f64>>) -> Result<(), String> {
        if self.active {
            return Err("Finish the active search before beginning another".into());
        }
        let num_legal = self.legal(0).len();
        if let Some(values) = &noise {
            if values.len() != num_legal
                || values.iter().any(|v| !v.is_finite() || *v < 0.0)
                || (num_legal > 0 && (values.iter().sum::<f64>() - 1.0).abs() > 1e-6)
            {
                return Err("Root noise must be a probability vector in legal-action order".into());
            }
        }
        self.num_legal = num_legal;
        self.root_player = self.root_state().current_player;
        self.initial_visits = self.nodes[0].visits;
        self.simulations_run = 0;
        self.history.clear();
        self.metadata = None;
        self.noise_enabled = noise.is_some();
        self.noise = noise;
        self.pending = None;
        self.active = num_legal > 0;
        self.hard_limit = self.cfg.simulations;
        if !self.cfg.adaptive_simulations {
            self.soft_budget = self.hard_limit;
            self.target_total = self.initial_visits + self.hard_limit as u64;
        } else if num_legal <= 1 {
            self.hard_limit = self.hard_limit.min(self.cfg.single_action_simulations);
            self.soft_budget = self.hard_limit;
            self.target_total = self.initial_visits + self.hard_limit as u64;
        } else {
            self.target_total =
                (self.cfg.target_visits_per_action * num_legal as f64).ceil() as u64;
            let required = self.target_total.saturating_sub(self.initial_visits) as usize;
            let soft = required
                .max(self.cfg.min_simulations.min(self.hard_limit))
                .min(self.hard_limit);
            self.soft_budget = soft
                .div_ceil(self.cfg.check_interval)
                .saturating_mul(self.cfg.check_interval)
                .min(self.hard_limit);
            if self.soft_budget == 0 {
                self.soft_budget = self.cfg.check_interval.min(self.hard_limit);
            }
        }
        if self.nodes[0].expanded && !self.nodes[0].children.is_empty() {
            self.apply_noise();
        }
        Ok(())
    }
    fn apply_noise(&mut self) {
        if let Some(noise) = self.noise.take() {
            for (&id, n) in self.nodes[0].children.clone().iter().zip(noise) {
                self.nodes[id].prior = (1.0 - self.cfg.dirichlet_epsilon) * self.nodes[id].prior
                    + self.cfg.dirichlet_epsilon * n;
            }
        }
    }
    fn materialize(&mut self, id: usize) {
        if self.nodes[id].state.is_some() {
            return;
        }
        let parent = self.nodes[id].parent.unwrap();
        let mut state = **self.nodes[parent].state.as_ref().unwrap();
        state.apply_legal(self.nodes[id].action.unwrap() as usize);
        self.nodes[id].state = Some(Box::new(state));
    }
    fn select(&mut self) -> usize {
        let mut id = 0;
        loop {
            let node = &self.nodes[id];
            if !node.expanded || node.children.is_empty() {
                return id;
            }
            let same_player = node.state.as_ref().unwrap().current_player == self.root_player;
            let sqrt_visits = (node.visits.max(1) as f64).sqrt();
            let mut best = node.children[0];
            let mut best_score = f64::NEG_INFINITY;
            for &child_id in &node.children {
                let child = &self.nodes[child_id];
                let q = if same_player { child.q() } else { -child.q() };
                let score =
                    q + self.cfg.c_puct * child.prior * sqrt_visits / (1 + child.visits) as f64;
                if score > best_score {
                    best = child_id;
                    best_score = score;
                }
            }
            self.materialize(best);
            id = best;
        }
    }
    fn terminal_value(&self, id: usize) -> f64 {
        let state = self.nodes[id].state.as_ref().unwrap();
        if state.winners[0] == state.winners[1] {
            0.0
        } else if state.winners[self.root_player] {
            1.0
        } else {
            -1.0
        }
    }
    fn request(&mut self, id: usize) -> Request {
        (
            self.nodes[id]
                .state
                .as_ref()
                .unwrap()
                .observation()
                .to_vec(),
            self.legal(id).to_vec(),
        )
    }
    /// Advances native selection/terminal backups until a neural leaf is needed.
    pub fn next_request(&mut self) -> Option<Request> {
        if let Some(id) = self.pending {
            return Some(self.request(id));
        }
        while self.active {
            let id = self.select();
            if self.nodes[id].state.as_mut().unwrap().check_terminated() {
                self.nodes[id].expanded = true;
                let value = self.terminal_value(id);
                self.complete_simulation(id, value);
            } else if self.legal(id).is_empty() {
                let state = self.nodes[id].state.as_mut().unwrap();
                state.game_over = true;
                state.winners = [false; 2];
                self.nodes[id].expanded = true;
                self.complete_simulation(id, 0.0);
            } else {
                self.pending = Some(id);
                return Some(self.request(id));
            }
        }
        None
    }
    /// Policy probabilities follow the request's ordered legal IDs. Value is player-to-move.
    pub fn respond(&mut self, priors: Vec<f64>, value: f64) -> Result<(), String> {
        let id = self.pending.ok_or("No neural evaluation is pending")?;
        if priors.len() != self.legal(id).len()
            || priors.iter().any(|v| !v.is_finite() || *v < 0.0)
            || !value.is_finite()
            || !(-1.0..=1.0).contains(&value)
        {
            return Err("Expected finite legal-order priors and a value in [-1, 1]".into());
        }
        let actions = self.legal(id).to_vec();
        let start = self.nodes.len();
        self.nodes.reserve(actions.len());
        for (action, prior) in actions.into_iter().zip(priors) {
            self.nodes
                .push(Node::new(None, Some(id), Some(action), prior));
        }
        self.nodes[id].children = (start..self.nodes.len()).collect();
        self.nodes[id].expanded = true;
        if id == 0 {
            self.apply_noise();
        }
        let value = if self.nodes[id].state.as_ref().unwrap().current_player == self.root_player {
            value
        } else {
            -value
        };
        self.pending = None;
        self.complete_simulation(id, value);
        Ok(())
    }
    fn best_child(&self) -> Option<usize> {
        let mut best = None;
        for &id in &self.nodes[0].children {
            if best.is_none_or(|b: usize| self.nodes[id].visits > self.nodes[b].visits) {
                best = Some(id);
            }
        }
        best
    }
    pub fn best_action(&self) -> Option<u16> {
        self.best_child().and_then(|id| self.nodes[id].action)
    }
    fn stable_count(&self) -> usize {
        self.history.last().map_or(0, |last| {
            self.history.iter().rev().take_while(|v| *v == last).count()
        })
    }
    fn complete_simulation(&mut self, id: usize, value: f64) {
        let mut current = Some(id);
        while let Some(i) = current {
            self.nodes[i].visits += 1;
            self.nodes[i].value += value;
            current = self.nodes[i].parent;
        }
        self.simulations_run += 1;
        if self.cfg.adaptive_simulations
            && self.num_legal == 1
            && self.simulations_run >= self.soft_budget
        {
            self.finish("single_legal_action");
            return;
        }
        let check = self.simulations_run.is_multiple_of(self.cfg.check_interval)
            || self.simulations_run == self.soft_budget
            || self.simulations_run == self.hard_limit;
        if check {
            if let Some(action) = self.best_action() {
                self.history.push(action);
            }
            if self.cfg.adaptive_simulations
                && self.simulations_run >= self.soft_budget
                && self.stable_count() >= self.cfg.stability_checks
            {
                self.finish("stable_after_soft_budget");
                return;
            }
        }
        if self.simulations_run >= self.hard_limit {
            self.finish(if self.cfg.adaptive_simulations {
                "max_simulations"
            } else {
                "fixed_budget"
            });
        }
    }
    fn finish(&mut self, reason: &str) {
        self.active = false;
        let mut data = json!({
            "adaptive_search": self.cfg.adaptive_simulations, "max_simulations": self.cfg.simulations,
            "min_simulations": self.cfg.min_simulations.min(self.cfg.simulations),
            "check_interval": self.cfg.check_interval, "target_visits_per_action": self.cfg.target_visits_per_action,
            "single_action_simulations": self.cfg.single_action_simulations,
            "stability_checks_required": self.cfg.stability_checks, "num_legal_actions": self.num_legal,
            "initial_root_visits": self.initial_visits, "target_total_root_visits": self.target_total,
            "soft_budget_simulations": self.soft_budget, "actual_simulations": self.simulations_run,
            "final_root_visits": self.nodes[0].visits, "stop_reason": reason,
            "best_action_stability_checks": self.stable_count(), "root_noise_enabled": self.noise_enabled
        });
        data.as_object_mut()
            .unwrap()
            .extend(self.root_stats().as_object().unwrap().clone());
        self.metadata = Some(data);
    }
    fn root_stats(&self) -> Value {
        let mut order = self.nodes[0].children.clone();
        // Stable sorting preserves Python's first-action tie handling.
        order.sort_by_key(|&i| std::cmp::Reverse(self.nodes[i].visits));
        let n = order.len();
        let visits: Vec<u64> = order.iter().map(|&i| self.nodes[i].visits).collect();
        let total = visits.iter().sum::<u64>() as f64;
        let first = visits.first().copied().unwrap_or(0);
        let second = visits.get(1).copied().unwrap_or(0);
        let share = |sum: u64| if total > 0.0 { sum as f64 / total } else { 0.0 };
        let entropy = |values: &[f64]| {
            let sum: f64 = values.iter().sum();
            if sum <= 0.0 || n <= 1 {
                return 0.0;
            }
            -values
                .iter()
                .filter(|&&v| v > 0.0)
                .map(|v| {
                    let p = v / sum;
                    p * p.ln()
                })
                .sum::<f64>()
                / (n as f64).ln()
        };
        let priors: Vec<f64> = self.nodes[0]
            .children
            .iter()
            .map(|&i| self.nodes[i].network_prior.max(0.0))
            .collect();
        let q1 = order.first().map_or(0.0, |&i| self.nodes[i].q());
        let q2 = order.get(1).map_or(0.0, |&i| self.nodes[i].q());
        json!({
            "best_action_key": order.first().and_then(|&i| self.nodes[i].action),
            "best_action_visits": first, "second_action_visits": second,
            "top1_visit_share": share(first), "top2_visit_share": share(visits.iter().take(2).sum()),
            "top3_visit_share": share(visits.iter().take(3).sum()),
            "visited_action_fraction": if n > 0 { visits.iter().filter(|&&v| v > 0).count() as f64 / n as f64 } else { 0.0 },
            "normalized_visit_entropy": entropy(&visits.iter().map(|&v| v as f64).collect::<Vec<_>>()),
            "visit_margin": first - second, "visit_margin_ratio": share(first - second),
            "winner_locked": n > 0 && (n == 1 || first > second + self.hard_limit.saturating_sub(self.simulations_run) as u64),
            "best_action_q": q1, "second_action_q": q2, "q_gap": q1 - q2,
            "network_prior_entropy": entropy(&priors)
        })
    }
    /// Discard siblings and compact the retained subtree; flip root-perspective values if needed.
    pub fn advance(&mut self, action: u16) -> Result<(u16, bool), String> {
        if self.active {
            return Err("Finish the active search before advancing".into());
        }
        let state = self.root_state();
        if state.game_over || !self.legal(0).contains(&action) {
            return Err("Cannot advance an illegal or terminal move".into());
        }
        // Search may mark a dead-end child game_over. Apply the real move from
        // the actual root so that marker cannot become a completed game result.
        let mut actual_next = state;
        let transition = actual_next.apply_legal(action as usize);
        let selected = self.nodes[0]
            .children
            .iter()
            .copied()
            .find(|&i| self.nodes[i].action == Some(action));
        if let Some(selected) = selected {
            self.nodes[selected].state = Some(Box::new(actual_next));
            let flip =
                self.nodes[selected].state.as_ref().unwrap().current_player != self.root_player;
            let mut old_nodes: Vec<_> = std::mem::take(&mut self.nodes)
                .into_iter()
                .map(Some)
                .collect();
            let mut queue = vec![(selected, None)];
            let mut retained = Vec::new();
            let mut cursor = 0;
            while cursor < queue.len() {
                let (old_id, parent) = queue[cursor];
                let mut node = old_nodes[old_id].take().unwrap();
                node.parent = parent;
                if flip {
                    node.value = -node.value;
                }
                let start = queue.len();
                queue.extend(node.children.iter().map(|&i| (i, Some(cursor))));
                node.children = (start..queue.len()).collect();
                retained.push(node);
                cursor += 1;
            }
            self.nodes = retained;
        } else {
            self.nodes = vec![Node::new(Some(actual_next), None, Some(action), 0.0)];
        }
        self.root_player = self.root_state().current_player;
        self.metadata = None;
        self.noise = None;
        Ok(transition)
    }
    pub fn pending_snapshot(&self) -> Option<String> {
        self.pending
            .map(|i| self.nodes[i].state.as_ref().unwrap().snapshot_json())
    }
    pub fn summary(&self) -> Value {
        let children: Vec<_> = self.nodes[0]
            .children
            .iter()
            .map(|&i| {
                let c = &self.nodes[i];
                json!({"action_id": c.action, "visits": c.visits, "value": c.value,
                "prior": c.prior, "network_prior": c.network_prior, "expanded": c.expanded})
            })
            .collect();
        json!({"visits": self.nodes[0].visits, "value": self.nodes[0].value,
            "expanded": self.nodes[0].expanded, "children": children, "metadata": self.metadata,
            "best_action_id": self.best_action(), "node_count": self.nodes.len(),
            "materialized_states": self.nodes.iter().filter(|n| n.state.is_some()).count()})
    }
    /// Diagnostic only: the complete arena, including lazy-state markers.
    pub fn tree(&self) -> Value {
        let rows: Vec<_> = self.nodes.iter().map(|n| json!({
            "parent": n.parent, "action_id": n.action, "children": n.children,
            "visits": n.visits, "value": n.value, "prior": n.prior, "network_prior": n.network_prior,
            "expanded": n.expanded, "state": n.state.as_ref().map(|s| s.snapshot_json())
        })).collect();
        json!(rows)
    }
}
