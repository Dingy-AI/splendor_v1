pub mod arena;
pub mod engine;
pub mod search;
mod snapshot;
#[rustfmt::skip]
mod tables;

#[cfg(feature = "python")]
mod bindings {
    use super::{arena, engine, search, tables};
    use engine::State;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyclass(name = "RustState", module = "splendor_rust_v2")]
    struct NativeState {
        state: State,
    }

    #[pymethods]
    impl NativeState {
        #[new]
        fn new(snapshot_json: &str) -> PyResult<Self> {
            Ok(Self {
                state: State::from_json(snapshot_json).map_err(PyValueError::new_err)?,
            })
        }

        fn clone(&self) -> Self {
            Self { state: self.state }
        }
        fn snapshot_json(&self) -> String {
            self.state.snapshot_json()
        }
        fn legal_action_ids(&self) -> Vec<u16> {
            self.state.legal_action_ids()
        }
        fn observation(&self) -> Vec<f32> {
            self.state.observation().to_vec()
        }
        fn step(&mut self, action_id: usize) -> PyResult<(u16, bool)> {
            if self.state.turn_number == u32::MAX {
                return Err(PyValueError::new_err("Turn counter is out of range"));
            }
            self.state.step(action_id).map_err(PyValueError::new_err)
        }
        #[getter]
        fn current_player(&self) -> usize {
            self.state.current_player
        }
        #[getter]
        fn node_type(&self) -> u8 {
            self.state.node_type
        }
        #[getter]
        fn turn_number(&self) -> u32 {
            self.state.turn_number
        }
        #[getter]
        fn game_over(&self) -> bool {
            self.state.game_over
        }
        #[getter]
        fn winners(&self) -> Vec<usize> {
            (0..2).filter(|&i| self.state.winners[i]).collect()
        }
    }

    #[pymodule]
    fn splendor_rust_v2(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<NativeState>()?;
        module.add_class::<NativeSearch>()?;
        module.add_class::<NativeArena>()?;
        module.add("ACTION_SPACE_SIZE", tables::ACTION_SPACE_SIZE)?;
        module.add("OBSERVATION_SIZE", engine::OBSERVATION_SIZE)?;
        module.add("STATE_BYTES", std::mem::size_of::<State>())?;
        Ok(())
    }

    type PackedBatch<'py> = (
        u64,
        Vec<usize>,
        Vec<usize>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        usize,
    );

    #[pyclass(name = "RustArena", module = "splendor_rust_v2")]
    struct NativeArena {
        arena: arena::Arena,
    }

    #[pymethods]
    impl NativeArena {
        #[new]
        fn new(capacity: usize) -> PyResult<Self> {
            Ok(Self {
                arena: arena::Arena::new(capacity).map_err(PyValueError::new_err)?,
            })
        }
        #[pyo3(signature = (slot, state, config_json="{}"))]
        fn add(&mut self, slot: usize, state: &NativeState, config_json: &str) -> PyResult<()> {
            let config = search::Config::from_json(config_json).map_err(PyValueError::new_err)?;
            self.arena
                .add(slot, state.state, config)
                .map_err(PyValueError::new_err)
        }
        fn remove(&mut self, slot: usize) -> PyResult<()> {
            self.arena.remove(slot).map_err(PyValueError::new_err)
        }
        #[pyo3(signature = (slot, noise=None))]
        fn begin(&mut self, slot: usize, noise: Option<Vec<f64>>) -> PyResult<()> {
            self.arena.begin(slot, noise).map_err(PyValueError::new_err)
        }
        fn gather<'py>(&mut self, py: Python<'py>, max_batch: usize) -> PyResult<PackedBatch<'py>> {
            let batch = py
                .detach(|| self.arena.gather(max_batch))
                .map_err(PyValueError::new_err)?;
            Ok((
                batch.token,
                batch.slots,
                batch.ready,
                PyBytes::new(py, &batch.observations),
                PyBytes::new(py, &batch.actions),
                PyBytes::new(py, &batch.mask),
                batch.width,
            ))
        }
        #[pyo3(signature = (token, payload, double=false))]
        fn respond(
            &mut self,
            py: Python<'_>,
            token: u64,
            payload: &[u8],
            double: bool,
        ) -> PyResult<()> {
            py.detach(|| self.arena.respond(token, payload, double))
                .map_err(PyValueError::new_err)
        }
        fn advance(
            &mut self,
            py: Python<'_>,
            slot: usize,
            action_id: u16,
        ) -> PyResult<(u16, bool)> {
            py.detach(|| self.arena.advance(slot, action_id))
                .map_err(PyValueError::new_err)
        }
        fn root_state(&self, slot: usize) -> PyResult<NativeState> {
            Ok(NativeState {
                state: self
                    .arena
                    .tree(slot)
                    .map_err(PyValueError::new_err)?
                    .root_state(),
            })
        }
        fn root_action_ids(&mut self, slot: usize) -> PyResult<Vec<u16>> {
            Ok(self
                .arena
                .tree_mut(slot)
                .map_err(PyValueError::new_err)?
                .root_actions())
        }
        fn summary_json(&self, slot: usize) -> PyResult<String> {
            Ok(self
                .arena
                .tree(slot)
                .map_err(PyValueError::new_err)?
                .summary()
                .to_string())
        }
        fn tree_json(&self, slot: usize) -> PyResult<String> {
            Ok(self
                .arena
                .tree(slot)
                .map_err(PyValueError::new_err)?
                .tree()
                .to_string())
        }
        fn active(&self, slot: usize) -> PyResult<bool> {
            Ok(self
                .arena
                .tree(slot)
                .map_err(PyValueError::new_err)?
                .is_active())
        }
    }

    #[pyclass(name = "RustSearch", module = "splendor_rust_v2")]
    struct NativeSearch {
        search: search::Search,
    }

    #[pymethods]
    impl NativeSearch {
        #[new]
        #[pyo3(signature = (state, config_json="{}"))]
        fn new(state: &NativeState, config_json: &str) -> PyResult<Self> {
            let config = search::Config::from_json(config_json).map_err(PyValueError::new_err)?;
            Ok(Self {
                search: search::Search::new(state.state, config),
            })
        }
        #[pyo3(signature = (noise=None))]
        fn begin(&mut self, noise: Option<Vec<f64>>) -> PyResult<()> {
            self.search.begin(noise).map_err(PyValueError::new_err)
        }
        fn next_request(&mut self, py: Python<'_>) -> Option<search::Request> {
            py.detach(|| self.search.next_request())
        }
        fn respond(&mut self, py: Python<'_>, priors: Vec<f64>, value: f64) -> PyResult<()> {
            py.detach(|| self.search.respond(priors, value))
                .map_err(PyValueError::new_err)
        }
        fn advance(&mut self, py: Python<'_>, action_id: u16) -> PyResult<(u16, bool)> {
            py.detach(|| self.search.advance(action_id))
                .map_err(PyValueError::new_err)
        }
        fn root_state(&self) -> NativeState {
            NativeState {
                state: self.search.root_state(),
            }
        }
        fn root_action_ids(&mut self) -> Vec<u16> {
            self.search.root_actions()
        }
        fn summary_json(&self) -> String {
            self.search.summary().to_string()
        }
        fn tree_json(&self) -> String {
            self.search.tree().to_string()
        }
        fn pending_snapshot_json(&self) -> Option<String> {
            self.search.pending_snapshot()
        }
        #[getter]
        fn active(&self) -> bool {
            self.search.is_active()
        }
    }
}
