pub mod engine;
pub mod search;
mod snapshot;
#[rustfmt::skip]
mod tables;

#[cfg(feature = "python")]
mod bindings {
    use super::{engine, search, tables};
    use engine::State;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyclass(name = "RustState", module = "splendor_rust")]
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
    fn splendor_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<NativeState>()?;
        module.add_class::<NativeSearch>()?;
        module.add("ACTION_SPACE_SIZE", tables::ACTION_SPACE_SIZE)?;
        module.add("OBSERVATION_SIZE", engine::OBSERVATION_SIZE)?;
        module.add("STATE_BYTES", std::mem::size_of::<State>())?;
        Ok(())
    }

    #[pyclass(name = "RustSearch", module = "splendor_rust")]
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
