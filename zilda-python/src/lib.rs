use pyo3::prelude::*;

#[pyclass]
pub struct ZildaEngine {
    #[pyo3(get)]
    pub model_name: String,
}

#[pymethods]
impl ZildaEngine {
    #[new]
    fn new(model_name: String) -> Self {
        ZildaEngine { model_name }
    }

    fn initialize(&self) -> PyResult<String> {
        Ok(format!(
            "[Zilda-Core] Moteur initialisé pour le modèle : {}",
            self.model_name
        ))
    }

    fn process_prompt(&self, prompt: String) -> PyResult<String> {
        if prompt.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("Le prompt ne peut pas être vide."));
        }
        Ok(format!("[Zilda-Core] Traitement du prompt : '{}' terminé avec succès.", prompt))
    }
}

#[pymodule]
fn zilda_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ZildaEngine>()?;
    Ok(())
}