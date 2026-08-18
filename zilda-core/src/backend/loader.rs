use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;
use std::path::Path;

use super::{Config, ZildaMoeBackend};

pub struct ModelLoader;

impl ModelLoader {
    pub fn load_from_safetensors<P: AsRef<Path>>(
        weights_path: P,
        config: &Config,
        device: &Device,
    ) -> Result<ZildaMoeBackend> {
        let path = weights_path.as_ref();
        println!("[Loader] Chargement de Safetensors depuis {:?}", path);

        let filenames = vec![path.to_path_buf()];
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, device)?
        };

        ZildaMoeBackend::load(vb, config)
    }
}