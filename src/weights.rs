use std::{io::Result, sync::Arc};

use cuda_core::CudaContext;

pub use crate::models::qwen3_5::weights::ModelWeights;
use crate::{file_loader::SafetensorsLoader, models::qwen3_5::config::Qwen3_5Config};

/// Load the text-model weights for the current Qwen checkpoint.
pub fn load_weights(ctx: &Arc<CudaContext>, config: &Qwen3_5Config) -> Result<ModelWeights> {
    let path: &str = "models/qwen-3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
    let mut loader = SafetensorsLoader::open(path, Arc::clone(ctx))?;
    crate::models::qwen3_5::weights::load_weights(&mut loader, &config.text_config)
}
