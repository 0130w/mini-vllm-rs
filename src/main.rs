use cuda_core::CudaContext;
use mini_vllm::{models::qwen3_5::config::Qwen3_5Config, weights::load_weights};

fn main() -> std::io::Result<()> {
    // bind to GPU 0
    let ctx = CudaContext::new(0).unwrap();
    let config: Qwen3_5Config =
        serde_json::from_reader(std::fs::File::open("models/qwen-3.5-0.8B/config.json")?)?;
    let _weights = load_weights(&ctx, &config)?;
    Ok(())
}
