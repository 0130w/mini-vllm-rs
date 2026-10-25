use cuda_core::CudaContext;
use mini_vllm::weights::load_weights;

fn main() -> std::io::Result<()> {
    // bind to GPU 0
    let ctx = CudaContext::new(0).unwrap();
    let _weights = load_weights(&ctx)?;
    Ok(())
}
