use mini_vllm::weights::load_weights;

fn main() -> std::io::Result<()> {
    let _weights = load_weights()?;
    Ok(())
}
