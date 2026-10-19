use std::{fs::File, io::Read};

fn main() -> std::io::Result<()> {
    let mut safetensor_file =
        File::open("models/qwen-3.5-0.8B/model.safetensors-00001-of-00001.safetensors")?; // contains
    // header_size, header and tensor
    let mut bytes = [0u8; 8];
    safetensor_file.read_exact(&mut bytes)?;
    let header_size = u64::from_le_bytes(bytes);
    println!("header_size: {header_size}"); // always 8 bytes
    Ok(())
}
