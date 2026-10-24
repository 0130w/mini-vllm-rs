use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod embeddings {
    use cuda_device::kernel;

    #[kernel]
    pub fn qwen_3_5_embedding() {
        unimplemented!()
    }
}
