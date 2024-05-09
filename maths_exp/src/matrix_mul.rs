use anyhow::Ok;
use candle::{Device, Tensor};
use std::time::Instant;

pub fn matrix_mul() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let start = Instant::now();
    let c = a.matmul(&b)?;
    let duration = start.elapsed();

    println!("Matrix c:\n{c}\nTime taken: {:?}", duration);

    Ok(())
}

