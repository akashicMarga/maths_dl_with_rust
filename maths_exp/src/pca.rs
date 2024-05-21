use anyhow::{Ok, Result};
use candle::{Device, Tensor, D};
use nalgebra::linalg::SymmetricEigen;
use nalgebra::DMatrix;
use std::time::Instant;
use crate::utils;
use rayon::prelude::*;

fn pca(normalized_data: &Tensor, device: &Device, variance: f32) -> Result<Tensor> {
    let (_, n) = normalized_data.shape().dims2()?;
    let cov = utils::cal_covariance(normalized_data, &device)?;
    let vec: Vec<f32> = cov
        .to_device(&device)?
        .to_vec2()?
        .into_par_iter()
        .flatten()
        .collect();
    let dmatrix = DMatrix::from_vec(n, n, vec);
    let eig = SymmetricEigen::new(dmatrix);
    let eigen_values = eig.eigenvalues.data.as_vec();
    let total = eigen_values.par_iter().sum::<f32>();
    let mut k = 0;
    for i in 0..n {
        let var = eigen_values[0..i].par_iter().sum::<f32>() / total;
        if var > variance {
            println!("{} components explain {}% of the variance", i, var * 100.0);
            k = i;
            break;
        }
    }

    let eigen_vectors = eig.eigenvectors.data.as_vec();
    let eigen_vectors = eigen_vectors
        .chunks(n)
        .take(k)
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let eigen_vectors = Tensor::from_slice(eigen_vectors.as_slice(), (k, n), device)?;
    Ok(eigen_vectors)
}

pub fn run() -> Result<()> {
    
    // let device =Device::new_metal(0)?;
    let device = Device::cuda_if_available(0)?;
    let data = utils::load_dataset("/Users/akashsingh/Documents/Customer_Data.csv", &device)?;
    println!("Data shape: {:?}", data.shape());
    let start = Instant::now();
    let normalized_data = utils::normalize(&data)?;
    let reduce = pca(&normalized_data, &device, 0.90)?;
    println!("Reduced Data shape: {:?}", reduce.shape());
    let compressed_data = data.matmul(&reduce.transpose(D::Minus1, D::Minus2)?)?;
    let duration = start.elapsed();
    println!("Compressed data {:?} with shape {:?} in {:?} time", compressed_data, compressed_data,duration);
    Ok(())
}