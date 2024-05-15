extern crate csv;
use std::vec;

use anyhow::Result;
use candle::{DType, Device, Tensor, D};
use rand::prelude::*;
use crate::utils;


/// Perform K-means clustering on the given data.
///
/// # Arguments
///
/// * `data` - The input data tensor.
/// * `k` - The number of clusters.
/// * `max_iter` - The maximum number of iterations.
/// * `device` - The device to perform computations on.
///
/// # Returns
///
/// A tuple containing two tensors:
/// 1. The final centroids of each cluster.
/// 2. The assignments of data points to clusters.
///
fn k_means(data: &Tensor, k: usize, max_iter: i32, device: &Device) -> Result<(Tensor, Tensor)> {
    let (n, _) = data.dims2()?;
    println!("Data dim: {:?}", n);
    let mut rng = rand::thread_rng();
    let mut indices = (0..n).collect::<Vec<_>>();
    indices.shuffle(&mut rng);

    // Initialize centroids with random data points
    let centroid_indices = indices[..k]
        .iter()
        .copied()
        .map(|x| x as i64)
        .collect::<Vec<_>>();

    let centroid_idx_tensor = Tensor::from_slice(centroid_indices.as_slice(), (k,), device)?;
    let mut centers = data.index_select(&centroid_idx_tensor, 0)?;
    let mut cluster_assignments = Tensor::zeros((n,), DType::U32, device)?;

    // Iteratively update centroids
    for _ in 0..max_iter {
        // Calculate Euclidean distance between data points and centroids
        let dist = utils::calc_squared_euclidean_dist(data, &centers)?;
        
        // Assign each data point to the nearest centroid
        cluster_assignments = dist.argmin(D::Minus1)?;
        
        // Update centroids based on mean of data points in each cluster
        let mut centers_vec = vec![];
        for i in 0..k {
            let mut indices = vec![];
            cluster_assignments
                .to_vec1::<u32>()?
                .iter()
                .enumerate()
                .for_each(|(j, x)| {
                    if *x == i as u32 {
                        indices.push(j as u32);
                    }
                });
            let indices = Tensor::from_slice(indices.as_slice(), (indices.len(),), device)?;
            let cluster_data = data.index_select(&indices, 0)?;
            let mean = cluster_data.mean(0)?;
            centers_vec.push(mean);
        }
        centers = Tensor::stack(centers_vec.as_slice(), 0)?;
    }

    Ok((centers, cluster_assignments))
}

pub fn run() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    // let device =Device::new_metal(0)?;
    let data = utils::load_dataset("/Users/akashsingh/Documents/Customer_Data.csv", &device)?;
    let start = std::time::Instant::now();
    let (centers, cluster_assignments) = k_means(&data, 3, 100, &device)?;
    println!("kmeans duration: {:?}", start.elapsed());
    println!("{}", centers);
    println!("{}", cluster_assignments);
    let cluster_sizes = cluster_assignments.to_vec1::<u32>()?;
    for i in 0..3 {
        let size = cluster_sizes.iter().filter(|&&x| x == i as u32).count();
        println!("Cluster {} size: {}", i, size);
    }
    let duration = start.elapsed();
    println!("{:?}", duration);
    Ok(())
}