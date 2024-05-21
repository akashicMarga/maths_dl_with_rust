use candle::{ Device, Tensor, D};
use anyhow::Result;
use rayon::prelude::*;


pub fn load_dataset(file_path: &str, device: &Device) -> Result<Tensor> {
    let mut rdr = csv::Reader::from_path(file_path)?;
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let mut row = vec![];
        for i in 1..5 {
            row.push(record[i].parse::<f32>()?);
        }
        data.push(row);
    }
    let feature_cnt = data[0].len();
    let sample_cnt = data.len();
    let data = data.into_par_iter().flatten().collect::<Vec<_>>();
    let data = Tensor::from_slice(data.as_slice(), (sample_cnt, feature_cnt), device)?;
    Ok(data)
}

pub fn normalize(data: &Tensor) -> Result<Tensor> {
    // This code is implementing Z-scoring, a common technique in statistics to normalize continuous data.
    // This process is often referred to as Z-scoring because the normalized values will have a mean of 0 
    // and a standard deviation of 1 (assuming the input data was centered around its mean).
    
    // Calculate the mean of the input data.
    let data_mean = data.mean(0)?;
    // Calculate the squared differences between each data point and the mean.
    let diff_data_mean = data.broadcast_sub(&data_mean)?.sqr()?;
    // Calculate the variance (average of the squared differences).
    let variance = diff_data_mean.mean(0)?;
    // Calculate the standard deviation by taking the square root of the variance.
    let standard_deviation = variance.sqrt()?;
    // Normalize each data point by subtracting the mean and then dividing by the standard deviation.
    let normalized = data.broadcast_sub(&data_mean)?.broadcast_div(&standard_deviation)?;
    Ok(normalized)
}

pub fn cal_covariance(data: &Tensor, device: &Device) -> Result<Tensor> {

    //calculate mean across dim 0
    let mean = data.mean(0)?;
    //subtract mean from data
    let centered = data.broadcast_sub(&mean)?;
    //get the column of data
    let (m, _) = data.shape().dims2()?;
    // calculate covariance
    let covariance_matrix = centered
        .transpose(D::Minus1, D::Minus2)?
        .matmul(&centered)?
        .broadcast_div(&Tensor::new(m as f32, device)?)?;

    Ok(covariance_matrix)
}

pub fn calc_euclidean_dist(x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let x1 = x1.unsqueeze(0)?;
    let x2 = x2.unsqueeze(1)?;
    Ok(x1
        .broadcast_sub(&x2)?
        .sqr()?
        .sum(D::Minus1)?
        .sqrt()?
        .transpose(D::Minus1, D::Minus2)?)
}

pub fn calc_squared_euclidean_dist(x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let x1 = x1.unsqueeze(0)?;
    let x2 = x2.unsqueeze(1)?;
    let diff = x1.broadcast_sub(&x2)?;
    let squared_dist = diff.sqr()?.sum(D::Minus1)?;
    Ok(squared_dist.transpose(D::Minus1, D::Minus2)?)
}
