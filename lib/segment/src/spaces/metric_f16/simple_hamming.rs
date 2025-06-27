use common::types::ScoreType;

use crate::data_types::vectors::{DenseVector, VectorElementTypeHalf};
use crate::spaces::metric::Metric;
#[cfg(target_arch = "x86_64")]
use crate::spaces::metric_f16::avx::hamming::avx_hamming_similarity_half;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::spaces::metric_f16::neon::hamming::neon_hamming_similarity_half;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::spaces::metric_f16::sse::hamming::sse_hamming_similarity_half;
#[cfg(target_arch = "x86_64")]
use crate::spaces::simple::MIN_DIM_SIZE_AVX;
use crate::spaces::simple::{MIN_DIM_SIZE_SIMD, HammingMetric};
use crate::types::Distance;

impl Metric<VectorElementTypeHalf> for HammingMetric {
    fn distance() -> Distance {
        Distance::Hamming
    }

    fn similarity(v1: &[VectorElementTypeHalf], v2: &[VectorElementTypeHalf]) -> ScoreType {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c")
                && v1.len() >= MIN_DIM_SIZE_AVX
            {
                return unsafe { avx_hamming_similarity_half(v1, v2) };
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") && v1.len() >= MIN_DIM_SIZE_SIMD {
                return unsafe { sse_hamming_similarity_half(v1, v2) };
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon")
                && std::arch::is_aarch64_feature_detected!("fp16")
                && v1.len() >= MIN_DIM_SIZE_SIMD
            {
                return unsafe { neon_hamming_similarity_half(v1, v2) };
            }
        }

        hamming_similarity_half(v1, v2)
    }

    fn preprocess(vector: DenseVector) -> DenseVector {
        vector
    }
}

pub fn hamming_similarity_half(
    v1: &[VectorElementTypeHalf],
    v2: &[VectorElementTypeHalf],
) -> ScoreType {
    -v1.iter()
        .zip(v2)
        .map(|(a, b)| if (a - b).to_f32().abs() < f32::EPSILON { 0.0 } else { 1.0 })
        .sum::<f32>()
}
