use std::arch::x86_64::*;

use crate::spaces::simple_avx::hsum256_ps_avx;

#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn avx_hamming_similarity_bytes(v1: &[u8], v2: &[u8]) -> f32 {
    debug_assert!(v1.len() == v2.len());
    debug_assert!(is_x86_feature_detected!("avx"));
    debug_assert!(is_x86_feature_detected!("avx2"));
    debug_assert!(is_x86_feature_detected!("fma"));

    let mut ptr1: *const u8 = v1.as_ptr();
    let mut ptr2: *const u8 = v2.as_ptr();

    unsafe {
        // sum accumulator for 8x32 bit integers
        let mut acc = _mm256_setzero_si256();
        let len = v1.len();
        for _ in 0..len / 32 {
            // load 32 bytes
            let p1 = _mm256_loadu_si256(ptr1.cast::<__m256i>());
            let p2 = _mm256_loadu_si256(ptr2.cast::<__m256i>());
            ptr1 = ptr1.add(32);
            ptr2 = ptr2.add(32);

            // XOR to find differing bits
            let xor = _mm256_xor_si256(p1, p2);
            
            // Count population (number of set bits) in each byte
            // We need to count bits in each byte and accumulate
            let mut bit_count = _mm256_setzero_si256();
            
            // Use a lookup table approach for popcount of each byte
            let lookup = _mm256_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
            );
            
            let mask = _mm256_set1_epi8(0x0F);
            let lo = _mm256_and_si256(xor, mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(xor, 4), mask);
            
            let popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
            let popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
            bit_count = _mm256_add_epi8(popcnt_lo, popcnt_hi);
            
            // Sum bytes horizontally using SAD against zero
            let sad = _mm256_sad_epu8(bit_count, _mm256_setzero_si256());
            acc = _mm256_add_epi32(acc, sad);
        }

        // convert 8x32 bit integers into 8x32 bit floats and calculate horizontal sum
        let mul_ps = _mm256_cvtepi32_ps(acc);
        let mut score = hsum256_ps_avx(mul_ps);

        let remainder = len % 32;
        if remainder != 0 {
            let mut remainder_score = 0;
            for _ in 0..remainder {
                let v1 = *ptr1;
                let v2 = *ptr2;
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
                // Count differing bits
                remainder_score += (v1 ^ v2).count_ones();
            }
            score += remainder_score as f32;
        }

        -score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::metric_uint::simple_hamming::hamming_similarity_bytes;

    #[test]
    fn test_spaces_avx() {
        if is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            let v1: Vec<u8> = vec![
                255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255, 0, 1, 2, 3,
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17,
            ];
            let v2: Vec<u8> = vec![
                255, 255, 0, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241,
                240, 239, 238, 255, 255, 255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245,
                244, 243, 242, 241, 240, 239, 238, 255, 255, 255, 254, 253, 252, 251, 250, 249,
                248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 255, 255, 255, 254, 253,
                252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 255,
                255, 255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241,
                240, 239, 238,
            ];

            let hamming_simd = unsafe { avx_hamming_similarity_bytes(&v1, &v2) };
            let hamming = hamming_similarity_bytes(&v1, &v2);
            assert_eq!(hamming_simd, hamming);
        } else {
            println!("avx test skipped");
        }
    }
}
