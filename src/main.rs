use core::slice;
use std::arch::x86_64::*;
use std::time::Duration;
use std::time::Instant;

mod memchr2;

use rand::Rng;

pub unsafe fn memchr_unroll_sse2_overlapping_idx(n1: u8, haystack: &[u8]) -> Option<usize> {
    const UNROLL_SIZE: usize = 1;

    let mut idx = 0;

    if haystack.len() < 16 {
        while idx != haystack.len() {
            if n1 == *haystack.get_unchecked(idx) {
                return Some(idx);
            }
            idx += 1;
        }
        return None;
    }

    // Can now assume that haystack.len() >= 16

    let n1v = _mm_set1_epi8(n1 as i8);

    let first_iter = _mm_loadu_si128(haystack.as_ptr().cast());

    let cmp = _mm_cmpeq_epi8(n1v, first_iter);
    let mask = _mm_movemask_epi8(cmp);

    if mask != 0 {
        return Some(mask.trailing_zeros() as usize);
    }

    // Conduct aligned, overlapping search
    // aligned_haystack > haystack (never equal)
    let aligned_haystack = ((16 * (haystack.as_ptr() as usize / 16)) + 16) as *const u8;
    let nonoverlapping_elements = aligned_haystack as usize - haystack.as_ptr() as usize;
    let aligned_haystack =
        slice::from_raw_parts(aligned_haystack, haystack.len() - nonoverlapping_elements);

    let mut cmps = [_mm_setzero_si128(); UNROLL_SIZE];

    let mut ptr = aligned_haystack.as_ptr();

    let end_ptr = aligned_haystack
        .as_ptr()
        .add((16 * UNROLL_SIZE) * (aligned_haystack.len() / (16 * UNROLL_SIZE)));

    while ptr != end_ptr {
        (0..UNROLL_SIZE)
            .map(|i| _mm_load_si128(ptr.add(16 * i).cast()))
            .zip(cmps.iter_mut())
            .for_each(|(src, cmp)| *cmp = src);

        cmps.iter_mut()
            .for_each(|cmp| *cmp = _mm_cmpeq_epi8(n1v, *cmp));

        let reduced_cmp = cmps
            .iter()
            .copied()
            .reduce(|a, b| _mm_or_si128(a, b))
            .unwrap();

        let mask = _mm_movemask_epi8(reduced_cmp);
        if mask != 0 {
            for (reg_index, cmp) in cmps.iter().copied().enumerate() {
                let submask = _mm_movemask_epi8(cmp);

                if submask != 0 {
                    return Some(
                        ptr as usize - haystack.as_ptr() as usize
                            + reg_index * 16
                            + submask.trailing_zeros() as usize,
                    );
                }
            }
        }

        ptr = ptr.add(UNROLL_SIZE * 16);
    }

    // Do remaining aligned chunks
    // residual chunks
    let aligned_16_chunks = aligned_haystack.len() / 16;
    let residual_aligned_elements = 16 * aligned_16_chunks;
    let end_ptr = aligned_haystack.as_ptr().add(residual_aligned_elements);

    while ptr != end_ptr {
        let cmp = _mm_cmpeq_epi8(_mm_load_si128(ptr.cast()), n1v);
        let mask = _mm_movemask_epi8(cmp);

        if mask != 0 {
            return Some(
                ptr as usize - haystack.as_ptr() as usize + mask.trailing_zeros() as usize,
            );
        }

        ptr = ptr.add(16);
    }

    // Last residual chunk
    if haystack.len() % 16 != 0 {
        let last_chunk_ptr = haystack.as_ptr().add(haystack.len()).offset(-16);
        let cmp = _mm_cmpeq_epi8(_mm_loadu_si128(last_chunk_ptr.cast()), n1v);
        let mask = _mm_movemask_epi8(cmp);

        if mask != 0 {
            return Some(
                last_chunk_ptr as usize - haystack.as_ptr() as usize
                    + mask.trailing_zeros() as usize,
            );
        }
    }

    None
}

fn main() {
    let mut rng = rand::thread_rng();

    let max = 2048;
    let mut buffer = vec![0; max];

    /* Find errors in implementation */
    /* for iter in 0.. {
        let size = rng.gen_range(0..=max);
        let buffer = &mut buffer[..size];

        buffer.fill_with(|| rng.gen());

        let x = rng.gen();

        let (a, b) = (
            unsafe { memchr_unroll_sse2_overlapping_idx(x, &buffer) },
            memchr::memchr(x, &buffer),
        );

        if a != b {
            dbg!(buffer, x);
            panic!("ASSERTION FAILED: SEE TEST CASE");
        } else {
            println!("successful iter {}", iter);
        }
    } */
    // --------------------------------------

    let mut instant;
    let mut a_time = Duration::from_secs(0);
    let mut b_time = Duration::from_secs(0);

    let iters = 100000;

    for _iter in 0..iters {
        let size = rng.gen_range(0..=max);
        let buffer = &mut buffer[..size];

        buffer.fill_with(|| rng.gen());

        let x = rng.gen();

        instant = Instant::now();
        let _a = unsafe { memchr_unroll_sse2_overlapping_idx(x, &buffer) };
        a_time += instant.elapsed();
    }

    for _iter in 0..iters {
        let size = rng.gen_range(0..=max);
        let buffer = &mut buffer[..size];

        buffer.fill_with(|| rng.gen());

        let x = rng.gen();

        instant = Instant::now();
        let _b = memchr2::memchr_sse2(x, &buffer);
        b_time += instant.elapsed();
    }

    dbg!(a_time, b_time);
    if a_time < b_time {
        println!("memchr_unroll_sse2_overlapping_idx was faster");
    } else {
        println!("memchr2::memchr_sse2 was faster");
    }
    println!(
        "Perf ratio: {:.2}%",
        100.0 * (b_time.as_secs_f64() / a_time.as_secs_f64())
    );
}
