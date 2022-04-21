pub fn memchr_sse2(n1: u8, haystack: &[u8]) -> Option<usize> {
    unsafe { sse2::memchr(n1, haystack) }
}

// Code is copied from memchr crate

mod sse2 {
    /// Subtract `b` from `a` and return the difference. `a` should be greater than
    /// or equal to `b`.
    #[inline]
    fn sub(a: *const u8, b: *const u8) -> usize {
        debug_assert!(a >= b);
        (a as usize) - (b as usize)
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn forward_search1(
        start_ptr: *const u8,
        end_ptr: *const u8,
        ptr: *const u8,
        vn1: __m128i,
    ) -> Option<usize> {
        debug_assert!(sub(end_ptr, start_ptr) >= VECTOR_SIZE);
        debug_assert!(start_ptr <= ptr);
        debug_assert!(ptr <= end_ptr.sub(VECTOR_SIZE));

        let chunk = _mm_loadu_si128(ptr as *const __m128i);
        let mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, vn1));
        if mask != 0 {
            Some(sub(ptr, start_ptr) + forward_pos(mask))
        } else {
            None
        }
    }

    /// Compute the position of the first matching byte from the given mask. The
    /// position returned is always in the range [0, 15].
    ///
    /// The mask given is expected to be the result of _mm_movemask_epi8.
    fn forward_pos(mask: i32) -> usize {
        // We are dealing with little endian here, where the most significant byte
        // is at a higher address. That means the least significant bit that is set
        // corresponds to the position of our first matching byte. That position
        // corresponds to the number of zeros after the least significant bit.
        mask.trailing_zeros() as usize
    }

    use std::{arch::x86_64::*, cmp, mem::size_of};

    const VECTOR_SIZE: usize = size_of::<__m128i>();
    const VECTOR_ALIGN: usize = VECTOR_SIZE - 1;

    // The number of bytes to loop at in one iteration of memchr/memrchr.
    const LOOP_SIZE: usize = 4 * VECTOR_SIZE;

    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn memchr(n1: u8, haystack: &[u8]) -> Option<usize> {
        // What follows is a fast SSE2-only algorithm to detect the position of
        // `n1` in `haystack` if it exists. From what I know, this is the "classic"
        // algorithm. I believe it can be found in places like glibc and Go's
        // standard library. It appears to be well known and is elaborated on in
        // more detail here: https://gms.tf/stdfind-and-memchr-optimizations.html
        //
        // While this routine is very long, the basic idea is actually very simple
        // and can be expressed straight-forwardly in pseudo code:
        //
        //     needle = (n1 << 15) | (n1 << 14) | ... | (n1 << 1) | n1
        //     // Note: shift amount in bytes
        //
        //     while i <= haystack.len() - 16:
        //       // A 16 byte vector. Each byte in chunk corresponds to a byte in
        //       // the haystack.
        //       chunk = haystack[i:i+16]
        //       // Compare bytes in needle with bytes in chunk. The result is a 16
        //       // byte chunk where each byte is 0xFF if the corresponding bytes
        //       // in needle and chunk were equal, or 0x00 otherwise.
        //       eqs = cmpeq(needle, chunk)
        //       // Return a 32 bit integer where the most significant 16 bits
        //       // are always 0 and the lower 16 bits correspond to whether the
        //       // most significant bit in the correspond byte in `eqs` is set.
        //       // In other words, `mask as u16` has bit i set if and only if
        //       // needle[i] == chunk[i].
        //       mask = movemask(eqs)
        //
        //       // Mask is 0 if there is no match, and non-zero otherwise.
        //       if mask != 0:
        //         // trailing_zeros tells us the position of the least significant
        //         // bit that is set.
        //         return i + trailing_zeros(mask)
        //
        //     // haystack length may not be a multiple of 16, so search the rest.
        //     while i < haystack.len():
        //       if haystack[i] == n1:
        //         return i
        //
        //     // No match found.
        //     return NULL
        //
        // In fact, we could loosely translate the above code to Rust line-for-line
        // and it would be a pretty fast algorithm. But, we pull out all the stops
        // to go as fast as possible:
        //
        // 1. We use aligned loads. That is, we do some finagling to make sure our
        //    primary loop not only proceeds in increments of 16 bytes, but that
        //    the address of haystack's pointer that we dereference is aligned to
        //    16 bytes. 16 is a magic number here because it is the size of SSE2
        //    128-bit vector. (For the AVX2 algorithm, 32 is the magic number.)
        //    Therefore, to get aligned loads, our pointer's address must be evenly
        //    divisible by 16.
        // 2. Our primary loop proceeds 64 bytes at a time instead of 16. It's
        //    kind of like loop unrolling, but we combine the equality comparisons
        //    using a vector OR such that we only need to extract a single mask to
        //    determine whether a match exists or not. If so, then we do some
        //    book-keeping to determine the precise location but otherwise mush on.
        // 3. We use our "chunk" comparison routine in as many places as possible,
        //    even if it means using unaligned loads. In particular, if haystack
        //    starts with an unaligned address, then we do an unaligned load to
        //    search the first 16 bytes. We then start our primary loop at the
        //    smallest subsequent aligned address, which will actually overlap with
        //    previously searched bytes. But we're OK with that. We do a similar
        //    dance at the end of our primary loop. Finally, to avoid a
        //    byte-at-a-time loop at the end, we do a final 16 byte unaligned load
        //    that may overlap with a previous load. This is OK because it converts
        //    a loop into a small number of very fast vector instructions.
        //
        // The primary downside of this algorithm is that it's effectively
        // completely unsafe. Therefore, we have to be super careful to avoid
        // undefined behavior:
        //
        // 1. We use raw pointers everywhere. Not only does dereferencing a pointer
        //    require the pointer to be valid, but we actually can't even store the
        //    address of an invalid pointer (unless it's 1 past the end of
        //    haystack) without sacrificing performance.
        // 2. _mm_loadu_si128 is used when you don't care about alignment, and
        //    _mm_load_si128 is used when you do care. You cannot use the latter
        //    on unaligned pointers.
        // 3. We make liberal use of debug_assert! to check assumptions.
        // 4. We make a concerted effort to stick with pointers instead of indices.
        //    Indices are nicer because there's less to worry about with them (see
        //    above about pointer offsets), but I could not get the compiler to
        //    produce as good of code as what the below produces. In any case,
        //    pointers are what we really care about here, and alignment is
        //    expressed a bit more naturally with them.
        //
        // In general, most of the algorithms in this crate have a similar
        // structure to what you see below, so this comment applies fairly well to
        // all of them.

        let vn1 = _mm_set1_epi8(n1 as i8);
        let len = haystack.len();
        let loop_size = cmp::min(LOOP_SIZE, len);
        let start_ptr = haystack.as_ptr();
        let end_ptr = start_ptr.add(haystack.len());
        let mut ptr = start_ptr;

        if haystack.len() < VECTOR_SIZE {
            while ptr < end_ptr {
                if *ptr == n1 {
                    return Some(sub(ptr, start_ptr));
                }
                ptr = ptr.offset(1);
            }
            return None;
        }

        if let Some(i) = forward_search1(start_ptr, end_ptr, ptr, vn1) {
            return Some(i);
        }

        ptr = ptr.add(VECTOR_SIZE - (start_ptr as usize & VECTOR_ALIGN));
        debug_assert!(ptr > start_ptr && end_ptr.sub(VECTOR_SIZE) >= start_ptr);
        while loop_size == LOOP_SIZE && ptr <= end_ptr.sub(loop_size) {
            debug_assert_eq!(0, (ptr as usize) % VECTOR_SIZE);

            let a = _mm_load_si128(ptr as *const __m128i);
            let b = _mm_load_si128(ptr.add(VECTOR_SIZE) as *const __m128i);
            let c = _mm_load_si128(ptr.add(2 * VECTOR_SIZE) as *const __m128i);
            let d = _mm_load_si128(ptr.add(3 * VECTOR_SIZE) as *const __m128i);
            let eqa = _mm_cmpeq_epi8(vn1, a);
            let eqb = _mm_cmpeq_epi8(vn1, b);
            let eqc = _mm_cmpeq_epi8(vn1, c);
            let eqd = _mm_cmpeq_epi8(vn1, d);
            let or1 = _mm_or_si128(eqa, eqb);
            let or2 = _mm_or_si128(eqc, eqd);
            let or3 = _mm_or_si128(or1, or2);
            if _mm_movemask_epi8(or3) != 0 {
                let mut at = sub(ptr, start_ptr);
                let mask = _mm_movemask_epi8(eqa);
                if mask != 0 {
                    return Some(at + forward_pos(mask));
                }

                at += VECTOR_SIZE;
                let mask = _mm_movemask_epi8(eqb);
                if mask != 0 {
                    return Some(at + forward_pos(mask));
                }

                at += VECTOR_SIZE;
                let mask = _mm_movemask_epi8(eqc);
                if mask != 0 {
                    return Some(at + forward_pos(mask));
                }

                at += VECTOR_SIZE;
                let mask = _mm_movemask_epi8(eqd);
                debug_assert!(mask != 0);
                return Some(at + forward_pos(mask));
            }
            ptr = ptr.add(loop_size);
        }
        while ptr <= end_ptr.sub(VECTOR_SIZE) {
            debug_assert!(sub(end_ptr, ptr) >= VECTOR_SIZE);

            if let Some(i) = forward_search1(start_ptr, end_ptr, ptr, vn1) {
                return Some(i);
            }
            ptr = ptr.add(VECTOR_SIZE);
        }
        if ptr < end_ptr {
            debug_assert!(sub(end_ptr, ptr) < VECTOR_SIZE);
            ptr = ptr.sub(VECTOR_SIZE - sub(end_ptr, ptr));
            debug_assert_eq!(sub(end_ptr, ptr), VECTOR_SIZE);

            return forward_search1(start_ptr, end_ptr, ptr, vn1);
        }
        None
    }
}
