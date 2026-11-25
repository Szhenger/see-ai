#ifndef SEEML_SOURCE_PLAN_INSTRUCTION_H_
#define SEEML_SOURCE_PLAN_INSTRUCTION_H_

#include <cstddef>
#include <bit>
#include <cmath>
#include <cstdint>

// =============================================================================
// instruction/ discipline of the plan ABI: the update instruction set —
// tensor references, the opcode vocabulary, and the fixed 64-byte
// instruction word the compiler's lowering emits and the runtime's
// dispatcher executes.
// =============================================================================

namespace seeml::update {

// An operand word that carries two 32-bit fields (B<<32|S, H<<32|d,
// rows<<32|cols, V<<32|D, or two f32 bit patterns) is taken apart and put
// together here and nowhere else (S8 #154). The halves are returned as the
// 64-bit values the validator's overflow-checked products and the kernels'
// loop bounds take; the *32 forms are the same halves in the width a GPU
// argument block stores. PackWords does not mask: a caller whose halves
// might not fit proves that first (the lowering's range check).
// Spelled High/Low, not Hi/Lo: Apple's Carbon headers, which Accelerate
// pulls into the CPU backend, define HiWord and LoWord as 16-bit macros.
inline constexpr uint64_t HighWord(uint64_t word) { return word >> 32; }
inline constexpr uint64_t LowWord(uint64_t word) { return word & 0xFFFFFFFFu; }
inline constexpr uint32_t HighWord32(uint64_t word) {
  return static_cast<uint32_t>(word >> 32);
}
inline constexpr uint32_t LowWord32(uint64_t word) {
  return static_cast<uint32_t>(word & 0xFFFFFFFFu);
}
inline constexpr uint64_t PackWords(uint64_t hi, uint64_t lo) {
  return (hi << 32) | lo;
}

// A tensor reference is a 64-bit word: bit 63 selects the address space
// (0 = mutable arena, 1 = read-only rodata), bits 0..61 are a byte offset.
// Bit 62 (plan v17, E12 #95) selects a third, read-only space: the SOURCE
// MODEL FILE the plan was compiled from, bound by the engine at run time
// (BindSourceModel). A source ref's offset is an absolute SMF data offset —
// the very offset the emit table patches — so an eval program lowered
// against it reads the f32 weights that ship, not the plan's int8 / bf16
// copies of them. Source refs are admitted in the eval program only.
inline constexpr uint64_t kRodataBit = 1ULL << 63;
inline constexpr uint64_t kSourceBit = 1ULL << 62;
inline constexpr uint64_t kNullRef = ~0ULL;

inline constexpr uint64_t MakeArenaRef(uint64_t offset) { return offset; }
inline constexpr uint64_t MakeRodataRef(uint64_t offset) {
  return offset | kRodataBit;
}
inline constexpr uint64_t MakeSourceRef(uint64_t offset) {
  return offset | kSourceBit;
}
inline constexpr bool IsRodataRef(uint64_t ref) {
  return (ref & kRodataBit) != 0;
}
inline constexpr bool IsSourceRef(uint64_t ref) {
  return (ref & (kRodataBit | kSourceBit)) == kSourceBit;
}
inline constexpr uint64_t RefOffset(uint64_t ref) {
  return ref & ~(kRodataBit | kSourceBit);
}

// The ISA table: an opcode to a line, its operands in the trailing comment.
// clang-format off
enum class OpCode : uint16_t {
  kNop = 0,
  // GEMM family. C[M,N]; refs in in[0..2], dims in out[0..2] = M, N, K.
  // v19 (F7 #138): the three f32 forms may carry a scalar alpha as f32
  // bits in `imm` (0 = 1.0, what every earlier plan meant): C = alpha *
  // A @ B [+ D]. See "The GEMM alpha" below for where each backend
  // applies it.
  kGemmNN = 1,     // C = A[M,K] @ B[K,N]
  kGemmNT = 2,     // C = A[M,K] @ B[N,K]^T   (VJP: dX = dC @ W^T)
  kGemmTN = 3,     // C = A[K,M]^T @ B[K,N]   (VJP: dW = X^T @ dC)
  kGemmAccNN = 4,  // C += alpha * A @ B; alpha f32 bits in in[3] (LoRA merge)
  // Elementwise / broadcast.
  kAddEW = 5,      // out = x + y            in: x, y, out; out[0] = count
  kAddBias = 6,    // out[N,M] = x + b[M]    in: x, b, out; out[0..1] = N, M
  kReluFwd = 7,    // out = max(x, 0)        in: x, out;    out[0] = count
  kReluBwd = 8,    // dx = x <= 0 ? 0 : dy   in: dy, x, dx; out[0] = count
  kScale = 9,      // out = alpha * x        in: x, out, alpha_bits; out[0] = count
  kReduceRows = 10,  // db[M] = sum_n dY[N,M]  in: dy, db;  out[0..1] = N, M
  // Losses (fwd caches what the bwd kernel needs; scalar loss slot).
  kSoftmaxXEntFwd = 11,  // in: logits, labels(i32), loss, probs; out[0..1]=N,C
  kSoftmaxXEntBwd = 12,  // in: probs, labels, seed, dlogits;     out[0..1]=N,C
  kMseFwd = 13,          // in: pred, target, loss;               out[0]=count
  kMseBwd = 14,          // in: pred, target, seed, dpred;        out[0]=count
  kKLDistillFwd = 15,  // in: s_logits, t_logits, loss, p_s; out[0]=p_t ref,
                       // out[1]=N<<32|C, out[2]=loss_scale bits<<32 |
                       // temperature f32 bits (v8: scale = T^2; a zero high
                       // word — every pre-v8 plan — means scale 1.0)
  kKLDistillBwd = 16,  // in: p_s, p_t, seed, dlogits; out[0]=N<<32|C,
                       // out[1]=loss_scale bits<<32 | temperature f32 bits
                       // (T: finite, within [kKlTemperatureMin,
                       // kKlTemperatureMax] — CR1 #142)
  // Optimizers (in-place; hyperparameters live in the PlanHeader).
  // v12: out[1] may carry a per-tensor clip threshold as f32 bits (0 = no
  // clip, every pre-v12 plan): the step then consumes g * min(1,
  // max/||g||_2) without ever writing g — kClipNorm folded into the step,
  // one full read+write pass over the gradient gone. A non-finite norm
  // aborts the step before p, m or v is touched.
  kSgdStep = 17,    // in: p, g;             out[0] = count, out[1] = clip bits
  kAdamWStep = 18,  // in: p, g, m, v;       out[0] = count, out[1] = clip bits
  // Utility.
  kFill = 19,  // in: dst, f32 value bits;   out[0] = count
  kCopy = 20,  // in: src, dst;              out[0] = count (floats)
  // Elementwise (v2).
  kMulEW = 21,    // out = x * y             in: x, y, out; out[0] = count
  kGeluFwd = 22,  // out = gelu(x)           in: x, out;    out[0] = count
  kGeluBwd = 23,  // dx = dy * gelu'(x)      in: dy, x, dx; out[0] = count
  kSiluFwd = 24,  // out = x * sigmoid(x)    in: x, out;    out[0] = count
  kSiluBwd = 25,  // dx = dy * silu'(x)      in: dy, x, dx; out[0] = count
  // LayerNorm over the last dim of x[N,D] with affine gamma/beta[D] (v2).
  kLayerNormFwd = 26,  // in: x, gamma, beta, y; out[0]=N<<32|D,
                       // out[1]=mean ref [N], out[2]=rstd ref [N]
  kLayerNormBwd = 27,  // in: dy, x, gamma, dx;  out[0]=mean ref,
                       // out[1]=rstd ref, out[2]=N<<32|D
  // Training utilities (v2).
  kClipNorm = 28,  // g *= min(1, max/||g||)  in: g, max f32 bits; out[0]=count
                   // (max: a finite positive float — CR1 #142; the kernel
                   // refuses a non-finite norm and leaves g as it was)
  // Quantized frozen weights (v2): B is per-tensor symmetric int8 in rodata,
  // dequantized on the fly as scale * q. Layouts mirror the f32 GEMMs.
  kGemmNNQ8 = 29,  // C = A @ dq(B);    in: A, Bq8, C, scale bits; out=M,N,K
  kGemmNTQ8 = 30,  // C = A @ dq(B)^T;  in: A, Bq8, C, scale bits; out=M,N,K
                   // (scale: a finite positive float, or — kFlagQ8ColScale —
                   // the rodata ref of the per-column scales, each finite
                   // and positive; CR1 #142)
  // --- Transformer family (plan v6). Sequence geometry rides packed dim
  // words: out words carry B<<32|S (sequences x positions) and H<<32|d
  // (heads x head width); activations are rank-2 [B*S, H*d] row-major with
  // heads interleaved along the row. The attention probability matrix
  // P[B,H,S,S] is flattened [B*H*S, S] and cached for the backward
  // primitives, exactly as LayerNorm caches mean/rstd.
  // RMSNorm over the last dim of x[N,D]: y = x * rstd(x) * gamma.
  kRmsNormFwd = 31,  // in: x, gamma, y, rstd[N]; out[0]=N<<32|D
  kRmsNormBwd = 32,  // in: dy, x, gamma, dx; out[0]=rstd ref, out[1]=N<<32|D
  // Rotary position embedding on interleaved pairs within each head;
  // requires d even. Backward is the transpose rotation (angle negated).
  // v12: in[2] may name a kRopeTable result for the same (S, d, base);
  // kNullRef (every pre-v12 plan) computes the angles in place.
  kRopeFwd = 33,  // in: x, y[, table];   out[0]=B<<32|S, out[1]=H<<32|d, out[2]=base bits
  kRopeBwd = 34,  // in: dy, dx[, table]; out[0]=B<<32|S, out[1]=H<<32|d, out[2]=base bits
  // Causal scaled-dot-product attention. Forward computes, per (b, h):
  //   P = softmax_rows(mask(Q K^T / sqrt(d))), O = P V
  // caching P for the backward primitives.
  kAttnFwd = 35,  // in: q, k, v, o; out[0]=P ref, out[1]=B<<32|S, out[2]=H<<32|d
  kAttnDP = 36,   // dP = dO V^T:  in: dO, v, dP;    out[0..1]=B|S, H|d
  kAttnDV = 37,   // dV = P^T dO:  in: P, dO, dV;    out[0..1]=B|S, H|d
  // Row-softmax backward: dS = P * (dP - rowsum(dP * P)), any [rows, cols].
  kSoftmaxRowsBwd = 38,  // in: P, dP, dS; out[0]=rows<<32|cols
  kAttnDQ = 39,   // dQ = (dS K)/sqrt(d):   in: dS, k, dQ; out[0..1]=B|S, H|d
  kAttnDK = 40,   // dK = (dS^T Q)/sqrt(d): in: dS, q, dK; out[0..1]=B|S, H|d
  // --- Token-native input (plan v7). ----------------------------------------
  // Frozen embedding gather: out[t, :] = table[tokens[t], :]. Tokens are
  // i32; the table is frozen rodata (never adapted — LoRA targets MatMuls).
  // The validator proves the extents; the RUNTIME bound tokens[t] < V is
  // the feeder contract's job, exactly as class labels are bounded by the
  // narrowest softmax width before anything executes.
  kEmbedFwd = 41,  // in: tokens(i32 [T]), table(ro [V,D]), out([T,D]);
                   // out[0]=T, out[1]=V<<32|D
  // --- Gradient accumulation (plan v9). -------------------------------------
  // dst += src, in place: the grad program folds each micro-batch's
  // gradient into its persistent accumulator; the step program then clips
  // and steps on the accumulator and zeroes it (kFill). One operand is
  // read and written through one pointer — not an alias.
  kAccumulate = 42,  // in: dst, src; out[0] = count
  // --- bf16 frozen weights (plan v10, roadmap 2c). ---------------------------
  // B is bfloat16 in rodata (2 bytes per element, exact widening to f32 on
  // the way into the tile); compute stays f32. Layouts mirror the f32
  // GEMMs; kGemmNNBF16 takes the same fused epilogue as kGemmNN (its in[3]
  // is free for the bias ref), kGemmNTBF16 none.
  kGemmNNBF16 = 43,  // C = A @ widen(B);    in: A, Bbf16, C[, bias]; out=M,N,K
  kGemmNTBF16 = 44,  // C = A @ widen(B)^T;  in: A, Bbf16, C;         out=M,N,K
  // --- RoPE angle table (plan v12, E3). --------------------------------------
  // table[s, c, 0..1] = cos, sin of angle(s, c) = s * base^(-2c/d), built
  // with the very fp32 recurrence kRopeFwd/kRopeBwd run (freq *= step), so
  // a rotation through the table is bit-identical to one that recomputes.
  // The angles depend on (s, c) alone; without the table every (b, h) unit
  // of every RoPE instruction recomputes them. Built on the device, never
  // by the compiler: its sin/cos are the device libm's, as before.
  kRopeTable = 45,  // in: table([S, d/2, 2]); out[0]=S<<32|d, out[1]=base bits
  // --- Fused elementwise chain (plan v13, E4 / roadmap Phase 1b). -------------
  // One pass over `count` elements running a micro-program of 1..4 stages
  // (FusedStage, below) on a running value that starts as in[0]:
  //   in[0] = x, in[1..2] = the chain's other tensor operands (kNullRef
  //   when unused), in[3] = out;  out[0] = count,  out[1] = the stages, one
  //   byte each, first stage in the low byte, zero-terminated,  out[2] =
  //   two f32 immediates (imm0 bits | imm1 bits << 32) for scale stages.
  // Each stage evaluates exactly the expression of the standalone opcode it
  // replaces, as its own loop over a cache-resident block, so a fused chain
  // is bit-identical to the instruction sequence it stands for; what it
  // removes is one arena-sized write and read per folded instruction.
  kFusedMap = 46,
  // --- Tiled attention (plan v15, E11 / #94). --------------------------------
  // The cached family above keeps P [B*H*S, S] from forward to backward and
  // materializes dP and dS at that size; the tiled family keeps four floats
  // per query row (kAttnStatsWidth: score max, 1/denominator, the softmax-
  // backward rowsum delta, one pad) and recomputes every probability a
  // backward pass needs — with the cached kernels' expressions, so the two
  // families compute identical bits and the compiler picks by memory alone.
  //   kAttnFwdTiled: in = q, k, v, o; out[0] = stats (written), out[1] =
  //     B<<32|S, out[2] = H<<32|d — the family's geometry words.
  //   kAttnDQTiled / kAttnDKTiled / kAttnDVTiled: in = q, k, v, dO;
  //     out[0] = stats (dQ also writes delta into it; dK reads delta, so a
  //     stream runs dQ before dK), out[1] = the result, out[2] = the
  //     geometry packed 16 bits each: B<<48 | S<<32 | H<<16 | d.
  kAttnFwdTiled = 47,
  kAttnDQTiled = 48,
  kAttnDKTiled = 49,
  kAttnDVTiled = 50,
  // The masked cross-entropy (S3 #149, plan v22 kDataFlagMaskedLoss): the
  // operands of 11 / 12, with a label of -1 (kSdsIgnoreLabel) marking a
  // position the loss ignores. The mean runs over the target rows alone
  // (0 for a batch with none), the probabilities are written for every
  // row, and an ignored row's gradient is zero. A plan carries these
  // exactly when its header carries the flag; a plain plan never meets a
  // -1 label (the feeder contract refuses a masked corpus for it).
  kSoftmaxXEntMaskedFwd = 51,  // in: logits, labels(i32), loss, probs; out[0..1]=N,C
  kSoftmaxXEntMaskedBwd = 52,  // in: probs, labels, seed, dlogits;     out[0..1]=N,C
  // The global-norm clip (#190, plan v24): every trainable gradient scaled
  // by the one factor min(1, max / sqrt(sum over tensors of ||g_t||^2)).
  // kSqNormAccum adds one gradient's squared norm into a one-float arena
  // slot (zeroed after the steps, as the gradient accumulators are);
  // kClipGlobal scales a gradient in place by the factor that slot implies,
  // refusing a non-finite norm as kClipNorm does. All the accumulations
  // precede the first scaling, so every tensor sees the same factor.
  kSqNormAccum = 53,  // acc += ||g||^2   in: g, acc[1];          out[0] = count
  kClipGlobal = 54,   // g *= min(1, max/sqrt(acc))  in: g, acc[1], max bits; out[0] = count
  // The attention segment mask (#191, plan v25): packed windows hold
  // several documents joined by one id, and a position must not attend to
  // another document. Segments are contiguous, so the mask is one start
  // per row: kSegmentStarts writes, for every token position t of window b,
  // the first position of t's document (one past the last out[1] id
  // strictly before t; 0 when none — the compiler passes the document
  // separator, the EOS or <|endoftext|> when the EOS ends a chat turn,
  // #218), and kAttnFwdSeg is kAttnFwd with keys j < that
  // start masked as the causal ones are — exact zeros in the P cache, so
  // the cached backward needs no operand of its own.
  kSegmentStarts = 55,  // in: tokens(i32 [B*S]), starts(i32 [B*S]); out[0]=B<<32|S, out[1]=eos id
  kAttnFwdSeg = 56,     // in: q, k, v, o; out[0]=P ref, out[1]=PackAttnGeometry, out[2]=starts ref
  // The mask on the tiled family (#191 step 2, plan v26). The tiled
  // backward opcodes have no free slot for a starts operand and each needs
  // every query row's start (dK and dV walk the query rows of a key), so
  // the forward carries the starts the way kAttnFwdSeg does and leaves each
  // row's CLAMPED start in the fourth float of its stats row
  // (kAttnStatsStart) — a pad until v26. Opcodes 48-50 are unchanged in
  // encoding and read that float: a key before the row's start is skipped.
  // kAttnFwdTiled writes 0 there, so a plan without the mask computes the
  // bits it always did. Over finite operands the masked tiled family
  // computes the masked cached family's bits: a skipped term is the exact
  // +/-0 the cached kernels add, and an accumulator that starts at +0 never
  // holds -0.
  kAttnFwdTiledSeg = 57,  // in: q, k, v, o; out[0]=stats, out[1]=PackAttnGeometry, out[2]=starts ref
  // 8-bit AdamW moments (S5 #192, plan v27; opt-in, recipe field moments
  // int8). kAdamWStep's step over moments stored block-wise in 8 bits: m
  // signed int8 [count], v unsigned uint8 [count], and one f32 scales
  // vector of 2 * blocks floats — every block's m absmax, then every
  // block's v max (kMomentBlock values a block, the last one short). Per
  // block the step dequantizes, runs kAdamWStep's f32 expression on the
  // dequantized moments (the parameter moves by the unquantized m and v of
  // this step), then requantizes without bias: between the two decodable
  // values around a new moment the code is the upper one with the
  // probability of the fraction of the way there, drawn from a stateless
  // integer hash of the optimizer step, the element and in[2] (round to
  // nearest froze every moment that moves less than half a level a step —
  // the #192 review). The codes, the dither and the dequantization are
  // MomentCodeM / MomentCodeV, MomentDither and MomentValueM / MomentValueV
  // below, the same rule on every backend and in the oracle, deterministic
  // given the step and the element (the coded state may still differ by a
  // level across backends: the f32 blend before the code is contracted
  // differently). A
  // non-finite moment (or a stored scale that is not a finite
  // non-negative float) refuses the step before anything is written, as a
  // non-finite clip norm does. out[1] is the fused clip word, as on 18.
  // Arithmetic the f32 opcode does not compute: the plan's ledger records
  // it as a quantized pass, and the compiler emits it only when asked.
  kAdamW8Step = 58,  // in: p, g, m8, v8; out[0]=count, out[1]=clip bits, out[2]=scales ref
};
// clang-format on

// --- 8-bit moments (plan v27, S5 #192). ----------------------------------------
// A block is kMomentBlock consecutive values of one tensor; each carries
// two f32 scales (the m block's absmax, the v block's max). 256, as in
// bitsandbytes' 8-bit Adam: two scales per 256 values add 8 / 256 = 0.03 B
// to the 2 B a value the codes take (8 B in f32: 3.94x smaller), and a
// block is eight SIMD lanes of 32 on the GPU, one simdgroup.
inline constexpr uint64_t kMomentBlock = 256;
/// Blocks of a tensor of `count` values (the last one may be short).
inline constexpr uint64_t MomentBlocks(uint64_t count) {
  return count / kMomentBlock + (count % kMomentBlock != 0 ? 1 : 0);
}
/// A block whose absmax (m) or max (v) is below this is stored as zeros,
/// scale 0: it keeps every dequantized value and every product the codes
/// below form a normal float (2^-80 * 2^-15.75 for m, 2^-80 * 2^-31.75 for
/// v), so no backend's subnormal handling enters, and sqrt(2^-80) is 1e-12
/// — far below the eps every AdamW denominator adds.
inline constexpr float kMomentScaleFloor = 0x1p-80f;
// Both codes are logarithmic: eight levels an octave below the block's
// scale S (m's absmax, v's max). Level n is the magnitude
//   S * (kMomentLevel[n % 8] * 2^-(n / 8)),
// level 0 the scale itself. The tables are f32 bit patterns — 2^(-r/8) and
// 2^(1 - (j + 1/2) / 8), each rounded to nearest once — so the CPU, the
// GPU and the oracle compare and multiply the same floats.
//   m: int8, sign and magnitude. Code 0 is exactly 0; code +/-q, q in
//      [1, 127], is level n = 127 - q in [0, 126]: 15.75 octaves (4.7
//      decades) under the absmax. (A linear m code — 127 equal steps of
//      absmax / 127 — leaves an element whose m is a thousandth of its
//      block's largest between code 0 and code 1, where the dither below
//      is noise of five times its update; the probe in
//      docs/ledger/moments-int8-2026-11-23/.) -128 is never written.
//   v: uint8. Code 0 is exactly 0; code q in [1, 255] is level
//      n = 255 - q in [0, 254]: 31.75 octaves (9.6 decades) under the max.
// The rounding is unbiased, not to nearest (#192 review). A moment moves
// by less than a level a step — beta2 moves v by 0.1 %, a level is 9 % —
// so a code rounded to its nearest level returns to where it was and
// never moves again: v could not decay, and m froze at whatever code a
// burst left it. Between the two decodable values lo <= x <= hi that
// bracket a magnitude x — the dequantizer's own floats, MomentLevelValue
// of two neighbouring levels, or 0 and the lowest level for m — the code
// is hi's when
//   MomentDither(...) < uint32(((x - lo) / (hi - lo)) * 2^24),
// f32 arithmetic, the conversion truncating: hi with probability
// (x - lo) / (hi - lo) to within 2^-24, so the dequantized value's
// expectation is x, in the decoded domain. An x that is a level's value
// keeps that level whatever the dither; an x at or above the scale is
// level 0. One deliberate bias: a nonzero v below v's lowest level takes
// the lowest level, never 0 — no element steps by m / eps for want of a v
// its m still carries (the failure a linear v code shows; the ledger).
// The dither is a stateless integer hash of the optimizer step, the
// element's index in its tensor and the m codes' operand (the plan's own
// word, the same on every backend and run): no generator state, so a
// resumed run, any worker count and any chunking of the work draw the
// same bits. With s the step and o the operand folded to 32 bits
// (low word xor high word), c the stream's constant (m and v draw
// independently) and i the index mod 2^32, all sums mod 2^32:
//   key = MomentMix(MomentMix(o ^ c) + s),
//   MomentDither = MomentMix(key + i) >> 8                (24 bits).
// Exponent-field bit operations, compares, one multiply a level value,
// two subtractions and one division, all single IEEE operations on named
// floats, and 32-bit integer arithmetic: no transcendental function.
inline constexpr uint32_t kMomentLevelBits[8] = {
    0x3f800000u, 0x3f6ac0c7u, 0x3f5744fdu, 0x3f45672au,
    0x3f3504f3u, 0x3f25fed7u, 0x3f1837f0u, 0x3f0b95c2u};
inline constexpr uint32_t kMomentThresholdBits[8] = {
    0x3ff5257du, 0x3fe0ccdfu, 0x3fce248cu, 0x3fbd08a4u,
    0x3fad583fu, 0x3f9ef532u, 0x3f91c3d3u, 0x3f85aac3u};
inline constexpr uint32_t kMomentMMaxLevel = 126;
inline constexpr uint32_t kMomentVMaxLevel = 254;
inline constexpr uint32_t kMomentDitherM = 0x9E3779B9u;
inline constexpr uint32_t kMomentDitherV = 0x85EBCA6Bu;
inline constexpr uint32_t kMomentMixA = 0x7FEB352Du;
inline constexpr uint32_t kMomentMixB = 0x846CA68Bu;
inline constexpr uint32_t kMomentDitherBits = 24;

/// The dither's mixer: a 32-bit multiply-xorshift bijection.
inline constexpr uint32_t MomentMix(uint32_t x) {
  x ^= x >> 16;
  x *= kMomentMixA;
  x ^= x >> 15;
  x *= kMomentMixB;
  x ^= x >> 16;
  return x;
}
/// One stream's key for optimizer step `step` of the tensor whose m codes
/// are operand `operand` (kAdamW8Step's in[2]).
inline constexpr uint32_t MomentDitherKey(uint64_t step, uint64_t operand,
                                          uint32_t stream) {
  const uint32_t s = static_cast<uint32_t>(step ^ (step >> 32));
  const uint32_t o = static_cast<uint32_t>(operand ^ (operand >> 32));
  return MomentMix(MomentMix(o ^ stream) + s);
}
/// The 24-bit dither of element `index` under `key`.
inline constexpr uint32_t MomentDither(uint32_t key, uint64_t index) {
  return MomentMix(key + static_cast<uint32_t>(index)) >>
         (32 - kMomentDitherBits);
}

/// The magnitude of level `n` in a block whose scale is `scale`.
inline float MomentLevelValue(uint32_t n, float scale) {
  const float octave = std::bit_cast<float>((127u - n / 8) << 23);
  const float unit = std::bit_cast<float>(kMomentLevelBits[n % 8]) * octave;
  return scale * unit;
}
/// The level of magnitude `x` > 0 in a block whose scale is `scale` (>= the
/// floor): one of the two levels whose values bracket x, by the rule
/// above. `max_level` + 1 stands for the zero code, returned only under
/// `to_zero` (m); without it a magnitude below the lowest level keeps the
/// lowest level (v).
inline uint32_t MomentLevel(float x, float scale, uint32_t max_level,
                            bool to_zero, uint32_t dither) {
  // The nearest level in the log domain: the ratio x / scale written
  // y * 2^E, y in [1, 2), is level -8 (E + 1) + #{j : y < threshold[j]}.
  const uint32_t bits = std::bit_cast<uint32_t>(x / scale);
  const uint32_t biased = (bits >> 23) & 0xFFu;
  uint32_t n = max_level;
  if (biased != 0) {  // a subnormal ratio is far below the lowest level
    const float y = std::bit_cast<float>((bits & 0x7FFFFFu) | 0x3F800000u);
    uint32_t below = 0;
    for (uint32_t t : kMomentThresholdBits)
      below += y < std::bit_cast<float>(t) ? 1u : 0u;
    const int64_t level = 8 * (126 - static_cast<int64_t>(biased)) + below;
    if (level < 0)
      n = 0;  // a ratio above 1: no block scale is below its values
    else if (level < static_cast<int64_t>(n))
      n = static_cast<uint32_t>(level);
  }
  // The bracket, in the dequantizer's floats: x's nearest level and the
  // neighbour on x's side of it.
  const float at = MomentLevelValue(n, scale);
  float lo = at, hi = at;
  uint32_t lo_n = n, hi_n = n;
  if (x > at) {
    if (n == 0) return 0;
    hi_n = n - 1;
    hi = MomentLevelValue(hi_n, scale);
  } else if (x < at) {
    if (n == max_level && !to_zero) return n;
    lo_n = n + 1;  // max_level + 1: the zero code
    lo = n == max_level ? 0.0f : MomentLevelValue(lo_n, scale);
  } else {
    return n;
  }
  if (!(x > lo)) return lo_n;
  if (!(x < hi)) return hi_n;
  const float above = x - lo;
  const float width = hi - lo;
  const float fraction = above / width;
  const float scaled = fraction * 0x1p24f;
  const uint32_t threshold = static_cast<uint32_t>(scaled);
  return dither < threshold ? hi_n : lo_n;
}

/// m's code for `m` in a block whose absmax is `amax` (>= the floor).
inline int8_t MomentCodeM(float m, float amax, uint32_t dither) {
  if (m == 0.0f) return 0;
  const uint32_t n =
      MomentLevel(m < 0.0f ? -m : m, amax, kMomentMMaxLevel, true, dither);
  const int q = static_cast<int>(kMomentMMaxLevel + 1) - static_cast<int>(n);
  return static_cast<int8_t>(m < 0.0f ? -q : q);
}
/// The dequantized m of code `q` in a block whose absmax is `amax`.
inline float MomentValueM(int8_t q, float amax) {
  if (q == 0) return 0.0f;
  const int magnitude = q < 0 ? -static_cast<int>(q) : static_cast<int>(q);
  // -128 is never written; it reads as -127 does.
  const uint32_t n =
      magnitude > 127 ? 0u : 127u - static_cast<uint32_t>(magnitude);
  const float value = MomentLevelValue(n, amax);
  return q < 0 ? -value : value;
}
/// v's code for `v` >= 0 in a block whose max is `vmax` (>= the floor).
inline uint8_t MomentCodeV(float v, float vmax, uint32_t dither) {
  if (!(v > 0.0f)) return 0;
  const uint32_t n = MomentLevel(v, vmax, kMomentVMaxLevel, false, dither);
  return static_cast<uint8_t>(255u - n);
}
/// The dequantized v of code `q` in a block whose max is `vmax`.
inline float MomentValueV(uint8_t q, float vmax) {
  if (q == 0) return 0.0f;
  return MomentLevelValue(255u - q, vmax);
}

inline constexpr uint64_t kAttnStatsWidth = 4;
/// The stats row's fields: the score max, 1 / the softmax denominator, the
/// softmax-backward rowsum (the dQ pass's), and the row's document start as
/// a float (v26; 0 when the forward carried no mask).
inline constexpr uint64_t kAttnStatsStart = 3;
/// The start a tiled backward kernel reads from a stats row, for query row
/// `i`: clamped into [0, i] in the float domain before the conversion, NaN
/// to 0, so no value an arena holds becomes a loop bound or an index
/// unchecked (the rule kAttnFwdSeg's kernel holds for its i32 starts).
inline uint64_t AttnStatsStart(float stored, uint64_t i) {
  if (!(stored > 0.0f)) return 0;  // NaN, negatives and zero
  const float cap = static_cast<float>(i);
  return stored >= cap ? i : static_cast<uint64_t>(stored);
}
/// The tiled backward's one-word geometry (each field must fit 16 bits).
inline constexpr uint64_t PackAttnGeometry(uint64_t B, uint64_t S, uint64_t H,
                                           uint64_t d) {
  return (B << 48) | (S << 32) | (H << 16) | d;
}
struct AttnGeometry {
  uint64_t B, S, H, d;
};
inline constexpr AttnGeometry UnpackAttnGeometry(uint64_t word) {
  return {word >> 48, (word >> 32) & 0xFFFFu, (word >> 16) & 0xFFFFu,
          word & 0xFFFFu};
}

// --- kFusedMap stages ----------------------------------------------------------
// A stage byte: the kind in the low nibble, an argument in the high nibble.
//   binary kinds (add, mul): bits 4..5 = which operand slot (1 or 2) holds
//     the other tensor; bit 7 set = the running value is the RIGHT operand
//     (x_other OP run), clear = the left (run OP x_other). IEEE add and mul
//     commute in value, but not in which NaN payload survives; the order of
//     the instruction being replaced is kept.
//   scale: bits 4..5 = which immediate (0 or 1) is alpha.
//   unary kinds: the high nibble is zero.
enum class FusedStage : uint8_t {
  kEnd = 0,
  kAdd = 1,    // run + x        (kAddEW)
  kMul = 2,    // run * x        (kMulEW)
  kScale = 3,  // alpha * run    (kScale)
  kRelu = 4,   // kReluFwd
  kGelu = 5,   // kGeluFwd
  kSilu = 6,   // kSiluFwd
};
inline constexpr uint8_t kFusedStageKindMask = 0x0F;
inline constexpr uint8_t kFusedStageArgShift = 4;
inline constexpr uint8_t kFusedStageArgMask = 0x03;
inline constexpr uint8_t kFusedStageRunIsRight = 0x80;
inline constexpr size_t kFusedMapMaxStages = 4;

inline constexpr uint8_t MakeFusedStage(FusedStage kind, uint8_t arg = 0,
                                        bool run_is_right = false) {
  return static_cast<uint8_t>(
      static_cast<uint8_t>(kind) |
      static_cast<uint8_t>((arg & kFusedStageArgMask) << kFusedStageArgShift) |
      (run_is_right ? kFusedStageRunIsRight : uint8_t{0}));
}
inline constexpr FusedStage FusedStageKind(uint8_t stage) {
  return static_cast<FusedStage>(stage & kFusedStageKindMask);
}
inline constexpr uint8_t FusedStageArg(uint8_t stage) {
  return (stage >> kFusedStageArgShift) & kFusedStageArgMask;
}

// --- Instruction flags (plan v5): fused GEMM epilogues. ----------------------
// Every plan before v5 carries flags == 0 on every instruction; from v5 on
// the validator rejects any bit it does not know, so a future flag can never
// be silently skipped. The epilogue applies to the C write-back of the
// forward GEMMs: C = act(A@B + bias). Per-element expression order is
// identical to the unfused kGemmNN + kAddBias + k<Act>Fwd sequence, so
// fusion changes memory traffic, never bits.
//
// Validity: kGemmNN and kGemmNNBF16 take bias and/or activation (a fused
// bias ref rides the otherwise-free in[3]); kGemmNNQ8 takes activation only
// — its in[3] already carries the dequant scale, so a bias has nowhere to
// ride. Every other opcode requires flags == 0.
inline constexpr uint16_t kFlagEpilogueBias = 1u << 0;  // in[3] = bias ref [N]
inline constexpr uint16_t kFlagEpilogueActShift = 1;  // bits 1..2: EpilogueAct
inline constexpr uint16_t kFlagEpilogueActMask = 3u << kFlagEpilogueActShift;
inline constexpr uint16_t kEpilogueFlagsMask =
    kFlagEpilogueBias | kFlagEpilogueActMask;
// --- The GEMM addend (plan v14, E10). -----------------------------------------
// The f32 GEMMs (kGemmNN, kGemmNT, kGemmTN): C = D + A @ B, with D's ref
// ([M, N] f32, read) riding the otherwise-free in[3]. Every LoRA site sums
// a rank-r GEMM into an activation-sized tensor twice per step — forward
// C' = C + (s*t)@B, backward dX = dC@W^T + dt@A^T — and each used to be a
// GEMM into a transient followed by a kAddEW over the whole tensor. The
// addend form writes the sum as the dot products come out of the core.
// Each element is `d + s` over the complete dot product s, exactly what
// the two-instruction form computed, so the fold changes memory traffic and
// never bits. It excludes the bias / activation epilogue (kGemmNN's in[3]
// holds one ref), and the narrow-weight GEMMs do not take it.
// in[3] = addend ref [M,N]
inline constexpr uint16_t kFlagGemmAddend = 1u << 3;
// --- Per-column int8 scales (plan v17, E12 #95). ------------------------------
// kGemmNNQ8 / kGemmNTQ8 only: in[3] is a rodata ref to one f32 scale per
// OUTPUT COLUMN of the quantized weight W [K, M] (float[M]) instead of the
// per-tensor scale's bits. A per-tensor max-abs scale lets one outlier
// column (LLM projections carry |w| outliers 10-50x the bulk) collapse
// every other column to a few int8 levels; a column's own scale keeps its
// resolution. In the forward (NN) the scale is the output column's —
// C[m, n] = act(s[n] * sum_k A[m, k] * q[k, n]); in the dX GEMM (NT, W read
// as [N = K, M]) it is the REDUCTION index's, applied as each int8 element
// widens — C[m, n] = sum_k A[m, k] * (q[n, k] * s[k]).
inline constexpr uint16_t kFlagQ8ColScale = 1u << 4;  // in[3] = scales ref
// --- Immediate floats (CR1 #142). ---------------------------------------------
// Every float an instruction carries as bits is proven finite and within a
// stated range by the validator before any kernel multiplies by it: the
// standalone clip threshold (kClipNorm's in[1]) and the fused one (the
// optimizer steps' out[1]; 0 = absent) finite and positive; the per-tensor
// int8 dequant scale (the q8 GEMMs' in[3] without kFlagQ8ColScale) finite
// and positive, and each per-column scale likewise (read from rodata by
// the load contract); the distillation temperature (the KL words' low
// half) finite and within [kKlTemperatureMin, kKlTemperatureMax] — a
// non-positive T inverts or dissolves both softmaxes, and the bounds keep
// logits / T representable at either end. The compiler's config check and
// the CLI apply the same bounds, so the three agree by construction.
inline constexpr float kKlTemperatureMin = 1e-3f;
inline constexpr float kKlTemperatureMax = 1e3f;
inline constexpr const char* kKlTemperatureRangeText = "[1e-3, 1e3]";
static_assert(kKlTemperatureMin == 1e-3f && kKlTemperatureMax == 1e3f,
              "kKlTemperatureRangeText spells the bounds");
// --- Relaxed arithmetic (plan v18, F2 #130 / F4 #132). ------------------------
// The six frozen-weight GEMMs (kGemmNN / kGemmNT and their int8 and bf16
// forms) may carry this bit: the backend MAY compute the product with a
// certified relaxed kernel. On Metal that is the tensor-op family, which
// rounds the f32 activations to bf16 once at the GEMM input and multiplies
// them against the exact stored weight (bf16 or int8; an f32 weight runs in
// the GPU's reduced-precision mode) into f32 accumulators; on the CPU it is
// a vendor BLAS (Accelerate) for f32 weights. The exact portable kernel is
// always a valid execution of a relaxed instruction — the bit is a
// permission, not a demand. The old path never emits it: a compiler sets it
// only under --precision certified-bf16, only in the train and step
// programs (the eval program scores what ships in exact arithmetic), and
// tool/pack_update.py refuses to package such a plan unless a numerics
// certificate (tool/certify_numerics.py) vouches for it — the certificate
// binds to the plan's hash, so it cannot live inside the plan it certifies.
// The on-device validator gates the bit on the version alone; the LOADER
// gates the plan (F2 #206): a plan carrying the bit loads only in the binary
// built beside its granted certificate, which holds that plan's seal
// (RelaxedPlanPermit, runtime/loader/contract.h). Relaxed
// kernels stay run-to-run bitwise on a device (fixed reduction order, no
// atomics); their bits differ from the reference's, so a relaxed plan
// compares to the exact one at the certificate's tolerance.
// Plan v20 (F3 #131) admits the same bit on the four tiled attention
// opcodes (kAttnFwdTiled .. kAttnDVTiled): on Metal, at head widths <= 64,
// the blocked kernels' passes with every block's products on
// simdgroup_matrix tiles (the f32 sums reordered, and the forward scales
// its accumulated row by 1/l once rather than each probability; the stats
// row and the causal mask the exact family's). The CPU runs the exact
// family; the compiler sets the bit under certified precision.
inline constexpr uint16_t kFlagRelaxed = 1u << 5;
inline constexpr uint16_t kKnownFlagsMask =
    kEpilogueFlagsMask | kFlagGemmAddend | kFlagQ8ColScale | kFlagRelaxed;
// --- The GEMM alpha (plan v19, F7 #138). -------------------------------------
// kGemmNN / kGemmNT / kGemmTN may carry a scalar alpha's f32 bits in `imm`
// (0 = 1.0, so every earlier plan means what it always meant), making the
// instruction C = alpha * A @ B [+ D]. It replaces the kScale a LoRA site
// spent on its rank-r activation twice per step — the forward's (alpha/r)*t
// and the backward's (alpha/r)*(dC @ B^T) — a whole dispatch per site per
// pass for one scalar the GEMM already multiplies by. Where the CPU
// applies it is each core's own: the NN and TN cores fold alpha into A's
// elements as they enter the product (one rounding each — the multiply
// the forward's and dB's kScale over t performed), the NT core multiplies
// the finished dot product (the multiply the backward's kScale over
// dC @ B^T performed), so the CPU's v19 program computes the v18
// program's bits for every alpha. Metal multiplies the finished dot
// product in every form, as its GemmAccNN kernel does: identical when
// alpha is a power of two (the default alpha/r = 16/8) and equal to
// rounding otherwise. A set alpha must be finite and nonzero; it is
// corruption on any other opcode, below v19, and beside kFlagRelaxed (the
// relaxed family is the frozen-weight products; an adapter product is
// never relaxed).
inline float GemmAlphaOf(uint32_t imm) {
  return imm == 0 ? 1.0f : std::bit_cast<float>(imm);
}

enum class EpilogueAct : uint16_t {
  kNone = 0,
  kRelu = 1,
  kGelu = 2,
  kSilu = 3
};

inline constexpr EpilogueAct EpilogueActOf(uint16_t flags) {
  return static_cast<EpilogueAct>((flags & kFlagEpilogueActMask) >>
                                  kFlagEpilogueActShift);
}
inline constexpr uint16_t MakeEpilogueFlags(bool bias, EpilogueAct act) {
  return (bias ? kFlagEpilogueBias : uint16_t{0}) |
         static_cast<uint16_t>(static_cast<uint16_t>(act)
                               << kFlagEpilogueActShift);
}

/// The epsilon a normalization forward's imm word means (0 = 1e-5).
inline constexpr float kDefaultNormEps = 1e-5f;
inline float NormEpsOf(uint32_t imm) {
  return imm == 0 ? kDefaultNormEps : std::bit_cast<float>(imm);
}

#pragma pack(push, 1)

/// One 64-byte instruction: a single L1 cache line, mirroring the design of
/// the inference-side SerializedInstruction in backend/serializer/schema.h.
///
/// `imm` (plan v16, P7 #96) is an opcode-defined 32-bit immediate. It was a
/// pad word every compiler wrote as zero and no runtime read; from v16
/// kLayerNormFwd and kRmsNormFwd carry the normalization epsilon's f32 bits
/// there (0 = the classic 1e-5, so every earlier plan means what it always
/// meant), and from v19 kGemmNN / kGemmNT / kGemmTN carry a scalar alpha's
/// f32 bits (0 = 1.0). On any other opcode, and below the version that
/// defined it, a nonzero imm is corruption — the flags word's discipline.
struct UpdateInstruction {
  uint16_t opcode = 0;
  uint16_t flags = 0;
  uint32_t imm = 0;
  uint64_t in[4] = {kNullRef, kNullRef, kNullRef, kNullRef};
  uint64_t out[3] = {0, 0, 0};
};

#pragma pack(pop)

static_assert(sizeof(UpdateInstruction) == 64,
              "UpdateInstruction must be exactly one cache line.");

}  // namespace seeml::update

#endif  // SEEML_SOURCE_PLAN_INSTRUCTION_H_
