/// The seeml-plan-probe trace container — the one byte seam that exists only
/// between two tools (the probe writes it, tool/frontier_exec.py reads it).
/// It lives in a header so seeml-abi can publish it like every other format.
///
///   u32 magic "SEPT" | u32 version | u64 instruction count
///   per instruction: u32 index | u16 opcode | u16 extents | u32 relaxed
///     per written extent: u64 arena offset | u64 bytes | the bytes
///
/// Version 2 (review F42) added `relaxed`: how many relaxed-kernel
/// executions the backend reported for the instruction (1 when a
/// kFlagRelaxed GEMM ran on Accelerate's SGEMM or Metal's tensor ops, 0
/// when it ran on the exact kernel or carried no flag), so the diff report
/// can say whether the C++ arm measured the certified kernel at all.
#pragma once

#include <cstdint>

namespace seeml::tool {

inline constexpr uint32_t kProbeTraceMagic = 0x54504553u;  // "SEPT"
inline constexpr uint32_t kProbeTraceVersion = 2;

}  // namespace seeml::tool
