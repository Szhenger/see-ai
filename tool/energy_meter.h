// Energy per token on macOS for seeml-bench (F6 #134, design item 2): the
// SoC's energy counters over an interval, read through IOReport — the
// library behind powermetrics — by an unprivileged process, opened at run
// time (dlopen: no link-time dependency, and the bench builds everywhere).
//
// The counters are cumulative per SoC domain (CPU, GPU, ANE, DRAM, ...) in
// the "Energy Model" channel group; Start() and Stop() sample them and
// Stop() returns each domain's delta in joules. Two limits, stated in every
// report: the counters are whole-SoC (every process on the machine), and
// what an unprivileged reader sees is the OS's choice — on macOS 26 the
// GPU's counter moves, the CPU's reads zero without root. The meter reports
// the domains that moved; a domain that read zero is not reported, and a
// reading with no domain is `available == false` with the reason. Off
// Apple, or without the library or the group, it is unavailable and never
// fails the bench. The bench takes the step-time regression for energy
// too: (hi run - lo run) / extra steps is the energy per step, the
// lifecycle removed.
#ifndef SEEML_TOOL_ENERGY_METER_H_
#define SEEML_TOOL_ENERGY_METER_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#endif

namespace seeml::bench {

struct EnergyReading {
  bool available = false;
  std::string reason;                    // when unavailable
  std::map<std::string, double> joules;  // per domain that moved
};

class EnergyMeter {
 public:
  EnergyMeter() { Open(); }
  ~EnergyMeter() { Close(); }
  EnergyMeter(const EnergyMeter&) = delete;
  EnergyMeter& operator=(const EnergyMeter&) = delete;

  bool available() const { return lib_ != nullptr; }
  const std::string& reason() const { return reason_; }

  void Start() {
#if defined(__APPLE__)
    if (!lib_) return;
    if (start_) CFRelease(start_);
    start_ = create_samples_(sub_, channels_, nullptr);
#endif
  }

  EnergyReading Stop() {
    EnergyReading out;
    out.reason = reason_;
#if defined(__APPLE__)
    if (!lib_ || !start_) {
      if (out.reason.empty()) out.reason = "meter not started";
      return out;
    }
    CFDictionaryRef end = create_samples_(sub_, channels_, nullptr);
    CFDictionaryRef delta = create_delta_(start_, end, nullptr);
    CFRelease(start_);
    start_ = nullptr;
    CFArrayRef rows = delta ? static_cast<CFArrayRef>(CFDictionaryGetValue(
                                  delta, CFSTR("IOReportChannels")))
                            : nullptr;
    const CFIndex n = rows ? CFArrayGetCount(rows) : 0;
    for (CFIndex i = 0; i < n; ++i) {
      CFDictionaryRef row =
          static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(rows, i));
      const int64_t value = get_int_(row, 0);
      if (value <= 0) continue;
      const std::string unit = Str(get_unit_(row));
      double scale = 0.0;
      if (unit == "nJ") scale = 1e-9;
      else if (unit == "uJ" || unit == "\xC2\xB5J") scale = 1e-6;
      else if (unit == "mJ") scale = 1e-3;
      else if (unit == "J") scale = 1.0;
      else continue;
      out.joules[Str(get_name_(row))] += static_cast<double>(value) * scale;
    }
    if (delta) CFRelease(delta);
    if (end) CFRelease(end);
    out.available = !out.joules.empty();
    if (!out.available)
      out.reason = "no Energy Model counter moved for this process "
                   "(unprivileged readers see the GPU's alone on macOS 26; "
                   "the CPU's need root)";
#else
    if (out.reason.empty()) out.reason = "no IOReport on this platform";
#endif
    return out;
  }

  /// The joules of `domain` in `r`, or a negative number when absent.
  static double Domain(const EnergyReading& r, const std::string& domain) {
    auto it = r.joules.find(domain);
    return it == r.joules.end() ? -1.0 : it->second;
  }

 private:
#if defined(__APPLE__)
  using CopyChannelsInGroup = CFDictionaryRef (*)(CFStringRef, CFStringRef,
                                                  uint64_t, uint64_t, uint64_t);
  using CreateSubscription = void* (*)(void*, CFMutableDictionaryRef,
                                       CFMutableDictionaryRef*, uint64_t,
                                       CFTypeRef);
  using CreateSamples = CFDictionaryRef (*)(void*, CFMutableDictionaryRef,
                                            CFTypeRef);
  using CreateSamplesDelta = CFDictionaryRef (*)(CFDictionaryRef, CFDictionaryRef,
                                                 CFTypeRef);
  using ChannelGetString = CFStringRef (*)(CFDictionaryRef);
  using SimpleGetInteger = int64_t (*)(CFDictionaryRef, int);

  static std::string Str(CFStringRef s) {
    char buf[256];
    if (s && CFStringGetCString(s, buf, sizeof buf, kCFStringEncodingUTF8))
      return buf;
    return "?";
  }

  void Open() {
    for (const char* path :
         {"/usr/lib/libIOReport.dylib",
          "/System/Library/PrivateFrameworks/IOReport.framework/IOReport"}) {
      lib_ = dlopen(path, RTLD_NOW);
      if (lib_) break;
    }
    if (!lib_) {
      reason_ = "libIOReport not found";
      return;
    }
    auto copy_group = reinterpret_cast<CopyChannelsInGroup>(
        dlsym(lib_, "IOReportCopyChannelsInGroup"));
    auto create_sub = reinterpret_cast<CreateSubscription>(
        dlsym(lib_, "IOReportCreateSubscription"));
    create_samples_ = reinterpret_cast<CreateSamples>(
        dlsym(lib_, "IOReportCreateSamples"));
    create_delta_ = reinterpret_cast<CreateSamplesDelta>(
        dlsym(lib_, "IOReportCreateSamplesDelta"));
    get_name_ = reinterpret_cast<ChannelGetString>(
        dlsym(lib_, "IOReportChannelGetChannelName"));
    get_unit_ = reinterpret_cast<ChannelGetString>(
        dlsym(lib_, "IOReportChannelGetUnitLabel"));
    get_int_ = reinterpret_cast<SimpleGetInteger>(
        dlsym(lib_, "IOReportSimpleGetIntegerValue"));
    if (!copy_group || !create_sub || !create_samples_ || !create_delta_ ||
        !get_name_ || !get_unit_ || !get_int_) {
      reason_ = "libIOReport lacks a symbol this meter needs";
      Close();
      return;
    }
    CFDictionaryRef channels =
        copy_group(CFSTR("Energy Model"), nullptr, 0, 0, 0);
    if (!channels) {
      reason_ = "no 'Energy Model' channel group on this device";
      Close();
      return;
    }
    CFMutableDictionaryRef wanted =
        CFDictionaryCreateMutableCopy(nullptr, 0, channels);
    CFRelease(channels);
    sub_ = create_sub(nullptr, wanted, &channels_, 0, nullptr);
    if (!sub_ || !channels_) {
      reason_ = "IOReport refused the subscription";
      Close();
    }
  }

  void Close() {
    if (start_) CFRelease(start_);
    start_ = nullptr;
    if (lib_) dlclose(lib_);
    lib_ = nullptr;
  }

  void* lib_ = nullptr;
  void* sub_ = nullptr;
  CFMutableDictionaryRef channels_ = nullptr;
  CFDictionaryRef start_ = nullptr;
  CreateSamples create_samples_ = nullptr;
  CreateSamplesDelta create_delta_ = nullptr;
  ChannelGetString get_name_ = nullptr;
  ChannelGetString get_unit_ = nullptr;
  SimpleGetInteger get_int_ = nullptr;
#else
  void Open() { reason_ = "no IOReport on this platform"; }
  void Close() {}
  void* lib_ = nullptr;
#endif
  std::string reason_;
};

}  // namespace seeml::bench

#endif  // SEEML_TOOL_ENERGY_METER_H_
