// Can an unprivileged process read the SoC's energy counters on this OS?
// Opens the private IOReport framework at run time (no link-time
// dependency), subscribes to the "Energy Model" channel group, samples
// twice one second apart and prints each channel's delta with its unit.
#import <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <cstdio>
#include <unistd.h>

typedef struct IOReportSubscription* IOReportSubscriptionRef;
typedef CFDictionaryRef IOReportSampleRef;
typedef CFDictionaryRef (*CopyChannelsInGroup)(CFStringRef, CFStringRef, uint64_t,
                                               uint64_t, uint64_t);
typedef IOReportSubscriptionRef (*CreateSubscription)(void*, CFMutableDictionaryRef,
                                                      CFMutableDictionaryRef*,
                                                      uint64_t, CFTypeRef);
typedef CFDictionaryRef (*CreateSamples)(IOReportSubscriptionRef,
                                         CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*CreateSamplesDelta)(CFDictionaryRef, CFDictionaryRef,
                                              CFTypeRef);
typedef CFStringRef (*ChannelGetString)(IOReportSampleRef);
typedef int64_t (*SimpleGetInteger)(IOReportSampleRef, int);

static void Print(CFStringRef s) {
  char buf[256];
  if (s && CFStringGetCString(s, buf, sizeof buf, kCFStringEncodingUTF8))
    std::fputs(buf, stdout);
  else
    std::fputs("?", stdout);
}

int main() {
  // macOS 26 ships it as /usr/lib/libIOReport.dylib (what powermetrics
  // links); earlier releases as a private framework.
  void* lib = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
  if (!lib)
    lib = dlopen("/System/Library/PrivateFrameworks/IOReport.framework/IOReport",
                 RTLD_NOW);
  if (!lib) {
    std::printf("unavailable: %s\n", dlerror());
    return 1;
  }
  auto copy_group = (CopyChannelsInGroup)dlsym(lib, "IOReportCopyChannelsInGroup");
  auto create_sub = (CreateSubscription)dlsym(lib, "IOReportCreateSubscription");
  auto create_samples = (CreateSamples)dlsym(lib, "IOReportCreateSamples");
  auto delta = (CreateSamplesDelta)dlsym(lib, "IOReportCreateSamplesDelta");
  auto get_name = (ChannelGetString)dlsym(lib, "IOReportChannelGetChannelName");
  auto get_group = (ChannelGetString)dlsym(lib, "IOReportChannelGetGroup");
  auto get_unit = (ChannelGetString)dlsym(lib, "IOReportChannelGetUnitLabel");
  auto get_int = (SimpleGetInteger)dlsym(lib, "IOReportSimpleGetIntegerValue");
  if (!copy_group || !create_sub || !create_samples || !delta || !get_name ||
      !get_unit || !get_int) {
    std::puts("unavailable: a symbol is missing");
    return 1;
  }
  CFDictionaryRef channels = copy_group(CFSTR("Energy Model"), nullptr, 0, 0, 0);
  if (!channels) {
    std::puts("unavailable: no Energy Model channel group");
    return 1;
  }
  CFMutableDictionaryRef wanted = CFDictionaryCreateMutableCopy(nullptr, 0, channels);
  CFMutableDictionaryRef subscribed = nullptr;
  IOReportSubscriptionRef sub = create_sub(nullptr, wanted, &subscribed, 0, nullptr);
  if (!sub || !subscribed) {
    std::puts("unavailable: subscription refused");
    return 1;
  }
  auto get_format = (int (*)(IOReportSampleRef))dlsym(lib, "IOReportChannelGetFormat");
  CFDictionaryRef a = create_samples(sub, subscribed, nullptr);
  // One second of arithmetic on this core, so the CPU channels have
  // something to count; the busy loop's result is printed so it survives.
  volatile double acc = 0.0;
  const CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
  while (CFAbsoluteTimeGetCurrent() - t0 < 1.0)
    for (int i = 0; i < 100000; ++i) acc = acc + 1e-9 * i;
  std::printf("busy %.1f\n", (double)acc);
  CFDictionaryRef b = create_samples(sub, subscribed, nullptr);
  CFDictionaryRef d = delta(a, b, nullptr);
  CFArrayRef rows = (CFArrayRef)CFDictionaryGetValue(d, CFSTR("IOReportChannels"));
  const CFIndex n = rows ? CFArrayGetCount(rows) : 0;
  std::printf("uid=%d channels=%ld (1 s delta)\n", (int)getuid(), (long)n);
  long quiet = 0;
  for (CFIndex i = 0; i < n; ++i) {
    IOReportSampleRef row = (IOReportSampleRef)CFArrayGetValueAtIndex(rows, i);
    const long long value = get_int(row, 0);
    if (value == 0) {  // the channels that read zero to this reader
      ++quiet;
      continue;
    }
    std::fputs("  ", stdout);
    Print(get_group(row));
    std::fputs(" / ", stdout);
    Print(get_name(row));
    std::printf(" = %lld ", value);
    Print(get_unit(row));
    std::printf(" (format %d)\n", get_format ? get_format(row) : -1);
  }
  std::printf("  %ld channel(s) read zero over the busy second (every CPU, "
              "DRAM and ANE channel among them)\n", quiet);
  return 0;
}
