#include "common/utils/DsaCrc.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <folly/logging/xlog.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace hf3fs::dsa {
namespace {

constexpr size_t kMaxXfer = 2 << 20;  // WQ max_transfer_size

#if defined(__x86_64__)
constexpr bool kSupported = true;

__attribute__((target("enqcmd"))) bool enqueue(void *portal, const void *desc) {
  _mm_sfence();  // descriptor and zeroed completion record must be visible before the device reads them
  for (int i = 0; i < 4096; i++) {
    if (!_enqcmd(portal, desc)) return true;  // ZF=0: accepted
    _mm_pause();
  }
  return false;
}

inline void cpuRelax() { _mm_pause(); }
#else
constexpr bool kSupported = false;
bool enqueue(void *, const void *) { return false; }
inline void cpuRelax() {}
#endif

struct State {
  std::vector<void *> portals;
  size_t minBytes = 64 << 10;
  std::atomic<uint64_t> ops{0}, bytes{0}, fallbacks{0};

  State() {
    const char *env = getenv("HF3FS_DSA_WQ");
    if (!env || !kSupported) return;
    if (const char *m = getenv("HF3FS_DSA_MIN")) minBytes = strtoul(m, nullptr, 0);
    std::string list(env);
    size_t pos = 0;
    while (pos <= list.size()) {
      auto end = list.find(',', pos);
      if (end == std::string::npos) end = list.size();
      auto path = list.substr(pos, end - pos);
      pos = end + 1;
      if (path.empty()) continue;
      int fd = open(path.c_str(), O_RDWR);
      if (fd < 0) {
        XLOGF(ERR, "DSA: open {} failed: {}", path, strerror(errno));
        continue;
      }
      void *p = mmap(nullptr, 0x1000, PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
      close(fd);
      if (p == MAP_FAILED) {
        XLOGF(ERR, "DSA: mmap {} failed: {}", path, strerror(errno));
        continue;
      }
      portals.push_back(p);
    }
    XLOGF(INFO, "DSA: {} work queue(s) from {}, min block {} bytes", portals.size(), env, minBytes);
  }
};

State &state() {
  static State s;
  return s;
}

std::atomic<unsigned> threadCounter{0};

void *portal() {
  thread_local unsigned idx = threadCounter.fetch_add(1);
  auto &p = state().portals;
  return p[idx % p.size()];
}

void count(size_t len, bool ok) {
  auto &s = state();
  if (!ok) {
    s.fallbacks.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  s.bytes.fetch_add(len, std::memory_order_relaxed);
  auto n = s.ops.fetch_add(1, std::memory_order_relaxed) + 1;
  if ((n & 0xffff) == 0) {
    XLOGF(INFO,
          "DSA: {} ops, {} GiB, {} fallbacks",
          n,
          s.bytes.load(std::memory_order_relaxed) >> 30,
          s.fallbacks.load(std::memory_order_relaxed));
  }
}

std::mutex hintMu;
std::unordered_map<const void *, std::pair<size_t, uint32_t>> hints;
std::atomic<size_t> hintCount{0};

}  // namespace

bool enabled() { return !state().portals.empty(); }

size_t minBytes() { return state().minBytes; }

bool submit(Job &job, void *dst, const void *src, size_t len) {
  if (!enabled() || len == 0 || len > kMaxXfer) return false;
  memset(&job.desc, 0, sizeof(job.desc));
  memset(&job.comp, 0, sizeof(job.comp));
  job.desc.opcode = dst ? DSA_OPCODE_COPY_CRC : DSA_OPCODE_CRCGEN;
  job.desc.flags = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_BOF | (dst ? IDXD_OP_FLAG_CC : 0);
  job.desc.src_addr = (uintptr_t)src;
  job.desc.dst_addr = (uintptr_t)dst;
  job.desc.xfer_size = len;
  job.desc.crc_seed = 0;
  job.desc.completion_addr = (uintptr_t)&job.comp;
  bool ok = enqueue(portal(), &job.desc);
  if (!ok) count(len, false);
  return ok;
}

bool done(const Job &job) { return reinterpret_cast<const volatile uint8_t &>(job.comp.status) != 0; }

std::optional<uint32_t> result(const Job &job) {
  bool ok = (job.comp.status & DSA_COMP_STATUS_MASK) == DSA_COMP_SUCCESS;
  count(job.desc.xfer_size, ok);
  if (!ok) {
    XLOGF(WARN, "DSA: op {} len {} status {:#x}", job.desc.opcode, job.desc.xfer_size, unsigned(job.comp.status));
    return std::nullopt;
  }
  // DSA with seed 0 returns the standard (final-inverted) CRC32C; 3FS keeps the non-inverted form.
  return ~(uint32_t)job.comp.crc_val;
}

std::optional<uint32_t> wait(Job &job) {
  while (!done(job)) cpuRelax();
  return result(job);
}

std::optional<uint32_t> crc32c(const void *src, size_t len) {
  Job job;
  if (!submit(job, nullptr, src, len)) return std::nullopt;
  return wait(job);
}

std::optional<uint32_t> copyCrc32c(void *dst, const void *src, size_t len) {
  Job job;
  if (!submit(job, dst, src, len)) return std::nullopt;
  return wait(job);
}

void putHint(const void *buf, size_t len, uint32_t crc) {
  std::lock_guard lock(hintMu);
  if (hints.insert_or_assign(buf, std::make_pair(len, crc)).second) hintCount.fetch_add(1);
}

void dropHint(const void *buf) {
  if (hintCount.load(std::memory_order_relaxed) == 0) return;
  std::lock_guard lock(hintMu);
  if (hints.erase(buf)) hintCount.fetch_sub(1);
}

std::optional<uint32_t> takeHint(const void *buf, size_t len) {
  if (hintCount.load(std::memory_order_relaxed) == 0) return std::nullopt;
  std::lock_guard lock(hintMu);
  auto it = hints.find(buf);
  if (it == hints.end()) return std::nullopt;
  auto [hlen, crc] = it->second;
  hints.erase(it);
  hintCount.fetch_sub(1);
  if (hlen != len) return std::nullopt;
  return crc;
}

}  // namespace hf3fs::dsa
