#pragma once

#include <cstddef>
#include <cstdint>
#include <linux/idxd.h>
#include <optional>

// CRC32C and copy+CRC32C offload to Intel DSA through shared user-mode work queues (ENQCMD).
// Enabled by env HF3FS_DSA_WQ=/dev/dsa/wq0.0[,/dev/dsa/wq2.0...]; blocks smaller than
// HF3FS_DSA_MIN (default 64KB) stay on the CPU. CRC values use the 3FS convention,
// i.e. the same value as folly::crc32c(data, len, ~0U).
namespace hf3fs::dsa {

struct alignas(64) Job {
  dsa_hw_desc desc;
  alignas(64) dsa_completion_record comp;
};

bool enabled();
size_t minBytes();

// dst == nullptr: CRC only; otherwise copy src to dst and CRC src in one descriptor.
bool submit(Job &job, void *dst, const void *src, size_t len);
bool done(const Job &job);
std::optional<uint32_t> result(const Job &job);  // nullopt if DSA reported an error
std::optional<uint32_t> wait(Job &job);

std::optional<uint32_t> crc32c(const void *src, size_t len);
std::optional<uint32_t> copyCrc32c(void *dst, const void *src, size_t len);

// CRC of a buffer produced by copyCrc32c, handed to the storage client so it need not re-read the data.
void putHint(const void *buf, size_t len, uint32_t crc);
void dropHint(const void *buf);
std::optional<uint32_t> takeHint(const void *buf, size_t len);

}  // namespace hf3fs::dsa
