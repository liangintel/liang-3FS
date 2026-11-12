//! CRC32C offload of incoming write data to Intel DSA (shared user WQ, ENQCMD), so the CRC check
//! runs on the accelerator while the data is being written to disk. Enabled by env HF3FS_DSA_WQ
//! (comma-separated WQ device paths); blocks below HF3FS_DSA_MIN (default 64KB) stay on the CPU.
//! On non-x86_64 targets DSA is never enabled and all callers fall back to the CPU.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

const OP_CRCGEN: u32 = 0x10;
const FLAGS: u32 = 0x8 | 0x4 | 0x2; // request completion record, completion address valid, block on fault
const MAX_XFER: usize = 2 << 20;

struct State {
    portals: Vec<usize>,
    min_bytes: usize,
}

fn state() -> &'static State {
    static STATE: OnceLock<State> = OnceLock::new();
    STATE.get_or_init(|| {
        let mut portals = vec![];
        let min_bytes = std::env::var("HF3FS_DSA_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64 << 10);
        if let Some(list) = std::env::var("HF3FS_DSA_WQ")
            .ok()
            .filter(|_| cfg!(target_arch = "x86_64"))
        {
            for path in list.split(',').filter(|p| !p.is_empty()) {
                let Ok(cpath) = std::ffi::CString::new(path) else {
                    continue;
                };
                unsafe {
                    let fd = libc::open(cpath.as_ptr(), libc::O_RDWR);
                    if fd < 0 {
                        tracing::error!("DSA: open {} failed", path);
                        continue;
                    }
                    let p = libc::mmap(
                        std::ptr::null_mut(),
                        0x1000,
                        libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_POPULATE,
                        fd,
                        0,
                    );
                    libc::close(fd);
                    if p == libc::MAP_FAILED {
                        tracing::error!("DSA: mmap {} failed", path);
                        continue;
                    }
                    portals.push(p as usize);
                }
            }
            tracing::info!(
                "DSA: {} work queue(s) from {}, min block {}",
                portals.len(),
                list,
                min_bytes
            );
        }
        State { portals, min_bytes }
    })
}

pub fn enabled() -> bool {
    !state().portals.is_empty()
}

pub fn min_bytes() -> usize {
    state().min_bytes
}

fn portal() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        static IDX: usize = COUNTER.fetch_add(1, Ordering::Relaxed);
    }
    let p = &state().portals;
    p[IDX.with(|i| *i) % p.len()]
}

#[repr(C, align(64))]
struct Job {
    desc: [u64; 8],
    comp: [u64; 8],
}

/// A CRC32C job in flight; dropping it waits for completion so DSA never writes freed memory.
pub struct Pending {
    job: Box<Job>,
}

impl Pending {
    /// Returns None if DSA is unavailable or the queue stays full; the caller then uses the CPU.
    pub fn submit(data: &[u8]) -> Option<Pending> {
        if !enabled() || data.len() < min_bytes() || data.len() > MAX_XFER {
            return None;
        }
        let mut job = Box::new(Job {
            desc: [0; 8],
            comp: [0; 8],
        });
        job.desc[0] = ((OP_CRCGEN << 24 | FLAGS) as u64) << 32;
        job.desc[1] = job.comp.as_ptr() as u64;
        job.desc[2] = data.as_ptr() as u64;
        job.desc[4] = data.len() as u64; // xfer_size; crc_seed (desc[5]) = 0
        let portal = portal();
        for _ in 0..4096 {
            if unsafe { enqcmd(portal, job.desc.as_ptr() as usize) } {
                return Some(Pending { job });
            }
            std::hint::spin_loop();
        }
        None
    }

    fn status(&self) -> u8 {
        unsafe { std::ptr::read_volatile(self.job.comp.as_ptr() as *const u8) }
    }

    /// Standard CRC32C (same as crc32c::crc32c), or None if DSA reported an error.
    pub fn wait(self) -> Option<u32> {
        while self.status() == 0 {
            std::hint::spin_loop();
        }
        if self.status() & 0x7f != 1 {
            tracing::warn!("DSA: crc status {:#x}", self.status());
            return None;
        }
        Some(unsafe { std::ptr::read_volatile(self.job.comp.as_ptr().add(2)) } as u32)
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        while self.status() == 0 {
            std::hint::spin_loop();
        }
    }
}

/// ENQCMD: returns true if the work queue accepted the descriptor.
#[cfg(target_arch = "x86_64")]
unsafe fn enqcmd(portal: usize, desc: usize) -> bool {
    let rejected: u8;
    std::arch::asm!(
        "sfence",
        "enqcmd {p}, zmmword ptr [{d}]",
        "setz {r}",
        p = in(reg) portal,
        d = in(reg) desc,
        r = out(reg_byte) rejected,
        options(nostack),
    );
    rejected == 0
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn enqcmd(_portal: usize, _desc: usize) -> bool {
    false
}
