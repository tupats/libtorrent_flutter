// memory_disk_io.hpp — in-memory libtorrent 2.x disk_interface backend.
//
// Replaces mmap_disk_io / posix_disk_io. All piece data lives in process
// memory (unordered_map<piece_index, vector<uint8_t>>); no file I/O.
// Target: streaming on Android where eMMC iowait pins the system.
//
// v0: bounded LRU eviction by global cap. No head-aware window.
// v1 (TODO): playback-head awareness + clear_piece plumbing on eviction.

#pragma once

#include <libtorrent/disk_interface.hpp>
#include <libtorrent/disk_buffer_holder.hpp>
#include <libtorrent/io_context.hpp>
#include <libtorrent/aux_/session_settings.hpp>
#include <libtorrent/aux_/vector.hpp>
#include <libtorrent/aux_/storage_free_list.hpp>
#include <libtorrent/performance_counters.hpp>
#include <libtorrent/file_storage.hpp>
#include <libtorrent/storage_defs.hpp>
#include <libtorrent/peer_request.hpp>
#include <libtorrent/sha1_hash.hpp>
#include <libtorrent/hasher.hpp>
#include <libtorrent/add_torrent_params.hpp>
#include <libtorrent/operations.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tb {

struct memory_stats_t {
    int64_t used_bytes;
    int64_t cap_bytes;
    int64_t piece_count;
    int64_t evicted_count;
};

class memory_disk_io;

class memory_storage_t {
public:
    memory_storage_t(libtorrent::storage_params const& params,
                     memory_disk_io* owner);
    ~memory_storage_t();

    // returns bytes read (>0) on success, 0 on EOF, sets ec on error
    int read(libtorrent::span<char> buf,
             libtorrent::piece_index_t piece, int offset,
             libtorrent::storage_error& ec);

    int write(libtorrent::span<char const> buf,
              libtorrent::piece_index_t piece, int offset,
              libtorrent::storage_error& ec);

    void clear_piece(libtorrent::piece_index_t piece);
    void clear_all();

    libtorrent::file_storage const& files() const { return m_files; }
    libtorrent::sha1_hash const& info_hash() const { return m_info_hash; }

    // evict LRU pieces until total owner usage ≤ cap. Skips pieces in the
    // playback window. Returns indices evicted (forwarded to clear_piece
    // subsystem by the owner).
    std::vector<libtorrent::piece_index_t> evict_lru_to_cap(int64_t cap_target);

    int64_t bytes_held() const { return m_bytes_held.load(); }
    int64_t piece_count() const {
        std::lock_guard<std::mutex> lk(m_mu);
        return static_cast<int64_t>(m_pieces.size());
    }

    // Playback head + window — bytes are absolute offsets into the torrent
    // (sum of all file sizes laid out left-to-right by file_storage).
    void set_head(int64_t byte_offset) { m_head_byte.store(byte_offset); }
    void set_window(int64_t rewind_bytes, int64_t prefetch_bytes) {
        m_rewind_bytes.store(rewind_bytes);
        m_prefetch_bytes.store(prefetch_bytes);
    }

private:
    // True iff piece [p] sits inside the live playback window
    // [head - rewind, head + prefetch] (bytes converted via piece_length).
    bool in_window(int p) const;

    libtorrent::file_storage m_files;        // owned copy
    libtorrent::sha1_hash    m_info_hash;
    memory_disk_io*          m_owner;

    mutable std::mutex m_mu;

    struct piece_buf {
        std::vector<uint8_t> data;   // full piece, sized lazily on first write
        uint64_t access_seq;
    };

    std::unordered_map<int, piece_buf> m_pieces;
    std::atomic<int64_t> m_bytes_held{0};

    std::atomic<int64_t> m_head_byte{-1};                            // -1 = unset
    std::atomic<int64_t> m_rewind_bytes{int64_t(16) * 1024 * 1024};  // ~16 MB default
    std::atomic<int64_t> m_prefetch_bytes{int64_t(160) * 1024 * 1024}; // ~160 MB default
};

// Minimal allocator backing disk_buffer_holder. libtorrent's internal
// disk_buffer_pool symbols are TORRENT_EXTRA_EXPORT and not present in the
// installed dylib, so we substitute a heap allocator.
class trivial_buffer_allocator final
    : public libtorrent::buffer_allocator_interface
{
public:
    void free_disk_buffer(char* b) override { delete[] b; }
};

class memory_disk_io final : public libtorrent::disk_interface {
public:
    memory_disk_io(libtorrent::io_context& ios,
                   libtorrent::settings_interface const& sett,
                   libtorrent::counters& cnt);
    ~memory_disk_io() override;

    // ── disk_interface ────────────────────────────────────────────────
    void settings_updated() override;
    libtorrent::storage_holder new_torrent(
        libtorrent::storage_params const& params,
        std::shared_ptr<void> const& torrent) override;
    void remove_torrent(libtorrent::storage_index_t idx) override;
    void abort(bool wait) override;

    void async_read(libtorrent::storage_index_t storage,
        libtorrent::peer_request const& r,
        std::function<void(libtorrent::disk_buffer_holder,
                           libtorrent::storage_error const&)> handler,
        libtorrent::disk_job_flags_t flags) override;

    bool async_write(libtorrent::storage_index_t storage,
        libtorrent::peer_request const& r,
        char const* buf,
        std::shared_ptr<libtorrent::disk_observer> o,
        std::function<void(libtorrent::storage_error const&)> handler,
        libtorrent::disk_job_flags_t flags) override;

    void async_hash(libtorrent::storage_index_t storage,
        libtorrent::piece_index_t piece,
        libtorrent::span<libtorrent::sha256_hash> v2,
        libtorrent::disk_job_flags_t flags,
        std::function<void(libtorrent::piece_index_t,
                           libtorrent::sha1_hash const&,
                           libtorrent::storage_error const&)> handler) override;

    void async_hash2(libtorrent::storage_index_t storage,
        libtorrent::piece_index_t piece, int offset,
        libtorrent::disk_job_flags_t flags,
        std::function<void(libtorrent::piece_index_t,
                           libtorrent::sha256_hash const&,
                           libtorrent::storage_error const&)> handler) override;

    void async_move_storage(libtorrent::storage_index_t storage,
        std::string p, libtorrent::move_flags_t flags,
        std::function<void(libtorrent::status_t, std::string const&,
                           libtorrent::storage_error const&)> handler) override;

    void async_release_files(libtorrent::storage_index_t storage,
        std::function<void()> handler) override;

    void async_check_files(libtorrent::storage_index_t storage,
        libtorrent::add_torrent_params const* resume_data,
        libtorrent::aux::vector<std::string, libtorrent::file_index_t> links,
        std::function<void(libtorrent::status_t,
                           libtorrent::storage_error const&)> handler) override;

    void async_stop_torrent(libtorrent::storage_index_t storage,
        std::function<void()> handler) override;

    void async_rename_file(libtorrent::storage_index_t storage,
        libtorrent::file_index_t idx, std::string name,
        std::function<void(std::string const&, libtorrent::file_index_t,
                           libtorrent::storage_error const&)> handler) override;

    void async_delete_files(libtorrent::storage_index_t storage,
        libtorrent::remove_flags_t options,
        std::function<void(libtorrent::storage_error const&)> handler) override;

    void async_set_file_priority(libtorrent::storage_index_t storage,
        libtorrent::aux::vector<libtorrent::download_priority_t,
                                 libtorrent::file_index_t> prio,
        std::function<void(libtorrent::storage_error const&,
            libtorrent::aux::vector<libtorrent::download_priority_t,
                                     libtorrent::file_index_t>)> handler) override;

    void async_clear_piece(libtorrent::storage_index_t storage,
        libtorrent::piece_index_t index,
        std::function<void(libtorrent::piece_index_t)> handler) override;

    void update_stats_counters(libtorrent::counters& c) const override;
    std::vector<libtorrent::open_file_state> get_status(
        libtorrent::storage_index_t) const override;
    void submit_jobs() override;

    // ── tb-specific API ───────────────────────────────────────────────
    libtorrent::settings_interface const& settings() const { return m_settings; }
    libtorrent::counters& stats_counters() { return m_stats_counters; }
    libtorrent::io_context& ios() { return m_ios; }

    int64_t used_bytes() const { return m_used_bytes.load(); }
    int64_t cap_bytes()  const { return m_cap_bytes.load(); }
    int64_t evicted_count() const { return m_evicted_count.load(); }

    void set_cap_bytes(int64_t bytes) { m_cap_bytes.store(bytes); }

    void add_used(int64_t delta) { m_used_bytes.fetch_add(delta); }
    void inc_evicted(int64_t n = 1) { m_evicted_count.fetch_add(n); }
    uint64_t next_access_seq() { return m_access_seq.fetch_add(1) + 1; }

    memory_stats_t stats() const;

    // Sweep all storages, evicting LRU until under cap.
    // Returns total evicted count.
    int64_t enforce_cap();

    // Process-singleton accessor (set in ctor, cleared in dtor).
    static memory_disk_io* instance();

    // ── playback head / window control ────────────────────────────────
    // Set per-torrent playback head, identified by info_hash.
    // byte_offset is absolute within the torrent (not within a file).
    bool set_head_for_info_hash(libtorrent::sha1_hash const& ih,
                                int64_t byte_offset);
    bool set_window_for_info_hash(libtorrent::sha1_hash const& ih,
                                  int64_t rewind_bytes,
                                  int64_t prefetch_bytes);

    // Eviction queue — pieces dropped from memory that libtorrent still
    // believes complete. Drained by the bridge alert thread, which calls
    // torrent_handle::clear_piece to make libtorrent re-fetch on demand.
    struct evicted_piece {
        libtorrent::sha1_hash info_hash;
        libtorrent::piece_index_t piece;
    };
    std::vector<evicted_piece> drain_eviction_queue();
    void push_eviction(libtorrent::sha1_hash const& ih,
                       libtorrent::piece_index_t p);

    trivial_buffer_allocator& buffer_allocator() { return m_buffer_allocator; }

private:
    libtorrent::settings_interface const& m_settings;
    trivial_buffer_allocator              m_buffer_allocator;
    libtorrent::counters&                 m_stats_counters;
    libtorrent::io_context&               m_ios;

    mutable std::mutex m_storages_mu;
    libtorrent::aux::vector<std::unique_ptr<memory_storage_t>,
                            libtorrent::storage_index_t> m_storages;
    libtorrent::aux::storage_free_list m_free_slots;

    std::atomic<int64_t>  m_used_bytes{0};
    std::atomic<int64_t>  m_cap_bytes{int64_t(256) * 1024 * 1024};  // 256 MB default
    std::atomic<int64_t>  m_evicted_count{0};
    std::atomic<uint64_t> m_access_seq{0};

    std::mutex                  m_evict_mu;
    std::vector<evicted_piece>  m_evict_queue;
};

// disk_io_constructor compatible signature.
std::unique_ptr<libtorrent::disk_interface> memory_disk_io_constructor(
    libtorrent::io_context& ios,
    libtorrent::settings_interface const& sett,
    libtorrent::counters& cnt);

} // namespace tb
