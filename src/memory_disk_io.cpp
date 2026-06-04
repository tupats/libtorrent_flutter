// memory_disk_io.cpp — see memory_disk_io.hpp.

#include "memory_disk_io.hpp"

#include <libtorrent/error_code.hpp>
#include <libtorrent/settings_pack.hpp>
#include <libtorrent/time.hpp>
#include <libtorrent/units.hpp>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <utility>
#include <vector>

namespace tb {

namespace lt = libtorrent;

namespace {

std::atomic<memory_disk_io*> g_instance{nullptr};

constexpr int kDefaultBlockSize = lt::default_block_size;  // 16 KiB

} // namespace

memory_disk_io* memory_disk_io::instance() { return g_instance.load(); }

// ── memory_storage_t ─────────────────────────────────────────────────────

memory_storage_t::memory_storage_t(lt::storage_params const& params,
                                   memory_disk_io* owner)
    : m_files(params.files)
    , m_info_hash(params.info_hash)
    , m_owner(owner)
{}

bool memory_storage_t::in_window(int p) const {
    int64_t const head = m_head_byte.load();
    if (head < 0) return false;
    int64_t const piece_len = static_cast<int64_t>(m_files.piece_length());
    if (piece_len <= 0) return false;
    int64_t const head_piece = head / piece_len;
    int64_t const rewind_pieces =
        (m_rewind_bytes.load() + piece_len - 1) / piece_len;
    int64_t const prefetch_pieces =
        (m_prefetch_bytes.load() + piece_len - 1) / piece_len;
    int64_t const lo = head_piece - rewind_pieces;
    int64_t const hi = head_piece + prefetch_pieces;
    return int64_t(p) >= lo && int64_t(p) <= hi;
}

memory_storage_t::~memory_storage_t() {
    clear_all();
}

int memory_storage_t::read(lt::span<char> buf,
                           lt::piece_index_t piece, int offset,
                           lt::storage_error& ec)
{
    int const p = static_cast<int>(piece);
    int const piece_sz = m_files.piece_size(piece);
    if (offset < 0 || offset >= piece_sz) {
        ec.ec = lt::errors::invalid_request;
        ec.operation = lt::operation_t::file_read;
        return -1;
    }
    int const len = std::min(int(buf.size()), piece_sz - offset);

    std::lock_guard<std::mutex> lk(m_mu);
    auto it = m_pieces.find(p);
    if (it == m_pieces.end() || it->second.data.empty()) {
        // Piece evicted or never written. Surface as short read; libtorrent
        // treats this as a missing piece and will re-fetch when prioritized.
        ec.ec = lt::errors::file_too_short;
        ec.operation = lt::operation_t::file_read;
        return 0;
    }
    it->second.access_seq = m_owner->next_access_seq();
    std::memcpy(buf.data(), it->second.data.data() + offset,
                static_cast<size_t>(len));
    return len;
}

int memory_storage_t::write(lt::span<char const> buf,
                            lt::piece_index_t piece, int offset,
                            lt::storage_error& ec)
{
    int const p = static_cast<int>(piece);
    int const piece_sz = m_files.piece_size(piece);
    if (offset < 0 || offset > piece_sz) {
        ec.ec = lt::errors::invalid_request;
        ec.operation = lt::operation_t::file_write;
        return -1;
    }
    int const len = std::min(int(buf.size()), piece_sz - offset);

    std::lock_guard<std::mutex> lk(m_mu);
    auto& pb = m_pieces[p];
    if (pb.data.empty()) {
        pb.data.assign(static_cast<size_t>(piece_sz), 0);
        m_bytes_held.fetch_add(piece_sz);
        m_owner->add_used(piece_sz);
    }
    pb.access_seq = m_owner->next_access_seq();
    std::memcpy(pb.data.data() + offset, buf.data(),
                static_cast<size_t>(len));
    return len;
}

void memory_storage_t::clear_piece(lt::piece_index_t piece) {
    int const p = static_cast<int>(piece);
    std::lock_guard<std::mutex> lk(m_mu);
    auto it = m_pieces.find(p);
    if (it == m_pieces.end()) return;
    int64_t const freed = static_cast<int64_t>(it->second.data.size());
    m_bytes_held.fetch_sub(freed);
    m_owner->add_used(-freed);
    m_pieces.erase(it);
}

void memory_storage_t::clear_all() {
    std::lock_guard<std::mutex> lk(m_mu);
    int64_t total = 0;
    for (auto& kv : m_pieces) total += kv.second.data.size();
    m_pieces.clear();
    m_bytes_held.fetch_sub(total);
    if (m_owner) m_owner->add_used(-total);
}

std::vector<lt::piece_index_t>
memory_storage_t::evict_lru_to_cap(int64_t cap_target)
{
    std::vector<lt::piece_index_t> evicted;
    if (m_owner->used_bytes() <= cap_target) return evicted;

    std::lock_guard<std::mutex> lk(m_mu);

    // Sort pieces by access_seq ascending (oldest first).
    std::vector<std::pair<uint64_t, int>> order;
    order.reserve(m_pieces.size());
    for (auto& kv : m_pieces) {
        order.emplace_back(kv.second.access_seq, kv.first);
    }
    std::sort(order.begin(), order.end(),
              [](auto const& a, auto const& b) { return a.first < b.first; });

    // Pass 1: evict only outside the playback window. This protects pieces
    // the player is currently consuming or about to consume from being
    // dropped while the cap is enforced.
    for (auto& [seq, p] : order) {
        if (m_owner->used_bytes() <= cap_target) break;
        if (in_window(p)) continue;
        auto it = m_pieces.find(p);
        if (it == m_pieces.end()) continue;
        int64_t const freed = static_cast<int64_t>(it->second.data.size());
        m_bytes_held.fetch_sub(freed);
        m_owner->add_used(-freed);
        m_pieces.erase(it);
        evicted.push_back(lt::piece_index_t{p});
    }

    // Pass 2: if still over cap (cap < window working set), fall back to
    // pure LRU including window pieces — better to evict than to OOM.
    if (m_owner->used_bytes() > cap_target) {
        for (auto& [seq, p] : order) {
            if (m_owner->used_bytes() <= cap_target) break;
            auto it = m_pieces.find(p);
            if (it == m_pieces.end()) continue;
            int64_t const freed = static_cast<int64_t>(it->second.data.size());
            m_bytes_held.fetch_sub(freed);
            m_owner->add_used(-freed);
            m_pieces.erase(it);
            evicted.push_back(lt::piece_index_t{p});
        }
    }
    return evicted;
}

// ── memory_disk_io ───────────────────────────────────────────────────────

memory_disk_io::memory_disk_io(lt::io_context& ios,
                               lt::settings_interface const& sett,
                               lt::counters& cnt)
    : m_settings(sett)
    , m_stats_counters(cnt)
    , m_ios(ios)
{
    memory_disk_io* expected = nullptr;
    g_instance.compare_exchange_strong(expected, this);
    settings_updated();
}

memory_disk_io::~memory_disk_io() {
    memory_disk_io* self = this;
    g_instance.compare_exchange_strong(self, nullptr);
}

void memory_disk_io::settings_updated() {}

lt::storage_holder memory_disk_io::new_torrent(
    lt::storage_params const& params,
    std::shared_ptr<void> const& /*torrent*/)
{
    std::lock_guard<std::mutex> lk(m_storages_mu);
    lt::storage_index_t const idx =
        m_free_slots.new_index(m_storages.end_index());
    auto st = std::make_unique<memory_storage_t>(params, this);
    if (idx == m_storages.end_index()) m_storages.emplace_back(std::move(st));
    else m_storages[idx] = std::move(st);
    return lt::storage_holder(idx, *this);
}

void memory_disk_io::remove_torrent(lt::storage_index_t idx) {
    std::lock_guard<std::mutex> lk(m_storages_mu);
    m_storages[idx].reset();
    m_free_slots.add(idx);
}

void memory_disk_io::abort(bool) {}

void memory_disk_io::async_read(lt::storage_index_t storage,
    lt::peer_request const& r,
    std::function<void(lt::disk_buffer_holder,
                       lt::storage_error const&)> handler,
    lt::disk_job_flags_t)
{
    lt::disk_buffer_holder buffer(m_buffer_allocator,
        new (std::nothrow) char[kDefaultBlockSize], kDefaultBlockSize);
    lt::storage_error error;
    if (!buffer) {
        error.ec = lt::errors::no_memory;
        error.operation = lt::operation_t::alloc_cache_piece;
        lt::post(m_ios, [this, error, h = std::move(handler)]() mutable {
            h(lt::disk_buffer_holder(m_buffer_allocator, nullptr, 0), error);
        });
        return;
    }

    lt::time_point const start = lt::clock_type::now();
    lt::span<char> const buf{buffer.data(), r.length};

    memory_storage_t* st = nullptr;
    {
        std::lock_guard<std::mutex> lk(m_storages_mu);
        st = m_storages[storage].get();
    }
    if (!st) {
        error.ec = lt::errors::torrent_removed;
        error.operation = lt::operation_t::file_read;
        lt::post(m_ios, [this, error, h = std::move(handler)]() mutable {
            h(lt::disk_buffer_holder(m_buffer_allocator, nullptr, 0), error);
        });
        return;
    }

    st->read(buf, r.piece, r.start, error);

    if (!error.ec) {
        int64_t const us = lt::total_microseconds(lt::clock_type::now() - start);
        m_stats_counters.inc_stats_counter(lt::counters::num_blocks_read);
        m_stats_counters.inc_stats_counter(lt::counters::num_read_ops);
        m_stats_counters.inc_stats_counter(lt::counters::disk_read_time, us);
        m_stats_counters.inc_stats_counter(lt::counters::disk_job_time, us);
    }

    lt::post(m_ios, [h = std::move(handler), b = std::move(buffer),
                     error]() mutable { h(std::move(b), error); });
}

bool memory_disk_io::async_write(lt::storage_index_t storage,
    lt::peer_request const& r, char const* buf,
    std::shared_ptr<lt::disk_observer>,
    std::function<void(lt::storage_error const&)> handler,
    lt::disk_job_flags_t)
{
    lt::span<char const> const b{buf, r.length};
    lt::time_point const start = lt::clock_type::now();
    lt::storage_error error;

    memory_storage_t* st = nullptr;
    {
        std::lock_guard<std::mutex> lk(m_storages_mu);
        st = m_storages[storage].get();
    }
    if (!st) {
        error.ec = lt::errors::torrent_removed;
        error.operation = lt::operation_t::file_write;
        lt::post(m_ios, [error, h = std::move(handler)]() { h(error); });
        return false;
    }

    st->write(b, r.piece, r.start, error);

    if (!error.ec) {
        int64_t const us = lt::total_microseconds(lt::clock_type::now() - start);
        m_stats_counters.inc_stats_counter(lt::counters::num_blocks_written);
        m_stats_counters.inc_stats_counter(lt::counters::num_write_ops);
        m_stats_counters.inc_stats_counter(lt::counters::disk_write_time, us);
        m_stats_counters.inc_stats_counter(lt::counters::disk_job_time, us);
    }

    // Best-effort eviction: keep total under cap. Synchronous within the
    // write callback path. Evicted pieces are simply forgotten; subsequent
    // reads return file_too_short which libtorrent surfaces upstream. v1
    // will plumb clear_piece so libtorrent re-fetches automatically.
    if (m_used_bytes.load() > m_cap_bytes.load()) {
        enforce_cap();
    }

    lt::post(m_ios, [error, h = std::move(handler)]() { h(error); });
    return false;
}

void memory_disk_io::async_hash(lt::storage_index_t storage,
    lt::piece_index_t const piece,
    lt::span<lt::sha256_hash> block_hashes,
    lt::disk_job_flags_t flags,
    std::function<void(lt::piece_index_t, lt::sha1_hash const&,
                       lt::storage_error const&)> handler)
{
    lt::time_point const start = lt::clock_type::now();

    bool const v1 = bool(flags & lt::disk_interface::v1_hash);
    bool const v2 = !block_hashes.empty();

    lt::disk_buffer_holder buffer(m_buffer_allocator,
        new (std::nothrow) char[kDefaultBlockSize], kDefaultBlockSize);
    lt::storage_error error;
    if (!buffer) {
        error.ec = lt::errors::no_memory;
        error.operation = lt::operation_t::alloc_cache_piece;
        lt::post(m_ios, [piece, error, h = std::move(handler)]() {
            h(piece, lt::sha1_hash{}, error);
        });
        return;
    }

    memory_storage_t* st = nullptr;
    {
        std::lock_guard<std::mutex> lk(m_storages_mu);
        st = m_storages[storage].get();
    }
    if (!st) {
        error.ec = lt::errors::torrent_removed;
        error.operation = lt::operation_t::file_read;
        lt::post(m_ios, [piece, error, h = std::move(handler)]() {
            h(piece, lt::sha1_hash{}, error);
        });
        return;
    }

    lt::hasher ph;

    int const piece_size  = v1 ? st->files().piece_size(piece) : 0;
    int const piece_size2 = v2 ? st->files().piece_size2(piece) : 0;
    int const blocks_in_piece  = v1 ? (piece_size + kDefaultBlockSize - 1) / kDefaultBlockSize : 0;
    int const blocks_in_piece2 = v2 ? st->files().blocks_in_piece2(piece) : 0;

    int offset = 0;
    int const blocks_to_read = std::max(blocks_in_piece, blocks_in_piece2);
    for (int i = 0; i < blocks_to_read; ++i) {
        bool const v2_block = i < blocks_in_piece2;
        int const len  = v1 ? std::min(kDefaultBlockSize, piece_size  - offset) : 0;
        int const len2 = v2_block ? std::min(kDefaultBlockSize, piece_size2 - offset) : 0;
        lt::span<char> const b{buffer.data(),
                               static_cast<std::ptrdiff_t>(std::max(len, len2))};
        int const ret = st->read(b, piece, offset, error);
        offset += kDefaultBlockSize;
        if (ret <= 0) break;
        if (v1) ph.update(b.first(std::min(ret, len)));
        if (v2_block) block_hashes[i] = lt::hasher256(b.first(std::min(ret, len2))).final();
    }

    lt::sha1_hash const hash = v1 ? ph.final() : lt::sha1_hash();

    if (!error.ec) {
        int64_t const us = lt::total_microseconds(lt::clock_type::now() - start);
        m_stats_counters.inc_stats_counter(lt::counters::num_read_back, blocks_to_read);
        m_stats_counters.inc_stats_counter(lt::counters::num_blocks_read, blocks_to_read);
        m_stats_counters.inc_stats_counter(lt::counters::num_read_ops, blocks_to_read);
        m_stats_counters.inc_stats_counter(lt::counters::disk_hash_time, us);
        m_stats_counters.inc_stats_counter(lt::counters::disk_job_time, us);
    }

    lt::post(m_ios, [piece, hash, error, h = std::move(handler)]() {
        h(piece, hash, error);
    });
}

void memory_disk_io::async_hash2(lt::storage_index_t storage,
    lt::piece_index_t const piece, int offset, lt::disk_job_flags_t,
    std::function<void(lt::piece_index_t, lt::sha256_hash const&,
                       lt::storage_error const&)> handler)
{
    lt::time_point const start = lt::clock_type::now();

    lt::disk_buffer_holder buffer(m_buffer_allocator,
        new (std::nothrow) char[kDefaultBlockSize], kDefaultBlockSize);
    lt::storage_error error;
    if (!buffer) {
        error.ec = lt::errors::no_memory;
        error.operation = lt::operation_t::alloc_cache_piece;
        lt::post(m_ios, [piece, error, h = std::move(handler)]() {
            h(piece, lt::sha256_hash{}, error);
        });
        return;
    }

    memory_storage_t* st = nullptr;
    {
        std::lock_guard<std::mutex> lk(m_storages_mu);
        st = m_storages[storage].get();
    }
    if (!st) {
        error.ec = lt::errors::torrent_removed;
        error.operation = lt::operation_t::file_read;
        lt::post(m_ios, [piece, error, h = std::move(handler)]() {
            h(piece, lt::sha256_hash{}, error);
        });
        return;
    }

    int const piece_size = st->files().piece_size2(piece);
    std::ptrdiff_t const len = std::min(std::ptrdiff_t(kDefaultBlockSize),
                                         std::ptrdiff_t(piece_size - offset));
    lt::hasher256 ph;
    lt::span<char> const b{buffer.data(), len};
    int const ret = st->read(b, piece, offset, error);
    if (ret > 0) ph.update(b.first(ret));

    lt::sha256_hash const hash = ph.final();

    if (!error.ec) {
        int64_t const us = lt::total_microseconds(lt::clock_type::now() - start);
        m_stats_counters.inc_stats_counter(lt::counters::num_read_back);
        m_stats_counters.inc_stats_counter(lt::counters::num_blocks_read);
        m_stats_counters.inc_stats_counter(lt::counters::num_read_ops);
        m_stats_counters.inc_stats_counter(lt::counters::disk_hash_time, us);
        m_stats_counters.inc_stats_counter(lt::counters::disk_job_time, us);
    }

    lt::post(m_ios, [piece, hash, error, h = std::move(handler)]() {
        h(piece, hash, error);
    });
}

void memory_disk_io::async_move_storage(lt::storage_index_t /*storage*/,
    std::string p, lt::move_flags_t /*flags*/,
    std::function<void(lt::status_t, std::string const&,
                       lt::storage_error const&)> handler)
{
    // No physical storage. Pretend the move succeeded.
    lt::storage_error error;
    lt::post(m_ios, [p = std::move(p), error, h = std::move(handler)]() mutable {
        h(lt::status_t::no_error, p, error);
    });
}

void memory_disk_io::async_release_files(lt::storage_index_t /*storage*/,
    std::function<void()> handler)
{
    if (!handler) return;
    lt::post(m_ios, [h = std::move(handler)]() { h(); });
}

void memory_disk_io::async_check_files(lt::storage_index_t /*storage*/,
    lt::add_torrent_params const* /*resume_data*/,
    lt::aux::vector<std::string, lt::file_index_t> /*links*/,
    std::function<void(lt::status_t,
                       lt::storage_error const&)> handler)
{
    // Fresh in-memory storage holds nothing. Report no-error so libtorrent
    // proceeds to download from scratch without a recheck.
    lt::storage_error error;
    lt::post(m_ios, [error, h = std::move(handler)]() {
        h(lt::status_t::no_error, error);
    });
}

void memory_disk_io::async_stop_torrent(lt::storage_index_t /*storage*/,
    std::function<void()> handler)
{
    if (!handler) return;
    lt::post(m_ios, std::move(handler));
}

void memory_disk_io::async_rename_file(lt::storage_index_t /*storage*/,
    lt::file_index_t const idx, std::string name,
    std::function<void(std::string const&, lt::file_index_t,
                       lt::storage_error const&)> handler)
{
    lt::storage_error error;
    lt::post(m_ios, [idx, error, h = std::move(handler),
                     n = std::move(name)]() mutable {
        h(std::move(n), idx, error);
    });
}

void memory_disk_io::async_delete_files(lt::storage_index_t storage,
    lt::remove_flags_t /*options*/,
    std::function<void(lt::storage_error const&)> handler)
{
    {
        std::lock_guard<std::mutex> lk(m_storages_mu);
        if (storage >= m_storages.end_index()) {
            lt::storage_error error;
            error.ec = lt::errors::torrent_removed;
            error.operation = lt::operation_t::file_remove;
            lt::post(m_ios, [error, h = std::move(handler)]() { h(error); });
            return;
        }
        if (m_storages[storage]) m_storages[storage]->clear_all();
    }
    lt::storage_error error;
    lt::post(m_ios, [error, h = std::move(handler)]() { h(error); });
}

void memory_disk_io::async_set_file_priority(lt::storage_index_t /*storage*/,
    lt::aux::vector<lt::download_priority_t, lt::file_index_t> prio,
    std::function<void(lt::storage_error const&,
        lt::aux::vector<lt::download_priority_t,
                         lt::file_index_t>)> handler)
{
    lt::storage_error error;
    lt::post(m_ios, [p = std::move(prio), error,
                     h = std::move(handler)]() mutable {
        h(error, std::move(p));
    });
}

void memory_disk_io::async_clear_piece(lt::storage_index_t storage,
    lt::piece_index_t index,
    std::function<void(lt::piece_index_t)> handler)
{
    {
        std::lock_guard<std::mutex> lk(m_storages_mu);
        if (storage < m_storages.end_index() && m_storages[storage]) {
            m_storages[storage]->clear_piece(index);
        }
    }
    lt::post(m_ios, [index, h = std::move(handler)]() { h(index); });
}

void memory_disk_io::update_stats_counters(lt::counters&) const {}

std::vector<lt::open_file_state>
memory_disk_io::get_status(lt::storage_index_t) const { return {}; }

void memory_disk_io::submit_jobs() {}

memory_stats_t memory_disk_io::stats() const {
    memory_stats_t s{};
    s.used_bytes = m_used_bytes.load();
    s.cap_bytes  = m_cap_bytes.load();
    s.evicted_count = m_evicted_count.load();
    int64_t pc = 0;
    {
        std::lock_guard<std::mutex> lk(m_storages_mu);
        for (auto& up : m_storages) {
            if (up) pc += up->piece_count();
        }
    }
    s.piece_count = pc;
    return s;
}

int64_t memory_disk_io::enforce_cap() {
    int64_t const cap = m_cap_bytes.load();
    if (m_used_bytes.load() <= cap) return 0;

    std::vector<memory_storage_t*> sts;
    {
        std::lock_guard<std::mutex> lk(m_storages_mu);
        for (auto& up : m_storages) if (up) sts.push_back(up.get());
    }
    int64_t total_evicted = 0;
    for (auto* st : sts) {
        if (m_used_bytes.load() <= cap) break;
        auto ev = st->evict_lru_to_cap(cap);
        if (!ev.empty()) {
            // Queue clear_piece notifications so libtorrent re-fetches on
            // demand instead of returning stale "complete" bitfield bits.
            lt::sha1_hash const& ih = st->info_hash();
            for (auto p : ev) push_eviction(ih, p);
        }
        total_evicted += static_cast<int64_t>(ev.size());
    }
    if (total_evicted) m_evicted_count.fetch_add(total_evicted);
    return total_evicted;
}

bool memory_disk_io::set_head_for_info_hash(lt::sha1_hash const& ih,
                                            int64_t byte_offset)
{
    std::lock_guard<std::mutex> lk(m_storages_mu);
    for (auto& up : m_storages) {
        if (!up) continue;
        if (up->info_hash() == ih) {
            up->set_head(byte_offset);
            return true;
        }
    }
    return false;
}

bool memory_disk_io::set_window_for_info_hash(lt::sha1_hash const& ih,
                                              int64_t rewind_bytes,
                                              int64_t prefetch_bytes)
{
    std::lock_guard<std::mutex> lk(m_storages_mu);
    for (auto& up : m_storages) {
        if (!up) continue;
        if (up->info_hash() == ih) {
            up->set_window(rewind_bytes, prefetch_bytes);
            return true;
        }
    }
    return false;
}

void memory_disk_io::push_eviction(lt::sha1_hash const& ih,
                                   lt::piece_index_t p)
{
    std::lock_guard<std::mutex> lk(m_evict_mu);
    m_evict_queue.push_back({ih, p});
}

std::vector<memory_disk_io::evicted_piece>
memory_disk_io::drain_eviction_queue()
{
    std::vector<evicted_piece> out;
    std::lock_guard<std::mutex> lk(m_evict_mu);
    out.swap(m_evict_queue);
    return out;
}

std::unique_ptr<lt::disk_interface> memory_disk_io_constructor(
    lt::io_context& ios,
    lt::settings_interface const& sett,
    lt::counters& cnt)
{
    return std::make_unique<memory_disk_io>(ios, sett, cnt);
}

} // namespace tb
