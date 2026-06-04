// Periodic system-RAM probe → memory_disk_io cap.
//
// Runs alongside the engine, polls free RAM every [interval], sets
// memory cap = clamp(avail * [memoryFraction], [minCap], [maxCap]).
// Hysteresis: skip writes unless |new - old| > [hysteresisBytes].

import 'dart:async';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'ffi_bindings.dart';

class MemoryBudgetController {
  final TorrentBridgeBindings _b;

  /// Fraction of currently-available RAM the cap targets. Default 60%.
  /// The remainder stays as headroom for foreground apps and the OS.
  final double memoryFraction;

  /// Floor cap. Default 128 MB. Below this, eviction churns the playback
  /// window and stalls. Hard minimum.
  final int minCap;

  /// Optional ceiling. Null means no ceiling — use whatever the device
  /// has free × [memoryFraction]. Set explicitly only if you want to
  /// reserve more for the rest of the app.
  final int? maxCap;

  /// Skip updates when the new cap differs from the current by less
  /// than this. Default 32 MB. Avoids cap thrash.
  final int hysteresisBytes;

  /// Poll interval. Default 5 s.
  final Duration interval;

  Timer? _timer;
  int _lastCap = 0;

  /// Last sample observed. Useful for diagnostics.
  int lastTotal = 0;
  int lastAvail = 0;

  MemoryBudgetController(
    this._b, {
    this.memoryFraction = 0.60,
    this.minCap = 128 * 1024 * 1024,
    this.maxCap,
    this.hysteresisBytes = 32 * 1024 * 1024,
    this.interval = const Duration(seconds: 5),
  });

  /// Start polling. Applies one cap immediately, then on every [interval].
  void start() {
    _tick();
    _timer ??= Timer.periodic(interval, (_) => _tick());
  }

  /// Stop polling. Does not change the current cap.
  void stop() {
    _timer?.cancel();
    _timer = null;
  }

  /// Force a one-shot probe + cap update (ignores hysteresis).
  void forceUpdate() {
    final cap = _computeCap();
    if (cap <= 0) return;
    _b.setMemoryCap(cap);
    _lastCap = cap;
  }

  /// Drop cap to floor immediately. Use when the OS reports memory pressure
  /// (e.g. from a foreign onTrimMemory bridge if one is later added).
  void enterLowMemoryMode() {
    _b.setMemoryCap(minCap);
    _lastCap = minCap;
  }

  void _tick() {
    final cap = _computeCap();
    if (cap <= 0) return;
    if ((_lastCap - cap).abs() < hysteresisBytes) return;
    _b.setMemoryCap(cap);
    _lastCap = cap;
  }

  int _computeCap() {
    final total = calloc<Int64>();
    final avail = calloc<Int64>();
    try {
      final rc = _b.getSystemMemory(total, avail);
      if (rc != 0) return 0;
      lastTotal = total.value;
      lastAvail = avail.value;
      final target = (avail.value * memoryFraction).toInt();
      if (target < minCap) return minCap;
      final ceiling = maxCap;
      if (ceiling != null && target > ceiling) return ceiling;
      return target;
    } finally {
      calloc.free(total);
      calloc.free(avail);
    }
  }
}
