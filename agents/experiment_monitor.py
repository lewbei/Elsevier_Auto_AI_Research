"""Lightweight experiment monitoring utilities.

Provides a minimal ExperimentMonitor that can be extended later.
Non-invasive: all methods are safe no-ops on exception.

Design goals:
- Zero external deps (pure stdlib)
- JSONL event log + summary JSON
- Basic anomaly detection (NaN metrics, stagnation, divergence)
- Hooks for resource sampling (placeholder: future GPU/CPU/mem integration)

Usage pattern (from iterate loop):
    from agents.experiment_monitor import get_monitor
    mon = get_monitor()
    run_id = mon.start(plan_context={...}, specs=[...])
    mon.update(run_id, name, metrics={'val_accuracy': 0.5}, step=i)
    mon.finalize(run_id)

Config flags (checked via lab.config.get_bool):
  feedback.enable (master)
  feedback.monitor.enable
If neither exists, defaults to disabled.
"""
from __future__ import annotations

import json, time, math, os, threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

try:
    from lab.config import get_bool
except Exception:  # fallback if lab.config not ready
    def get_bool(key: str, default: bool=False) -> bool:  # type: ignore
        return bool(default)

RUNS_DIR = Path("runs")
MONITOR_DIR = RUNS_DIR / "monitor"

@dataclass
class MonitorRules:
    plateau_patience: int = 4           # intervals with no improvement
    min_improve: float = 1e-4           # min delta to count as improvement
    divergence_ratio: float = 0.5       # drop > 50% from best
    max_wall_time_sec: float = 0.0      # 0 = disabled
    check_interval: float = 30.0        # seconds between background checks


class ExperimentMonitor:
    def __init__(self, rules: Optional[MonitorRules] = None) -> None:
        self.enabled = self._enabled()
        self.events: List[Dict[str, Any]] = []            # lifecycle + high-level updates
        self.progress_events: List[Dict[str, Any]] = []   # fine‑grained step/epoch metrics
        self.metrics_history: List[Dict[str, Any]] = []   # condensed ordered metric snapshots
        self.decisions: List[Dict[str, Any]] = []         # decision records (trend-based)
        self._pending_action: Optional[Dict[str, Any]] = None
        self.run_start_ts: Optional[float] = None
        self.last_metric: Dict[str, float] = {}
        self.best_acc: Optional[float] = None
        self.run_id: Optional[str] = None
        self.rules = rules or MonitorRules()
        self._bg_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._request_stop = False
        self._stop_reason: Optional[str] = None
        self._last_improve_ts: Optional[float] = None
        MONITOR_DIR.mkdir(parents=True, exist_ok=True)

    def _enabled(self) -> bool:
        try:
            return bool(get_bool("feedback.monitor.enable", get_bool("feedback.enable", False)))
        except Exception:
            return False

    # ---- Public API ----
    def start(self, run_id: str, plan_context: Optional[Dict[str, Any]]=None, specs: Optional[List[Dict[str, Any]]]=None) -> str:
        if not self.enabled:
            return run_id
        self.run_id = run_id
        self.run_start_ts = time.time()
        self._log({
            'event': 'start',
            'run_id': run_id,
            'plan_context': plan_context or {},
            'spec_count': len(specs or []),
            'ts': self._now()
        })
        # Launch background thread for rule evaluation
        if self._bg_thread is None:
            self._bg_thread = threading.Thread(target=self._monitor_loop, name=f"monitor-{run_id}", daemon=True)
            try:
                self._bg_thread.start()
            except Exception:
                self._bg_thread = None
        return run_id

    def update(self, run_id: str, name: str, metrics: Dict[str, Any], step: int | None=None) -> None:
        if not self.enabled:
            return
        mflat = self._flatten_metrics(metrics)
        anomalies = self._detect_anomalies(mflat)
        self._log({
            'event': 'update',
            'run_id': run_id,
            'name': name,
            'step': step,
            'metrics': mflat,
            'anomalies': anomalies,
            'ts': self._now()
        })
        # store last metric for divergence checks
        for k,v in mflat.items():
            if isinstance(v, (int,float)) and not math.isnan(float(v)):
                self.last_metric[k] = float(v)

    def finalize(self, run_id: str) -> None:
        if not self.enabled:
            return
        elapsed = None
        if self.run_start_ts is not None:
            elapsed = time.time() - self.run_start_ts
        # Signal background thread to stop
        self._stop_event.set()
        try:
            if self._bg_thread and self._bg_thread.is_alive():
                self._bg_thread.join(timeout=2.0)
        except Exception:
            pass
        summary = self._summarize(elapsed)
        try:
            (MONITOR_DIR / f"{run_id}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass
        # also write a consolidated log
        try:
            log_path = MONITOR_DIR / f"{run_id}.jsonl"
            with log_path.open('w', encoding='utf-8') as fh:
                for e in self.events:
                    fh.write(json.dumps(e, ensure_ascii=False) + '\n')
        except Exception:
            pass

    # ---- Internals ----
    def _log(self, rec: Dict[str, Any]) -> None:
        rec['t_rel'] = None
        if self.run_start_ts is not None:
            rec['t_rel'] = time.time() - self.run_start_ts
        self.events.append(rec)

    # ---- Real-time progress ingestion (called from training heartbeat) ----
    def progress(self, run_id: str, name: str, step: int, metrics: Dict[str, Any]) -> None:
        if not self.enabled or self.run_id != run_id:
            return
        try:
            val_acc = None
            # Accept either flat or nested metrics
            if 'val_accuracy' in metrics:
                val_acc = float(metrics.get('val_accuracy'))
            elif 'metrics' in metrics and isinstance(metrics['metrics'], dict):
                if 'val_accuracy' in metrics['metrics']:
                    val_acc = float(metrics['metrics']['val_accuracy'])
            evt = {
                'event': 'progress', 'run_id': run_id, 'name': name,
                'step': step, 'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))}, 'ts': self._now()
            }
            self.progress_events.append(evt)
            # Maintain condensed metrics history (val_accuracy prioritized)
            snap = {'step': step, 'ts': evt['ts']}
            if 'val_accuracy' in evt['metrics']:
                snap['val_accuracy'] = evt['metrics']['val_accuracy']
            if 'train_loss' in evt['metrics']:
                snap['train_loss'] = evt['metrics']['train_loss']
            self.metrics_history.append(snap)
            # Trigger adaptive analysis opportunistically
            if step % max(1, int(os.getenv('MONITOR_ANALYZE_INTERVAL', '25') or 25)) == 0:
                try:
                    self._analyze_trends()
                except Exception:
                    pass
            if val_acc is not None and not math.isnan(val_acc):
                if self.best_acc is None or val_acc > self.best_acc + self.rules.min_improve:
                    self.best_acc = val_acc
                    self._last_improve_ts = time.time()
        except Exception:
            pass

    # ---- Early stop query exposed to training loop ----
    def should_stop(self, run_id: str) -> bool:
        if not self.enabled or self.run_id != run_id:
            return False
        return self._request_stop

    def stop_reason(self) -> Optional[str]:
        return self._stop_reason

    # ---- Background monitoring loop ----
    def _monitor_loop(self) -> None:
        interval = max(1.0, float(self.rules.check_interval))
        while not self._stop_event.wait(interval):
            try:
                self._evaluate_rules()
            except Exception:
                continue

    def _evaluate_rules(self) -> None:
        if self._request_stop:
            return
        now = time.time()
        # Always attempt analytic trend pass (lightweight)
        try:
            self._analyze_trends()
        except Exception:
            pass
        # Plateau detection
        if self.rules.plateau_patience > 0 and self.best_acc is not None:
            if self._last_improve_ts is None:
                self._last_improve_ts = self.run_start_ts or now
            since_improve = now - self._last_improve_ts
            intervals = since_improve / max(1e-6, self.rules.check_interval)
            if intervals >= self.rules.plateau_patience:
                self._trigger_stop('plateau')
                return
        # Wall time budget
        if self.rules.max_wall_time_sec > 0 and self.run_start_ts is not None:
            if (now - self.run_start_ts) > self.rules.max_wall_time_sec:
                self._trigger_stop('time_budget')
                return
        # Divergence detection
        if self.best_acc is not None and self.best_acc > 0:
            # look at latest val_acc in progress events
            for ev in reversed(self.progress_events[-5:]):
                try:
                    cur = ev.get('metrics', {}).get('val_accuracy')
                    if isinstance(cur, (int, float)) and cur < (1.0 - self.rules.divergence_ratio) * self.best_acc:
                        self._trigger_stop('divergence')
                        return
                except Exception:
                    continue

    def _trigger_stop(self, reason: str) -> None:
        self._request_stop = True
        self._stop_reason = reason
        self._log({'event': 'early_stop', 'run_id': self.run_id, 'reason': reason, 'ts': self._now()})
        try:
            # sentinel file
            if self.run_id:
                (MONITOR_DIR / f"{self.run_id}_EARLY_STOP").write_text(reason, encoding='utf-8')
        except Exception:
            pass

    # ---- Trend Analysis & Adaptive Decision Logic ----
    def _analyze_trends(self) -> None:
        if not self.metrics_history:
            return
        # Minimum points to analyze
        if len(self.metrics_history) < 5:
            return
        recent = self.metrics_history[-20:]
        val_points = [m for m in recent if 'val_accuracy' in m]
        loss_points = [m for m in recent if 'train_loss' in m]
        trend: Dict[str, Any] = {'ts': self._now(), 'n': len(recent)}
        def _slope(seq, key):
            if len(seq) < 3:
                return 0.0
            try:
                # simple least squares slope
                n = len(seq)
                xs = list(range(n))
                mean_x = sum(xs)/n
                ys = [float(x.get(key, 0.0)) for x in seq]
                mean_y = sum(ys)/n
                num = sum((xs[i]-mean_x)*(ys[i]-mean_y) for i in range(n))
                den = sum((xs[i]-mean_x)**2 for i in range(n)) + 1e-9
                return num/den
            except Exception:
                return 0.0
        if val_points:
            trend['val_acc_slope'] = _slope(val_points, 'val_accuracy')
            trend['val_acc_last'] = float(val_points[-1]['val_accuracy'])
            trend['val_acc_best'] = float(max(p['val_accuracy'] for p in val_points))
        if loss_points:
            trend['train_loss_slope'] = _slope(loss_points, 'train_loss')
            trend['train_loss_last'] = float(loss_points[-1]['train_loss'])
        # Derive heuristic signals
        signals: List[str] = []
        slope = trend.get('val_acc_slope', 0.0)
        if 'val_acc_slope' in trend and abs(slope) < 1e-4:
            signals.append('plateau_val')
        if slope < -5e-4:
            signals.append('decline_val')
        lslope = trend.get('train_loss_slope', 0.0)
        if lslope > -1e-4 and lslope < 1e-4:
            signals.append('plateau_loss')
        if lslope < -1e-3 and slope <= 0:
            signals.append('potential_overfit')  # loss improving, val not
        trend['signals'] = signals
        action = self._decide_action(trend)
        if action:
            rec = {'event': 'decision', 'run_id': self.run_id, 'ts': self._now(), 'trend': trend, 'action': action}
            self.decisions.append(rec)
            self._pending_action = action
            self._log(rec)
            # Persist incremental decisions (append)
            try:
                with open(MONITOR_DIR / f"{self.run_id}_decisions.jsonl", 'a', encoding='utf-8') as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            except Exception:
                pass

    def _decide_action(self, trend: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        signals = trend.get('signals', [])
        if not signals:
            return None
        # Priority ordering
        if 'decline_val' in signals:
            return {'type': 'adjust_lr', 'factor': 0.5, 'reason': 'validation declining'}
        if 'potential_overfit' in signals:
            return {'type': 'regularize', 'dropout_delta': 0.1, 'reason': 'loss improving without val gains'}
        if 'plateau_val' in signals and 'plateau_loss' in signals:
            return {'type': 'increase_capacity', 'reason': 'both metrics plateau'}
        if 'plateau_val' in signals:
            return {'type': 'explore_hparam', 'param': 'lr', 'strategy': 'increase_small', 'reason': 'val plateau'}
        return None

    # Public adaptive API
    def should_adapt_plan(self) -> bool:
        return self._pending_action is not None

    def next_action(self) -> Optional[Dict[str, Any]]:
        return self._pending_action

    def clear_action(self) -> None:
        self._pending_action = None

    @staticmethod
    def _now() -> str:
        return time.strftime('%Y%m%d-%H%M%S', time.localtime())

    @staticmethod
    def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not isinstance(metrics, dict):
            return out
        for k,v in metrics.items():
            if isinstance(v,(int,float)):
                try:
                    f = float(v)
                    out[k] = f
                except Exception:
                    continue
            elif isinstance(v, dict):
                # one-level flatten
                for k2,v2 in v.items():
                    if isinstance(v2,(int,float)):
                        try:
                            out[f"{k}.{k2}"] = float(v2)
                        except Exception:
                            pass
        return out

    def _detect_anomalies(self, metrics: Dict[str,float]) -> List[str]:
        anomalies: List[str] = []
        for k,v in metrics.items():
            if math.isnan(v) or math.isinf(v):
                anomalies.append(f"{k}=NaN/inf")
            if k in self.last_metric:
                prev = self.last_metric[k]
                # crude divergence: drop more than 50% for accuracy-like metrics
                if prev > 0 and (prev - v)/max(1e-9, prev) > 0.5 and 'acc' in k:
                    anomalies.append(f"{k}_divergence prev={prev:.4f} now={v:.4f}")
        # Placeholder resource sampling (future integration)
        if os.getenv('MONITOR_FAKE_RESOURCE','0') == '1':
            anomalies.append('resource_sampling_disabled')
        return anomalies

    def _summarize(self, elapsed: Optional[float]) -> Dict[str, Any]:
        # compute final metrics snapshot (last seen per key)
        final_metrics = {k:v for k,v in self.last_metric.items()}
        total_updates = len([e for e in self.events if e.get('event')=='update'])
        anomalies = []
        for e in self.events:
            anomalies.extend(e.get('anomalies',[]))
        return {
            'run_id': self.run_id,
            'elapsed_sec': elapsed,
            'total_events': len(self.events),
            'updates': total_updates,
            'final_metrics': final_metrics,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies[:50],  # cap
            'early_stop': self._request_stop,
            'early_stop_reason': self._stop_reason,
            'decisions': len(self.decisions),
        }


_singleton: ExperimentMonitor | None = None

def get_monitor() -> ExperimentMonitor:
    global _singleton
    if _singleton is None:
        # Allow rule overrides via env (simple)
        try:
            plateau = int(os.getenv('MONITOR_PLATEAU_PATIENCE', '4') or 4)
            max_time = float(os.getenv('MONITOR_MAX_WALL', '0') or 0)
            rules = MonitorRules(plateau_patience=plateau, max_wall_time_sec=max_time)
        except Exception:
            rules = MonitorRules()
        _singleton = ExperimentMonitor(rules)
    return _singleton

__all__ = ['ExperimentMonitor','get_monitor','MonitorRules']
