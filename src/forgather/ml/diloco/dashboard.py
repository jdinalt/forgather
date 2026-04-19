"""
DiLoCo Server Dashboard.

Self-contained HTML dashboard served by the DiLoCo parameter server.
Uses Alpine.js (CDN) for reactivity. No build step or static file directory.
"""

from http.server import BaseHTTPRequestHandler


DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiLoCo Dashboard</title>
<style>
:root {
    --bg: #1a1b26;
    --bg-surface: #24283b;
    --bg-hover: #292e42;
    --border: #3b4261;
    --text: #c0caf5;
    --text-dim: #565f89;
    --text-bright: #e0e6ff;
    --accent: #7aa2f7;
    --green: #9ece6a;
    --yellow: #e0af68;
    --red: #f7768e;
    --orange: #ff9e64;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', 'Cascadia Code', monospace;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    line-height: 1.5;
}
.container { max-width: 1200px; margin: 0 auto; padding: 16px; }

/* Header */
.header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 16px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.header h1 { font-size: 16px; color: var(--text-bright); font-weight: 600; }
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}
.badge-sync { background: #1a3a5c; color: var(--accent); }
.badge-async { background: #3a2a1a; color: var(--orange); }
.header-stat { color: var(--text-dim); }
.header-stat b { color: var(--text); }
.header-right { margin-left: auto; display: flex; align-items: center; gap: 8px; }
.header-right select {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 6px;
    font-family: inherit;
    font-size: 12px;
}

/* Error banner */
.error-banner {
    background: #2d1520;
    border: 1px solid var(--red);
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 16px;
    color: var(--red);
}

/* Grid layout */
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-full { grid-column: 1 / -1; }

/* Panels */
.panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
}
.panel-title {
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
}
.panel-body { padding: 12px 16px; }

/* Worker table */
table { width: 100%; border-collapse: collapse; }
th {
    text-align: left;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
}
td { padding: 6px 10px; border-bottom: 1px solid var(--border); }
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--bg-hover); }

/* Status dots */
.dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-green { background: var(--green); }
.dot-yellow { background: var(--yellow); }
.dot-red { background: var(--red); }

/* Metrics */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}
.metric {
    padding: 6px 0;
}
.metric-label { font-size: 11px; color: var(--text-dim); }
.metric-value { font-size: 14px; color: var(--text-bright); font-weight: 600; }

/* Progress bar */
.progress-bar {
    height: 6px;
    background: var(--bg);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 4px;
}
.progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.3s ease;
}

/* Controls */
.control-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.control-card {
    padding: 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
}
.control-card h4 {
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.control-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.control-row:last-child { margin-bottom: 0; }
.control-row label {
    font-size: 12px;
    color: var(--text-dim);
    min-width: 70px;
}
input[type="number"], input[type="text"] {
    background: var(--bg-surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 8px;
    font-family: inherit;
    font-size: 12px;
    width: 90px;
}
input:focus { outline: none; border-color: var(--accent); }
button {
    background: var(--bg-surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 12px;
    font-family: inherit;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
}
button:hover { background: var(--bg-hover); border-color: var(--accent); color: var(--text-bright); }
button:active { transform: scale(0.97); }
.btn-danger { border-color: var(--red); color: var(--red); }
.btn-danger:hover { background: #2d1520; border-color: var(--red); color: var(--red); }
.btn-primary { border-color: var(--accent); color: var(--accent); }
.btn-kick { padding: 2px 8px; font-size: 11px; }

/* Control feedback message */
.control-msg {
    padding: 6px 12px;
    margin-bottom: 12px;
    border-radius: 4px;
    font-size: 12px;
}
.control-msg-ok { background: #1a2e1a; color: var(--green); border: 1px solid var(--green); }
.control-msg-err { background: #2d1520; color: var(--red); border: 1px solid var(--red); }

/* Empty state */
.empty-state { padding: 20px; text-align: center; color: var(--text-dim); }

/* Confirm overlay */
.confirm-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
}
.confirm-box {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
    max-width: 400px;
    text-align: center;
}
.confirm-box p { margin-bottom: 16px; color: var(--text-bright); }
.confirm-box .btn-row { display: flex; gap: 12px; justify-content: center; }
</style>
</head>
<body>
<div class="container" x-data="dashboard" x-init="init()">

    <!-- Error banner -->
    <template x-if="error">
        <div class="error-banner" x-text="error"></div>
    </template>

    <!-- Header -->
    <div class="header">
        <h1>DiLoCo</h1>
        <template x-if="status">
            <span class="badge"
                  :class="status.mode === 'async' ? 'badge-async' : 'badge-sync'"
                  x-text="status.mode"></span>
        </template>
        <template x-if="status">
            <span class="header-stat">Round <b x-text="status.sync_round"></b></span>
        </template>
        <template x-if="status">
            <span class="header-stat" x-text="uptime"></span>
        </template>
        <template x-if="status && status.model_params">
            <span class="header-stat"
                  x-text="formatParams(status.model_params) + ' (' + status.model_size_mb + ' MB)'"></span>
        </template>
        <div class="header-right">
            <label class="header-stat" for="refresh-sel">Refresh</label>
            <select id="refresh-sel" x-model.number="refreshMs" @change="resetInterval()">
                <option value="1000">1s</option>
                <option value="2000">2s</option>
                <option value="5000">5s</option>
                <option value="10000">10s</option>
                <option value="30000">30s</option>
            </select>
        </div>
    </div>

    <!-- Control feedback -->
    <template x-if="controlMsg">
        <div class="control-msg"
             :class="controlMsg.ok ? 'control-msg-ok' : 'control-msg-err'"
             x-text="controlMsg.text"></div>
    </template>

    <div class="grid">

        <!-- Worker table -->
        <div class="panel grid-full">
            <div class="panel-title">Workers
                <template x-if="status">
                    <span x-text="'(' + status.num_registered + '/' + status.num_workers + ')'"></span>
                </template>
            </div>
            <template x-if="status && workerList.length > 0">
                <table>
                    <thead>
                        <tr>
                            <th></th>
                            <th>ID</th>
                            <th>Hostname</th>
                            <th>Round</th>
                            <th>Steps/s</th>
                            <th>Last Heartbeat</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody>
                        <template x-for="w in workerList" :key="w.id">
                            <tr>
                                <td><span class="dot" :class="workerHealthClass(w)"></span></td>
                                <td :title="w.id" x-text="truncId(w.id)"></td>
                                <td x-text="w.hostname"></td>
                                <td x-text="w.sync_round"></td>
                                <td x-text="w.steps_per_second > 0 ? w.steps_per_second.toFixed(2) : '-'"></td>
                                <td x-text="relativeTime(w.last_heartbeat)"></td>
                                <td>
                                    <button class="btn-kick btn-danger"
                                            @click="kickWorker(w.id)">Kick</button>
                                </td>
                            </tr>
                        </template>
                    </tbody>
                </table>
            </template>
            <template x-if="!status || workerList.length === 0">
                <div class="empty-state">No workers connected</div>
            </template>
        </div>

        <!-- Server metrics -->
        <div class="panel">
            <div class="panel-title">Server Metrics</div>
            <div class="panel-body">
                <template x-if="status">
                    <div>
                        <div class="metrics-grid">
                            <div class="metric">
                                <div class="metric-label">Outer LR</div>
                                <div class="metric-value" x-text="status.outer_lr"></div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Outer Momentum</div>
                                <div class="metric-value" x-text="status.outer_momentum"></div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Worker Deaths</div>
                                <div class="metric-value" x-text="status.total_worker_deaths || 0"></div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">HB Timeout</div>
                                <div class="metric-value" x-text="status.heartbeat_timeout + 's'"></div>
                            </div>
                        </div>

                        <!-- Pending submissions (sync mode) -->
                        <template x-if="status.mode === 'sync' && status.pending_submissions">
                            <div class="metric" style="margin-top: 8px;">
                                <div class="metric-label">
                                    Pending Submissions
                                    (<span x-text="status.pending_submissions.length"></span>/<span x-text="status.num_workers"></span>)
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill"
                                         :style="'width:' + (status.pending_submissions.length / Math.max(status.num_workers, 1) * 100) + '%'"></div>
                                </div>
                            </div>
                        </template>

                        <!-- Async metrics -->
                        <template x-if="status.mode === 'async'">
                            <div class="metrics-grid" style="margin-top: 8px;">
                                <div class="metric">
                                    <div class="metric-label">Total Submissions</div>
                                    <div class="metric-value" x-text="status.total_submissions || 0"></div>
                                </div>
                                <div class="metric">
                                    <div class="metric-label">DN Buffer</div>
                                    <div class="metric-value"
                                         x-text="status.dn_buffer_size > 0 ? (status.dn_buffered + '/' + status.dn_buffer_size) : 'off'"></div>
                                </div>
                                <div class="metric">
                                    <div class="metric-label">DyLU</div>
                                    <div class="metric-value"
                                         x-text="status.dylu_enabled ? ('on (H=' + status.dylu_base_sync_every + ')') : 'off'"></div>
                                </div>
                            </div>
                        </template>

                        <!-- Fragment submissions -->
                        <template x-if="status.fragment_submissions">
                            <div class="metric" style="margin-top: 8px;">
                                <div class="metric-label">Fragment Submissions</div>
                                <div class="metric-value" x-text="status.fragment_submissions"></div>
                            </div>
                        </template>
                    </div>
                </template>
            </div>
        </div>

        <!-- Control panel -->
        <div class="panel">
            <div class="panel-title">Control</div>
            <div class="panel-body">
                <div class="control-grid">
                    <div class="control-card">
                        <h4>Save State</h4>
                        <button class="btn-primary"
                                :disabled="!status || !status.save_dir"
                                @click="saveState()">
                            Save Checkpoint
                        </button>
                        <template x-if="status && !status.save_dir">
                            <div style="margin-top:4px;font-size:11px;color:var(--text-dim)">No save_dir configured</div>
                        </template>
                    </div>
                    <div class="control-card">
                        <h4>Shutdown</h4>
                        <button class="btn-danger" @click="confirmShutdown = true">Shutdown Server</button>
                    </div>
                    <div class="control-card">
                        <h4>Optimizer</h4>
                        <div class="control-row">
                            <label>LR</label>
                            <input type="number" step="any" x-model.number="formLr">
                        </div>
                        <div class="control-row">
                            <label>Momentum</label>
                            <input type="number" step="any" x-model.number="formMomentum">
                        </div>
                        <div class="control-row">
                            <button class="btn-primary" @click="updateOptimizer()">Apply</button>
                        </div>
                    </div>
                    <div class="control-card">
                        <h4>Workers</h4>
                        <div class="control-row">
                            <label>Expected</label>
                            <input type="number" min="1" x-model.number="formNumWorkers">
                        </div>
                        <div class="control-row">
                            <button class="btn-primary" @click="updateNumWorkers()">Apply</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </div><!-- /grid -->

    <!-- Shutdown confirm -->
    <template x-if="confirmShutdown">
        <div class="confirm-overlay" @click.self="confirmShutdown = false">
            <div class="confirm-box">
                <p>Shut down the DiLoCo server? This will stop all training.</p>
                <div class="btn-row">
                    <button @click="confirmShutdown = false">Cancel</button>
                    <button class="btn-danger" @click="shutdown()">Confirm Shutdown</button>
                </div>
            </div>
        </div>
    </template>

</div>

<script src="https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js" defer></script>
<script>
document.addEventListener('alpine:init', () => {
    Alpine.data('dashboard', () => ({
        status: null,
        error: null,
        refreshMs: 2000,
        controlMsg: null,
        confirmShutdown: false,
        _interval: null,

        // Form state
        formLr: 0.7,
        formMomentum: 0.9,
        formNumWorkers: 1,

        init() {
            this.fetchStatus();
            this.resetInterval();
        },

        resetInterval() {
            if (this._interval) clearInterval(this._interval);
            this._interval = setInterval(() => this.fetchStatus(), this.refreshMs);
        },

        async fetchStatus() {
            try {
                const r = await fetch('/status');
                if (!r.ok) throw new Error('HTTP ' + r.status);
                this.status = await r.json();
                this.error = null;
                // Sync form values on first load
                if (this.formLr === 0.7 && this.status.outer_lr !== undefined) {
                    this.formLr = this.status.outer_lr;
                }
                if (this.formMomentum === 0.9 && this.status.outer_momentum !== undefined) {
                    this.formMomentum = this.status.outer_momentum;
                }
                if (this.formNumWorkers === 1 && this.status.num_workers !== undefined) {
                    this.formNumWorkers = this.status.num_workers;
                }
            } catch (e) {
                this.error = 'Connection error: ' + e.message;
            }
        },

        async postControl(action, body) {
            try {
                const r = await fetch('/control/' + action, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body || {})
                });
                const data = await r.json();
                if (!r.ok || data.error) {
                    this.showMsg(data.error || 'Request failed', false);
                    return null;
                }
                this.showMsg(action.replace(/_/g, ' ') + ': OK', true);
                this.fetchStatus();
                return data;
            } catch (e) {
                this.showMsg('Error: ' + e.message, false);
                return null;
            }
        },

        showMsg(text, ok) {
            this.controlMsg = {text, ok};
            setTimeout(() => { this.controlMsg = null; }, 4000);
        },

        async saveState() { await this.postControl('save_state'); },
        async kickWorker(id) { await this.postControl('kick_worker', {worker_id: id}); },
        async updateOptimizer() {
            await this.postControl('update_optimizer', {
                lr: this.formLr,
                momentum: this.formMomentum
            });
        },
        async updateNumWorkers() {
            await this.postControl('update_num_workers', {num_workers: this.formNumWorkers});
        },
        async shutdown() {
            this.confirmShutdown = false;
            await this.postControl('shutdown');
        },

        // Computed
        get uptime() {
            if (!this.status || !this.status.uptime_seconds) return '';
            const s = Math.floor(this.status.uptime_seconds);
            const h = Math.floor(s / 3600);
            const m = Math.floor((s % 3600) / 60);
            const sec = s % 60;
            if (h > 0) return h + 'h ' + m + 'm';
            if (m > 0) return m + 'm ' + sec + 's';
            return sec + 's';
        },

        get workerList() {
            if (!this.status || !this.status.workers) return [];
            return Object.entries(this.status.workers).map(([id, w]) => ({id, ...w}));
        },

        truncId(id) {
            return id.length > 20 ? id.substring(0, 17) + '...' : id;
        },

        formatParams(n) {
            if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
            if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
            if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
            return n.toString();
        },

        relativeTime(ts) {
            if (!ts) return '-';
            const ago = Math.floor(Date.now() / 1000 - ts);
            if (ago < 0) return 'just now';
            if (ago < 60) return ago + 's ago';
            if (ago < 3600) return Math.floor(ago / 60) + 'm ago';
            return Math.floor(ago / 3600) + 'h ago';
        },

        workerHealthClass(w) {
            if (!w.last_heartbeat) return 'dot-red';
            const ago = Date.now() / 1000 - w.last_heartbeat;
            if (ago < 60) return 'dot-green';
            if (ago < 120) return 'dot-yellow';
            return 'dot-red';
        }
    }));
});
</script>
</body>
</html>
"""


def send_dashboard_response(handler: BaseHTTPRequestHandler):
    """Send the dashboard HTML page."""
    body = DASHBOARD_HTML.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
