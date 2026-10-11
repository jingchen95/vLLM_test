#!/bin/bash
# Apply the two accepted vLLM 0.15.1 P2P-NCCL patches to the interpreter given as $1.
# Verbatim extraction of the Minerva launcher's patch blocks (run_disagg_benchmark.sh,
# run_experiment_E); idempotent. Usage: patch_vllm_p2p.sh /path/to/python
set -uo pipefail
PY="${1:?python interpreter}"
# vLLM logs to stdout on import (e.g. "Triton not installed" on a GPU-less
# frontend); silence it and keep only the module path line.
export VLLM_LOGGING_LEVEL=ERROR
_p2p_engine_file=$("${PY}" -c \
    "import vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine as m; print(m.__file__)" \
    2>/dev/null | tail -1 || echo "")
if [ -n "${_p2p_engine_file}" ]; then
    echo "  Patching recv_tensor with diagnostic timeout: ${_p2p_engine_file}"
    "${PY}" - "${_p2p_engine_file}" << 'RECV_PATCH_EOF'
import sys, re

path = sys.argv[1]
with open(path, "r") as f:
    src = f.read()

# Only patch once — skip if already patched
if "recv_tensor TIMEOUT" in src:
    print(f"  [patch] already applied, skipping")
    sys.exit(0)

old = (
    "                while tensor_id not in self.recv_store:\n"
    "                    self.recv_store_cv.wait()\n"
    "                tensor = self.recv_store[tensor_id]"
)
new = (
    "                _waited = 0\n"
    "                while tensor_id not in self.recv_store:\n"
    "                    _ok = self.recv_store_cv.wait(timeout=5.0)\n"
    "                    if not _ok:\n"
    "                        _waited += 5\n"
    "                        logger.warning(\n"
    "                            \"⏰recv_tensor TIMEOUT %ds: waiting for \"\n"
    "                            \"tensor_id=[%s] | recv_store has %d keys: %s \"\n"
    "                            \"| rank=%d\",\n"
    "                            _waited, tensor_id, len(self.recv_store),\n"
    "                            sorted(self.recv_store.keys())[:5], self.rank,\n"
    "                        )\n"
    "                        if _waited >= 60:\n"
    "                            logger.warning(\n"
    "                                \"⏰recv_tensor GIVE_UP after %ds for \"\n"
    "                                \"tensor_id=[%s]; returning None\",\n"
    "                                _waited, tensor_id,\n"
    "                            )\n"
    "                            return None\n"
    "                tensor = self.recv_store[tensor_id]"
)

if old not in src:
    print(f"  [patch] ERROR: expected recv_tensor pattern not found in {path}")
    print("  [patch] Snippet around 'recv_store_cv.wait':")
    for i, line in enumerate(src.splitlines()):
        if "recv_store_cv.wait" in line:
            print(f"    L{i}: {line!r}")
    sys.exit(1)

patched = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(patched)
print(f"  [patch] recv_tensor timeout patch applied to {path}")
RECV_PATCH_EOF
else
    echo "  WARNING: could not locate p2p_nccl_engine.py — skipping recv_tensor patch"
fi

# ---------------------------------------------------------------------------
# Root-cause fix: canonical tensor_id in P2pNcclConnector
#
# vLLM internally wraps the user-supplied request_id as:
#   generate-tokens-{user_id}-{random8hex}
# where random8hex is generated *independently* on each server.
# So neptune's tensor key ends in e.g. -8707aa6f and europa's lookup ends
# in e.g. -9d3c1320 — they never match → europa waits forever.
#
# Fix: add _tensor_key() static method that strips both the
# "generate-tokens-" prefix and the trailing "-[0-9a-f]{8}" suffix,
# recovering the stable user-supplied request_id that is identical on both
# servers.  Use it in all three call sites that build tensor_ids:
#   • start_load_kv  (consumer/europa) — recv_tensor lookup key
#   • save_kv_layer  (producer/neptune) — send_tensor key
#   • get_finished   (both sides)       — recv_store cleanup key
# ---------------------------------------------------------------------------
_p2p_connector_file=$("${PY}" -c \
    "import vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector as m; print(m.__file__)" \
    2>/dev/null | tail -1 || echo "")
if [ -n "${_p2p_connector_file}" ]; then
    echo "  Patching _tensor_key canonical fix: ${_p2p_connector_file}"
    "${PY}" - "${_p2p_connector_file}" << 'CONNECTOR_PATCH_EOF'
import sys, re

path = sys.argv[1]
with open(path, "r") as f:
    src = f.read()

# Only patch once
if "_tensor_key" in src:
    print(f"  [patch] _tensor_key already applied, skipping")
    sys.exit(0)

# ---- 1. Add _tensor_key static method before start_load_kv ----
tensor_key_method = (
    "    @staticmethod\n"
    "    def _tensor_key(request_id: str) -> str:\n"
    "        \"\"\"Return a stable tensor key from vLLM's internal request_id.\n"
    "        vLLM wraps user-supplied request_id as:\n"
    "            generate-tokens-{user_id}-{random8hex}\n"
    "        The 8-hex suffix is generated independently per-server, causing\n"
    "        producer/consumer tensor_id mismatches.  Strip both components to\n"
    "        recover the original user_id that is identical on all servers.\n"
    "        \"\"\"\n"
    "        import re as _re\n"
    "        s = _re.sub(r'^generate-tokens-', '', request_id)\n"
    "        s = _re.sub(r'-[0-9a-f]{8}$', '', s)\n"
    "        return s\n"
    "\n"
)

# Insert just before start_load_kv definition
anchor = "    def start_load_kv("
if anchor not in src:
    print(f"  [patch] ERROR: could not find start_load_kv anchor in {path}")
    sys.exit(1)
src = src.replace(anchor, tensor_key_method + anchor, 1)

# ---- 2. Fix start_load_kv (consumer recv_tensor lookup key) ----
old2 = 'request.request_id + "#" + layer_name, remote_address'
new2 = 'self._tensor_key(request.request_id) + "#" + layer_name, remote_address'
if old2 not in src:
    print(f"  [patch] ERROR: could not find start_load_kv tensor_id site in {path}")
    sys.exit(1)
src = src.replace(old2, new2, 1)

# ---- 3. Fix save_kv_layer (producer send_tensor key) ----
old3 = 'request_id + "#" + layer_name, kv_cache, remote_address'
new3 = 'self._tensor_key(request_id) + "#" + layer_name, kv_cache, remote_address'
if old3 not in src:
    print(f"  [patch] ERROR: could not find save_kv_layer tensor_id site in {path}")
    sys.exit(1)
src = src.replace(old3, new3, 1)

# ---- 4. Fix get_finished (recv_store cleanup — pass canonical IDs) ----
old4 = "        return self.p2p_nccl_engine.get_finished(finished_req_ids, no_compile_layers)"
new4 = (
    "        canonical_ids = {self._tensor_key(r) for r in finished_req_ids}\n"
    "        return self.p2p_nccl_engine.get_finished(canonical_ids, no_compile_layers)"
)
if old4 not in src:
    print(f"  [patch] ERROR: could not find get_finished call site in {path}")
    sys.exit(1)
src = src.replace(old4, new4, 1)

with open(path, "w") as f:
    f.write(src)
print(f"  [patch] _tensor_key canonical fix applied to {path}")
CONNECTOR_PATCH_EOF
else
    echo "  WARNING: could not locate p2p_nccl_connector.py — skipping _tensor_key patch"
fi
