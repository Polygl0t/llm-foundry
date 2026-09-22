#!/usr/bin/env bash

#############################################
# Helper to make the RULER haystack work OFFLINE
#   JSC JUPITER Stages/2026
#############################################

RULER_HAYSTACK_SUBDIR="ruler_haystack"
_RULER_PREPARE_REL="lm_eval/tasks/ruler_pt/prepare_niah.py"

ruler_haystack_url() {
    local harness_dir="$1"
    sed -n -E 's/^[[:space:]]*HAYSTACK_URL[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' \
        "$harness_dir/$_RULER_PREPARE_REL" | head -1
}

# Download the haystack into the shared cache.
ensure_ruler_haystack_file() {
    local cache_dir="$1"
    local harness_dir="${2:-}"

    if [[ -z "$harness_dir" ]]; then
        echo "ensure_ruler_haystack_file: harness_dir is required" >&2
        return 1
    fi

    local url
    url="$(ruler_haystack_url "$harness_dir")"
    if [[ -z "$url" ]]; then
        echo "[haystack] WARNING: no HAYSTACK_URL found in $_RULER_PREPARE_REL" >&2
        return 1
    fi

    local target_dir="$cache_dir/$RULER_HAYSTACK_SUBDIR"
    local target="$target_dir/$(basename "$url")"
    mkdir -p "$target_dir"

    if [[ -s "$target" ]]; then
        echo "[haystack] cached: $target ($(du -h "$target" | cut -f1))"
        return 0
    fi

    echo "[haystack] downloading $(basename "$url")"
    if python3 - "$url" "$target" <<'PY'
import sys
import urllib.request

url, target = sys.argv[1], sys.argv[2]
# A UA avoids a 403 from some raw-content endpoints.
request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
with urllib.request.urlopen(request, timeout=180) as response:
    payload = response.read()
if not payload:
    raise SystemExit("empty response")
with open(target, "wb") as handle:
    handle.write(payload)
print(f"  {len(payload):,} bytes")
PY
    then
        echo "[haystack] saved: $target"
        return 0
    fi

    rm -f "$target"
    echo "[haystack] ERROR: download failed (needs internet -- run on a login node)" >&2
    return 1
}

apply_ruler_haystack_patch() {
    local harness_dir="$1"
    local prepare="$harness_dir/$_RULER_PREPARE_REL"

    if [[ ! -f "$prepare" ]]; then
        echo "[patch] WARNING: not found: $prepare" >&2
        return 1
    fi

    python3 - "$prepare" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
source = path.read_text(encoding="utf-8")

if "RULER_HAYSTACK_DIR" in source:
    print("[patch] already applied (no-op)")
    raise SystemExit(0)

ORIGINAL = '''        response = requests.get(HAYSTACK_URL)
        response.raise_for_status()
        essay = response.text
'''

PATCHED = '''        # --- local patch (JUPITER) -------------------------------------------
        # Prefer a pre-downloaded copy of the haystack when RULER_HAYSTACK_DIR
        # is set. RULER builds this dataset with `custom_dataset`, which
        # bypasses the HF datasets cache, so otherwise this URL is fetched on
        # every run -- and compute nodes have no internet.
        # Applied by ruler_patch.sh; safe to re-run.
        _local_dir = os.environ.get("RULER_HAYSTACK_DIR", "")
        if _local_dir:
            _local_file = os.path.join(_local_dir, os.path.basename(HAYSTACK_URL))
            if not os.path.exists(_local_file):
                raise FileNotFoundError(
                    "RULER haystack missing from the offline cache: "
                    f"{_local_file}. Fetch it with "
                    "jupiter_installation_eval_harness_2026.sh on a login node."
                )
            with open(_local_file, encoding="utf-8") as _fh:
                essay = _fh.read()
        else:
            response = requests.get(HAYSTACK_URL)
            response.raise_for_status()
            essay = response.text
        # --- end local patch -------------------------------------------------
'''

if ORIGINAL not in source:
    print(
        "[patch] ERROR: the expected get_haystack() block was not found -- "
        f"{path} differs from what this patch was written against.",
        file=sys.stderr,
    )
    raise SystemExit(1)

path.write_text(source.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
print("[patch] prepare_niah.py now reads the cached haystack")
PY
}
