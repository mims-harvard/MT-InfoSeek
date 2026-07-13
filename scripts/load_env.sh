# Shared .env loader for the dataset scripts.  (source this, don't execute it)
#
# `.env` supplies DEFAULTS. Anything already present in the environment wins.
#
# This matters: run_eval.py passes DATA_DIR / RESULTS_DIR / VLLM_PORT / ... to
# each dataset script. The old `set -a; source .env` overwrote them, so results
# were written to the wrong directory and `data/smoke` was silently replaced by
# the full dataset. Environment-beats-file is also how run_eval.py reads .env,
# so the two agree.

_load_env() {
    local f="${1:-}"
    [ -f "$f" ] || return 0
    local line key
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line#"${line%%[![:space:]]*}"}"     # strip leading whitespace
        case "$line" in ''|'#'*) continue ;; esac
        line="${line#export }"
        key="${line%%=*}"
        case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac   # skip odd lines
        [ -n "${!key+x}" ] && continue               # already set -> keep it
        eval "export $line"
    done < "$f"
}
