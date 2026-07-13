#!/usr/bin/env bash
# One-time setup for MT-InfoSeek.
#
# Creates the evaluator environment and installs two GeneReg dependencies:
#   1. Python evaluator environment      -> .venv
#   2. GeneReg-MT attractor/basin cache  -> extracted from the committed tarball.
#   3. Kadelka Boolean GRN models        -> sparse-cloned at a pinned commit.
# vLLM is optional because its build depends on the serving hardware. Install it
# on the GPU machine with --with-vllm; it is kept in .venv-vllm.
#
# Idempotent: re-running verifies and skips work already done. Never touches
# data/ or any result. Writes only under the locations named in .env (or the
# defaults below).
#
#   bash setup.sh                 # evaluator + all released datasets
#   bash setup.sh --with-vllm     # also install vLLM (run on the GPU machine)
#   bash setup.sh --no-genereg    # evaluator only
#   bash setup.sh --verify        # check only; make no changes
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
die() { printf "\033[31mERROR: %s\033[0m\n" "$*" >&2; exit 1; }

VERIFY_ONLY=0
VERIFY_TAG=""
VERIFY_MISSING=0
WITH_VLLM=0
SKIP_GENEREG=0
for arg in "$@"; do
    case "$arg" in
        --verify) VERIFY_ONLY=1; VERIFY_TAG=" (verify only)" ;;
        --with-vllm) WITH_VLLM=1 ;;
        --no-genereg) SKIP_GENEREG=1 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *) die "unknown option: $arg" ;;
    esac
done

# Pull paths from .env if present.
_ENV_FILE="${ROOT_DIR}/.env"
[ -f "$_ENV_FILE" ] && set -a && source "$_ENV_FILE" && set +a

# Pinned Kadelka commit. Attractor labels are index-based, so a different commit
# could reorder them; keep this fixed for reproducibility.
KADELKA_REPO="https://github.com/ckadelka/DesignPrinciplesGeneNetworks"
KADELKA_SHA="932844cded53a0f9da9886710917803b42b3edb7"
KADELKA_SUBDIR="update_rules_122_models_Kadelka_SciAdv"

CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/.grn_cache}"
KADELKA_DIR="${KADELKA_DIR:-${ROOT_DIR}/.kadelka}"
MODELS_DIR="${MODELS_DIR:-${KADELKA_DIR}/${KADELKA_SUBDIR}}"

CACHE_TARBALL="${ROOT_DIR}/genereg_mt/assets/models_cache.tar.gz"
EVAL_VENV="${ROOT_DIR}/.venv"
VLLM_VENV="${ROOT_DIR}/.venv-vllm"

echo "============================================================"
echo "  MT-InfoSeek setup${VERIFY_TAG}"
echo "    evaluator   : ${EVAL_VENV}"
if [ "$WITH_VLLM" -eq 1 ]; then
    echo "    vLLM        : ${VLLM_VENV}"
fi
echo "    CACHE_DIR   : ${CACHE_DIR}"
echo "    MODELS_DIR  : ${MODELS_DIR}"
echo "============================================================"

# ── 0. Evaluator environment ────────────────────────────────────────────────
setup_evaluator() {
    if [ "$VERIFY_ONLY" -eq 1 ]; then
        if [ ! -x "${EVAL_VENV}/bin/python" ]; then
            yellow "[evaluator] MISSING (${EVAL_VENV}); run without --verify"
            VERIFY_MISSING=1
            return
        fi
        if ! "${EVAL_VENV}/bin/python" -c \
            'import networkx, numpy, openai, pandas, transformers, yaml' >/dev/null 2>&1; then
            yellow "[evaluator] INCOMPLETE; run without --verify to install requirements"
            VERIFY_MISSING=1
            return
        fi
        green "[evaluator] present (${EVAL_VENV})"
        return
    fi

    command -v uv >/dev/null 2>&1 || die \
        "uv is required (https://docs.astral.sh/uv/getting-started/installation/)"
    if [ ! -x "${EVAL_VENV}/bin/python" ]; then
        uv venv --python 3.10 "${EVAL_VENV}"
    fi
    uv pip install --python "${EVAL_VENV}/bin/python" -r "${ROOT_DIR}/requirements.txt"
    green "[evaluator] ready (${EVAL_VENV})"
}

setup_vllm() {
    if [ "$WITH_VLLM" -ne 1 ]; then
        return 0
    fi
    if [ "$VERIFY_ONLY" -eq 1 ]; then
        if [ ! -x "${VLLM_VENV}/bin/vllm" ]; then
            yellow "[vllm] MISSING (${VLLM_VENV}); run --with-vllm without --verify"
            VERIFY_MISSING=1
        else
            green "[vllm] present (${VLLM_VENV})"
        fi
        return
    fi

    command -v uv >/dev/null 2>&1 || die \
        "uv is required (https://docs.astral.sh/uv/getting-started/installation/)"
    if [ ! -x "${VLLM_VENV}/bin/python" ]; then
        uv venv --python 3.12 --seed "${VLLM_VENV}"
    fi
    if uv pip install --help 2>&1 | grep -q -- '--torch-backend'; then
        uv pip install --python "${VLLM_VENV}/bin/python" vllm --torch-backend=auto
    else
        yellow "[vllm] this uv version lacks --torch-backend=auto; using the default PyPI build"
        uv pip install --python "${VLLM_VENV}/bin/python" vllm
    fi
    green "[vllm] ready (${VLLM_VENV})"
}

# ── 1. GeneReg-MT attractor/basin cache ──────────────────────────────────────
count_pkl() {   # never fails, even when the directory does not exist yet
    [ -d "$1" ] || { echo 0; return 0; }
    find "$1" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l | tr -d ' '
}

setup_cache() {
    local n_pkl
    n_pkl=$(count_pkl "$CACHE_DIR")
    if [ "${n_pkl:-0}" -ge 167 ]; then
        green "[cache] present (${n_pkl} pickles in ${CACHE_DIR})"
        return
    fi
    if [ "$VERIFY_ONLY" -eq 1 ]; then
        yellow "[cache] MISSING (${n_pkl:-0} pickles); run without --verify to extract"
        VERIFY_MISSING=1
        return
    fi
    [ -f "$CACHE_TARBALL" ] || die "cache tarball not found: $CACHE_TARBALL"
    if command -v sha256sum >/dev/null 2>&1 && [ -f "${CACHE_TARBALL}.sha256" ]; then
        ( cd "$(dirname "$CACHE_TARBALL")" && sha256sum -c "$(basename "$CACHE_TARBALL").sha256" ) \
            || die "cache tarball checksum mismatch"
    fi
    mkdir -p "$CACHE_DIR"
    tar -xzf "$CACHE_TARBALL" -C "$CACHE_DIR"
    n_pkl=$(count_pkl "$CACHE_DIR")
    green "[cache] extracted ${n_pkl} pickles -> ${CACHE_DIR}"
}

# ── 2. Kadelka Boolean GRN models ────────────────────────────────────────────
setup_kadelka() {
    local n_pickle=0
    [ -d "$MODELS_DIR" ] && n_pickle=$(find "$MODELS_DIR" -maxdepth 1 -name '*.pickle' 2>/dev/null | wc -l | tr -d ' ')
    if [ "${n_pickle:-0}" -ge 122 ]; then
        green "[kadelka] present (${MODELS_DIR})"
        return
    fi
    if [ "$VERIFY_ONLY" -eq 1 ]; then
        yellow "[kadelka] MISSING; run without --verify to clone at ${KADELKA_SHA:0:12}"
        VERIFY_MISSING=1
        return
    fi
    command -v git >/dev/null 2>&1 || die "git is required to fetch the Kadelka models"

    if [ ! -d "${KADELKA_DIR}/.git" ]; then
        # Blobless + sparse: fetch only the one subdir we use, at the pinned SHA.
        git clone --filter=blob:none --no-checkout "$KADELKA_REPO" "$KADELKA_DIR"
    fi
    (
        cd "$KADELKA_DIR"
        git sparse-checkout init --cone >/dev/null 2>&1 || git sparse-checkout init >/dev/null 2>&1
        git sparse-checkout set "$KADELKA_SUBDIR"
        git fetch --depth 1 origin "$KADELKA_SHA" 2>/dev/null || git fetch origin
        git checkout "$KADELKA_SHA" -- "$KADELKA_SUBDIR" 2>/dev/null || git checkout "$KADELKA_SHA"
    )
    [ -d "$MODELS_DIR" ] || die "Kadelka checkout did not produce ${MODELS_DIR}"
    green "[kadelka] checked out ${KADELKA_SUBDIR} @ ${KADELKA_SHA:0:12}"
}

setup_evaluator
setup_vllm
if [ "$SKIP_GENEREG" -eq 0 ]; then
    setup_cache
    setup_kadelka
fi

echo
if [ "$VERIFY_ONLY" -eq 1 ]; then
    if [ "$VERIFY_MISSING" -ne 0 ]; then
        die "setup verification failed; run 'bash setup.sh' to install missing components"
    fi
    green "Setup verification passed."
    exit 0
fi
green "Setup complete."
echo "    evaluator: ${EVAL_VENV}/bin/python"
if [ "$WITH_VLLM" -eq 1 ]; then
    echo "    vLLM:      ${VLLM_VENV}/bin/vllm"
fi
if [ "$SKIP_GENEREG" -eq 0 ]; then
    echo "    CACHE_DIR: ${CACHE_DIR}"
    echo "    MODELS_DIR: ${MODELS_DIR}"
fi
