#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

drama3_resolve_python() {
    local runtime_python
    if [[ -n ${DRAMA3_PYTHON:-} ]]; then
        runtime_python=$DRAMA3_PYTHON
    elif [[ -n ${VIRTUAL_ENV:-} ]]; then
        runtime_python="$VIRTUAL_ENV/bin/python"
    elif [[ -n ${PYENV_VERSION:-} ]] && command -v pyenv >/dev/null 2>&1; then
        runtime_python=$(pyenv which python) || return
    else
        echo "Activate your Python environment, or set DRAMA3_PYTHON to its Python executable." >&2
        return 1
    fi
    if [[ ! -x "$runtime_python" ]]; then
        echo "Activate your installed vLLM environment, or set DRAMA3_PYTHON to its Python executable." >&2
        return 1
    fi
    printf '%s/%s\n' "$(cd -- "$(dirname -- "$runtime_python")" && pwd -P)" "$(basename -- "$runtime_python")"
}
