# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

from vllm.ray.ray_env import RAY_NON_CARRY_OVER_ENV_VARS


NODE_LOCAL_PATH_ENV_VARS = (
    "CUTE_DSL_LIBS",
    "CUTE_EXPERIMENTAL_DSL_LIBS",
    "TVM_LIBRARY_PATH",
    "TVM_IMPORT_PYTHON_PATH",
    "TL_CUTLASS_PATH",
    "TL_COMPOSABLE_KERNEL_PATH",
    "TL_TEMPLATE_PATH",
)


def get_driver_env_vars(
    worker_specific_vars: set[str],
) -> dict[str, str]:
    """Return driver env vars to propagate to Ray workers.

    Returns everything from ``os.environ`` except ``worker_specific_vars``
    and user-configured exclusions (``RAY_NON_CARRY_OVER_ENV_VARS``).
    """
    exclude_vars = worker_specific_vars | RAY_NON_CARRY_OVER_ENV_VARS

    return {key: value for key, value in os.environ.items() if key not in exclude_vars}


def sanitize_node_local_path_env_vars() -> list[str]:
    """Remove path entries inherited from another Ray node."""
    sanitized: list[str] = []
    for name in NODE_LOCAL_PATH_ENV_VARS:
        value = os.environ.get(name)
        if not value:
            continue
        paths = [path for path in value.split(os.pathsep) if path]
        local_paths = [path for path in paths if os.path.exists(path)]
        if local_paths == paths:
            continue
        sanitized.append(name)
        if local_paths:
            os.environ[name] = os.pathsep.join(local_paths)
        else:
            os.environ.pop(name, None)
    return sanitized
