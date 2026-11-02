# SPDX-License-Identifier: Apache-2.0
"""Worker for the pinned core/Ascend pair, including its Mamba ABI bridge.

This host compatibility choice is explicit in the reproduction launcher and
also applies to the compression-disabled control. Matched newer hosts can use
their native worker without this class.
"""

from vllm_ascend.worker.worker import NPUWorker


class PyramidKVHostWorker(NPUWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .host_compat import install_mamba_v1_sd_compat

        install_mamba_v1_sd_compat()
