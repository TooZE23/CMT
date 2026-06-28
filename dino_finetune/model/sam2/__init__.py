# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

# Always expose this bundled package as top-level `sam2` to avoid
# accidentally importing another project/package with the same name.
sys.modules["sam2"] = sys.modules[__name__]

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("sam2", version_base="1.2")

