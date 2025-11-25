# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import torch

from vllm.config import VllmConfig


class HiddenStatesProcessor(ABC):
    req_hidden_states: dict[str, torch.Tensor] = defaultdict(lambda: None)

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    @abstractmethod
    def apply(self, data: torch.Tensor, req_id: str = "") -> Any:
        ...
