# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.plugins.hidden_states_processors.interface import (
    HiddenStatesProcessor)


class IdentityHiddenStatesProcessor(HiddenStatesProcessor):

    def apply(self, data: torch.Tensor, req_id="") -> Any:
        """
        This is the default identity hidden states processor
        that returns the hidden_states data as is
        """
        if not req_id:
            return data

        if data.dim() == 1:
            data = data.unsqueeze(0)

        if self.req_hidden_states[req_id] is None:
            self.req_hidden_states[req_id] = data
        else:
            self.req_hidden_states[req_id] = torch.cat([self.req_hidden_states[req_id], data], dim=0)

        return self.req_hidden_states[req_id]