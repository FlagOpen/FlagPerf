// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>

torch::Tensor dotBasedInteractFwdTorch(torch::Tensor input,
                                       torch::Tensor bottom_mlp_output);
std::vector<torch::Tensor> dotBasedInteractBwdTorch(torch::Tensor input,
                                                    torch::Tensor upstreamGrad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dotBasedInteractFwd", &dotBasedInteractFwdTorch, "", py::arg("input"),
        py::arg("bottom_mlp_output"));
  m.def("dotBasedInteractBwd", &dotBasedInteractBwdTorch, "", py::arg("input"),
        py::arg("upstreamGrad"));
}
