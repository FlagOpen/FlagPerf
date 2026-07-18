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

torch::Tensor gather_gpu_fwd(torch::Tensor input, torch::Tensor weight);
void gather_gpu_bwd_fuse_sgd(const torch::Tensor grad, const torch::Tensor indices, float lr, torch::Tensor weight);
torch::Tensor gather_gpu_bwd(const torch::Tensor grad, const torch::Tensor indices, const int num_features);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_gpu_fwd", &gather_gpu_fwd, "Embedding gather", py::arg("indices"), py::arg("weight"));
  m.def("gather_gpu_bwd_fuse_sgd", &gather_gpu_bwd_fuse_sgd, "Embedding gather backward with fused plain SGD",
        py::arg("grad"), py::arg("indices"), py::arg("lr"), py::arg("weight"));
  m.def("gather_gpu_bwd", &gather_gpu_bwd, "Embedding gather backward",
        py::arg("grad"), py::arg("indices"), py::arg("num_features"));
}
