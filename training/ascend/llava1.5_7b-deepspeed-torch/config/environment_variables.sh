# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# When using Conda, please specify the location of your environment variable
# If you are using the system's environment, there is no need to set this variable
source /root/anaconda3/bin/activate /opt/nvme1n1/conda-envs/patch; 

# Activate the environment related to Ascend
source /usr/local/Ascend/driver/bin/setenv.bash; 
source /usr/local/Ascend/ascend-toolkit/set_env.sh
