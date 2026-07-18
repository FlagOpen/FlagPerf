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

FROM zhiyuan_flagscale_llama:newest
# FROM iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.29
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
ENV PATH /root/miniconda/envs/python39_torch201_cuda/bin:$PATH

RUN /bin/bash -c 'source /root/.bashrc; sed -i.bak "s/Port 4323/Port 3702/" /etc/ssh/sshd_config'
RUN /bin/bash -c 'source /root/.bashrc; sed -i.bak "s/    Port 4323/    Port 3702/" /root/.ssh/config'
RUN /bin/bash -c "source /root/.bashrc; service ssh restart"

COPY output/xpytorch-cp39-torch201-ubuntu2004-x64.run /root/.local/
RUN /bin/bash -c "source /root/.bashrc; cd /root/.local; bash /root/.local/xpytorch-cp39-torch201-ubuntu2004-x64.run"

RUN echo '#!/bin/bash\nservice ssh restart\nexec "$@"' > /docker-start.sh

RUN chmod +x /docker-start.sh
RUN apt-get install -y openssh-server
ENTRYPOINT ["sh", "/docker-start.sh"]
