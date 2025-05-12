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
