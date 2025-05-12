From zhiyuan_flagscale_mixtral:newest

RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
ENV PATH /root/miniconda/envs/python38_torch201_cuda/bin:$PATH

RUN /bin/bash -c 'source /root/.bashrc; sed -i.bak "s/Port 4323/Port 3702/" /etc/ssh/sshd_config'
RUN /bin/bash -c 'source /root/.bashrc; sed -i.bak "s/    Port 4323/    Port 3702/" /root/.ssh/config'
RUN /bin/bash -c 'source /root/.bashrc; sed -i.bak "/StrictHostKeyChecking/c StrictHostKeyChecking no" /root/.ssh/config'
RUN /bin/bash -c "source /root/.bashrc; service ssh restart"

RUN echo '#!/bin/bash\nservice ssh restart\nexec "$@"' > /docker-start.sh

RUN chmod +x /docker-start.sh
RUN apt-get install -y openssh-server
ENTRYPOINT ["sh", "/docker-start.sh"]
