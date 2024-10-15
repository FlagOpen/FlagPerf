FROM zhiyuan_flagscale_llava:newest
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
ENV PATH /root/miniconda/envs/python39_torch201_cuda/bin:$PATH

RUN echo '#!/bin/bash\nservice ssh restart\nexec "$@"' > /docker-start.sh

RUN chmod +x /docker-start.sh
RUN apt-get install -y openssh-server
ENTRYPOINT ["sh", "/docker-start.sh"]

