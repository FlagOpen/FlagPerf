FROM nvcr.io/nvidia/pytorch:22.11-py3
RUN /bin/bash -c "pip config set global.index-url https://mirror.baidu.com/pypi/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN pip install editdistance==0.6.0 librosa==0.8.0 pyarrow==6.0.1 pip install soundfile==0.10.3.post1 pip install sox==1.4.1 pip install tqdm==4.53.0
RUN pip install git+https://github.com/NVIDIA/dllogger@v1.0.0#egg=dllogger