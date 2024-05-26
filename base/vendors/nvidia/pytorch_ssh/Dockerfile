FROM nvcr.io/nvidia/pytorch:23.12-py3
RUN apt-get update && \
    apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:password' | chpasswd
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh
RUN printf "Host *\n  User root\n  SendEnv PATH\n  SendEnv LD_LIBRARY_PATH\n  Port 60128" > /root/.ssh/config && chmod 600 /root/.ssh/config
COPY id_rsa.pub /root/.ssh/authorized_keys
COPY id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/authorized_keys
RUN sed -i 's/#Port 22/Port 60128/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
RUN echo 'AcceptEnv PATH LD_LIBRARY_PATH' >> /etc/ssh/sshd_config
EXPOSE 60128
CMD ["/usr/sbin/sshd", "-D"]
