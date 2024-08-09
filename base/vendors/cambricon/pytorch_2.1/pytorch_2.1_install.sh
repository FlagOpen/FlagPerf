set -xe
pip install schedule loguru
pushd /usr/local/neuware/share/cnclbenchmark/cnmpi_wrapper/
bash build.sh
popd

#配置免密
echo 'root:123456' | sudo chpasswd
rm -rf ~/.ssh/* && ssh-keygen -t rsa -N '' -f /root/.ssh/id_rsa -q
sed -i '/StrictHostKeyChecking/c StrictHostKeyChecking no' /etc/ssh/ssh_config
sed -i 's/#Port 22/Port 1234/g' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/g' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
/etc/init.d/ssh restart
#免密
sshpass -p "123456" ssh-copy-id -i ~/.ssh/id_rsa.pub -p 1234 root@`hostname -i | awk '{print $1}'`
