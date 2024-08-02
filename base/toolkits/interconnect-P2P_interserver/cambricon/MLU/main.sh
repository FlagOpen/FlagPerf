# step-1 获取ip
file="../../../../configs/host.yaml"
hosts=$(grep "HOSTS" "$file" | sed -n 's/.*\[\(.*\)\].*/\1/p')
IFS=',' read -ra ADDR <<< "$hosts"
ip1=$(echo "${ADDR[0]}" | sed 's/^[ \t]*//;s/[ \t]*$//' | sed 's/"//g')
ip2=$(echo "${ADDR[1]}" | sed 's/^[ \t]*//;s/[ \t]*$//' | sed 's/"//g')

# step-1 配置免密
echo 'root:123456' | sudo chpasswd
rm -rf ~/.ssh/* && ssh-keygen -t rsa -N '' -f /root/.ssh/id_rsa -q
sed -i '/StrictHostKeyChecking/c StrictHostKeyChecking no' /etc/ssh/ssh_config
sed -i 's/#Port 22/Port 1234/g' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/g' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
/etc/init.d/ssh restart
sleep 10
sshpass -p "123456" ssh-copy-id -i ~/.ssh/id_rsa.pub -p 1234 root@${ip1}
sshpass -p "123456" ssh-copy-id -i ~/.ssh/id_rsa.pub -p 1234 root@${ip2}

# step-3 正式测试
cur_ip=`hostname -i | awk '{print $1}'`
if [ "$cur_ip" == "$ip1" ]; then
    export MLU_VISIBLE_DEVICES=0
else
    export MLU_VISIBLE_DEVICES=1
fi
LOG_PATH=`pwd`/`hostname -i | awk '{print $1}'`_run_log
tcp_if_include=`echo ${ip1} | awk -F'.' '{print $1"."$2"."$3}'`
/usr/local/openmpi/bin/mpirun \
        --allow-run-as-root -n 2 --host ${ip1}:1,${ip2}:1 \
        -x PATH -x LD_LIBRARY_PATH -x MLU_VISIBLE_DEVICES \
        -mca btl ^openib  -bind-to none -map-by slot -mca plm_rsh_args \
        "-p 1234" -mca btl_tcp_if_include ${tcp_if_include}.0/24 \
        /usr/local/neuware/bin/sendrecv --warmup_loop 21 --thread 1 --loop 250 --mincount 1 --maxcount 512M --multifactor 2 --async 1 --block 0 \
        2>&1 | tee ${LOG_PATH}
data=$(tail -n 2 ${LOG_PATH} | awk '{print $10}')
sleep 30
echo "[FlagPerf Result]interconnect-P2P_interserver-bandwidth=$data GB/s"
rm -rf ${LOG_PATH}