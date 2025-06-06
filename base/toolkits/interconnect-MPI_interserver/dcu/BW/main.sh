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

LOG_PATH=$(pwd)/$(ip a | grep -w 'inet' | grep 'global' | sed 's/.*inet //;s/\/.*//' | awk 'NR==1{print $1}')_run_log

cur_ip=$(ip a | grep -w 'inet' | grep 'global' | sed 's/.*inet //;s/\/.*//' | awk 'NR==1{print $1}')
if [[ "${cur_ip}" == "${ip1}" ]];then
        mpirun -x NCCL_IB_HCA=mlx5_10,mlx5_11,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9 \
            -x NCCL_GRAPH_FILE=/opt/graph.xml -x NCCL_TOPO_FILE=/opt/topo.xml \
            --mca btl_tcp_if_include ${ip1}/24 \
            --allow-run-as-root \
            --host ${ip1}:8,${ip2}:8 \
            -x ROCM_PATH=/opt/dtk \
            -x RCCL_SDMA_COUNT_ENABLE=1 -x RCCL_SDMA_COPY_ENABLE=1 -x RCCL_COLL_XHCL_CHANNEL_NUM=28 \
            -mca plm_rsh_args "-p 1234" -np 16 -x LD_LIBRARY_PATH=/opt/hyhal/lib:${LD_LIBRARY_PATH} \
            -x UCX_NET_DEVICES=mlx5_6:1 -x NCCL_NET_GDR_LEVEL=5 -x NCCL_NET_GDR_READ=1 all_reduce_perf -b 256m -e 256m -g 1 -f 2 2>&1 | tee ${LOG_PATH}
            
        data=$(grep "# Avg bus bandwidth" ${LOG_PATH} | awk '{print $NF}')
        result=$(python3 -c "print(float($data) * 2)")
        while  [ ! -f ${ip2}_run_log ] || ! grep -q "Avg bus bandwidth" ${ip2}_run_log ; do
                sleep 1 
        done
        echo "[FlagPerf Result]interconnect-MPI_interserver-bandwidth=$result GB/s"
        rm -rf ${ip1}_run_log ${ip2}_run_log
else
        while [ ! -f ${ip1}_run_log ] || ! grep -q "Avg bus bandwidth" ${ip1}_run_log ; do
                sleep 1  
        done

        mpirun -x NCCL_IB_HCA=mlx5_10,mlx5_11,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9 \
            -x NCCL_GRAPH_FILE=/opt/graph.xml -x NCCL_TOPO_FILE=/opt/topo.xml \
            --mca btl_tcp_if_include ${ip1}/24 \
            --allow-run-as-root \
            --host ${ip1}:8,${ip2}:8 \
            -x ROCM_PATH=/opt/dtk \
            -x RCCL_SDMA_COUNT_ENABLE=1 -x RCCL_SDMA_COPY_ENABLE=1 -x RCCL_COLL_XHCL_CHANNEL_NUM=28 \
            -mca plm_rsh_args "-p 1234" -np 16 -x LD_LIBRARY_PATH=/opt/hyhal/lib:${LD_LIBRARY_PATH} \
            -x UCX_NET_DEVICES=mlx5_6:1 -x NCCL_NET_GDR_LEVEL=5 -x NCCL_NET_GDR_READ=1 all_reduce_perf -b 256m -e 256m -g 1 -f 2 2>&1 | tee ${LOG_PATH}
            
        data=$(grep "# Avg bus bandwidth" ${LOG_PATH} | awk '{print $NF}')
        result=$(python3 -c "print(float($data) * 2)")
        echo "[FlagPerf Result]interconnect-MPI_interserver-bandwidth=$result GB/s"

fi