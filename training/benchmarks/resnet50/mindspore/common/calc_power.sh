#!/bin/bash
# need to specify ip user password

SUCCESS=0
FAIL=1

function get_power()
{
    result=`ipmitool -H ${BMC_IP} -I lanplus -U ${BMC_USER} -P ${BMC_PASSWORD} raw 0x30 0x93 0xdb 0x07 0x00 0x11 0x00`
    first=`echo ${result} | awk '{print $1}'`
    except='db'
    if [ "${except}" != "${first}" ];then
        echo "ERROR:ipmitool run fail,result=${result}"
        BMC_POWER=0
        return ${FAIL}
    fi
    high=`echo ${result} | awk '{print $5}'`
    low=`echo ${result} | awk '{print $4}'`
    power_str="${high}${low}"
    power=$((16#${power_str}))
    BMC_POWER=${power}
    #echo `date`" power:${BMC_POWER}"
    return ${SUCCESS}
}

function calculate_emptyload_power()
{
    max_num=$1
    num_i=0

    # clear
    BMC_POWER=0
    # MAX_POWER=0
    SUM_POWER=0

    while [[ num_i -lt max_num ]]; do
        get_power
        if [ $? -eq ${FAIL} ];then
            echo "ERROR:get power fail, get_power exit"
            return ${FAIL}
        fi
        # if [ "${BMC_POWER}" -gt "${MAX_POWER}" ];then
        #     MAX_POWER=${BMC_POWER}
        # fi
        SUM_POWER=$((10#${SUM_POWER}+${BMC_POWER}))
        let num_i=num_i+1
        sleep 6
    done
    export EMPTYLOAD_AVERAGE_POWER=$((10#${SUM_POWER}/${max_num}))
}

exit_trap()
{
    trap - USR2
    RUNING_AVERAGE_POWER=$((10#${RUNING_SUM_POWER}/${loop_count}))

    AVERAGE_POWER=$(python -c "import ais_utils; print(ais_utils.calc_single_avg_power( $EMPTYLOAD_AVERAGE_POWER, $RUNING_AVERAGE_POWER))")
    MAX_POWER=$(python -c "import ais_utils; print(ais_utils.calc_single_max_power(${EMPTYLOAD_AVERAGE_POWER}, ${RUNING_MAX_POWER}))")

    python $CUR_PATH/ais_utils.py set_result "training" "average_power" ${AVERAGE_POWER}
    python $CUR_PATH/ais_utils.py set_result "training" "max_power" ${MAX_POWER}

    echo "RUNING_AVERAGE_POWER:$RUNING_AVERAGE_POWER EMPTYLOAD_AVERAGE_POWER:$EMPTYLOAD_AVERAGE_POWER AVERAGE_POWER:$AVERAGE_POWER MAX_POWER:$MAX_POWER"

    echo "exit end $$ $loopflag loop end"
}

function calculate_runing_power()
{
    trap exit_trap USR2
    loop_count=0

    # clear
    BMC_POWER=0
    RUNING_MAX_POWER=0
    RUNING_SUM_POWER=0

    while [[ true ]]; do
        get_power
        if [ $? -eq ${FAIL} ];then
            echo "ERROR:get power fail, get_power exit"
            return ${FAIL}
        fi
        if [ "${BMC_POWER}" -gt "${RUNING_MAX_POWER}" ];then
             RUNING_MAX_POWER=${BMC_POWER}
        fi
        RUNING_SUM_POWER=$((10#${RUNING_SUM_POWER}+${BMC_POWER}))
        let loop_count=loop_count+1
        sleep 6
    done
    echo "calc end"
}

check_command_exist()
{
    command=$1
    if type $command >/dev/null  2>&1;then
        return 0
    else
        return 1
    fi
}

function calc_powerinfo_backgroud()
{
    check_command_exist "ipmitool"
    if [[ $? -ne 0 || -z "$BMC_IP" || -z "$BMC_USER" || -z "$BMC_PASSWORD" ]];then
        echo "not valid power env ret"
        return 0
    fi
    timeout 4 ping -c3 -i1 $BMC_IP >> /dev/null 2>&1
    if [ $? -ne 0 ];then
        echo "not valid bmc_ip"
        return 0
    fi
    calculate_emptyload_power 3

    calculate_runing_power &
    export power_monitor_pid=$!
    echo "power monitor pid:$power_monitor_pid"
}

function set_powerinfo()
{
    if [ ! -z "$power_monitor_pid" ];then
        kill -12 $power_monitor_pid
        sleep 10
        echo "send signel sleep done. power_monitor_pid: $power_monitor_pid"
        kill -9 $power_monitor_pid
    fi
}

# main()
# {
#     export BMC_IP="90.90.66.25"
#     export BMC_USER="Administrator"
#     export BMC_PASSWORD="Admin@9000"
#     export BMC_PASSWORD=""

#     calc_powerinfo_backgroud
#     echo "get power ret:$?"

#     sleep 10
#     echo "sleep done now calc and kill "
#     set_powerinfo
#     echo "set power power ret:$?"
# }

# main $@
