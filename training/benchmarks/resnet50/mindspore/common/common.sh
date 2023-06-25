#!/bin/bash

check_file_valid()
{
    if [ ! -f "$1" ]; then
        return 1
    fi
    return 0
}

check_path_valid()
{
    if [ ! -d "$1" ]; then
        return 1
    fi
    return 0
}

function check_command_exist()
{
    command=$1
    if type ${command} > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_python_package_is_install()
{
    local PYTHON_COMMAND=$1
    ${PYTHON_COMMAND} -c "import $2" >> /dev/null 2>&1
    ret=$?
    if [ $ret != 0 ]; then
        echo "python package:$1 not install"
        return 1
    fi
    return 0
}

check_mindspore_run_ok()
{
    local PYTHON_COMMAND=$1
    ${PYTHON_COMMAND} -c "import mindspore;mindspore.run_check()" >> /dev/null 2>&1
    ret=$?
    if [ $ret != 0 ]; then
        echo "mindspore run not ok"
        return 1
    fi
}
