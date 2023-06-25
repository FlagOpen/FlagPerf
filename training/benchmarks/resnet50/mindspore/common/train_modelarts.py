import argparse
import logging
import os
import sys
from statistics import mean

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import ais_utils
from config.modelarts_config import access_config
from config.modelarts_config import session_config as session_config_v1
from config.modelarts_config import session_config_v2
from modelarts_handler_v2 import modelarts_handler as modelarts_handler_v2

from modelarts_handler import logger, modelarts_handler


def report_result(handler):
    ranksize_file_url = os.path.join(handler.output_url, 'ranksize.json')
    ranksize = int(handler.get_obs_url_content(ranksize_file_url))
    print("url:{} read ranksize:{}".format(ranksize_file_url, ranksize))

    total_throughput = 0.0
    for rankid in range(0, ranksize):
        throughput_url = os.path.join(handler.output_url, 'throughput_' + str(rankid) + '.json')
        single_throughput_rate = float(handler.get_obs_url_content(throughput_url))
        print("rankid:{} url:{} read throughput:{}".format(rankid, throughput_url, single_throughput_rate))
        total_throughput = total_throughput + single_throughput_rate
    print("report result total_throughput : {}".format(total_throughput))
    ais_utils.set_result("training", "throughput_ratio", total_throughput)

    accuracy_file_url = os.path.join(handler.output_url, 'accuracy.json')
    accuracy = float(handler.get_obs_url_content(accuracy_file_url))
    print("url:{} read accuracy:{}".format(accuracy_file_url, accuracy))

    print("report result accuracy:{}".format(accuracy))
    ais_utils.set_result("training", "accuracy", accuracy)

# 单设备运行模式
def report_result_singlesever_mode(handler, server_count):
    # 单设备运行模式下默认都是8卡
    cards_per_server = 8
    print("server_count:{} cards_per_server:{}".format(server_count, cards_per_server))

    throughput_list = []
    accuracy_list = []
    for server_id in range(server_count):
        single_server_throughput = 0.0
        for rankid in range(cards_per_server):
            throughput_url = os.path.join(handler.output_url, str(server_id), 'throughput_' + str(rankid) + '.json')
            single_card_throughput = float(handler.get_obs_url_content(throughput_url))
            print("rankid:{} url:{} read throughput:{}".format(rankid, throughput_url, single_card_throughput))
            single_server_throughput = single_server_throughput + single_card_throughput
        print("serverid:{} count:{} service_throughput:{}".format(server_id, server_count, single_server_throughput))
        throughput_list.append(single_server_throughput)

        accuracy_file_url = os.path.join(handler.output_url, 'accuracy_{}.json'.format(server_id))
        single_server_accuracy = float(handler.get_obs_url_content(accuracy_file_url))
        print("serverid:{} url:{} read accuracy:{}".format(server_id, accuracy_file_url, single_server_accuracy))
        accuracy_list.append(single_server_accuracy)

    print("report >> throughput_list:{} average:{}".format(throughput_list, mean(throughput_list)))
    print("report >> accuracy_list:{} average:{}".format(accuracy_list, mean(accuracy_list)))

    ais_utils.set_result("training", "throughput_ratio", mean(throughput_list))
    ais_utils.set_result("training", "accuracy", mean(accuracy_list))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_code_path", help="the local path of run code")
    parser.add_argument("--single_server_mode", action="store_true", help="the local path of run code")
    parser.add_argument("--action", default="run", choices=["run", "stop"], help="action (run or stop)")
    parser.add_argument("--modelarts_version", default="V1", choices=["V1", "V2"], help="modelarts version (V1 or V2)")
    parser.add_argument("--job_id", default="None",  help="job id used to stop given job")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    logger.setLevel(logging.DEBUG)
    session_config = session_config_v1 if args.modelarts_version == 'V1' else session_config_v2

    handler = modelarts_handler() if args.modelarts_version == 'V1' else modelarts_handler_v2()
    handler.create_session(access_config)

    if args.action == "stop":
        if args.modelarts_version == 'V1':
            handler.stop_new_versions(session_config)
        else:
            handler.stop_job(args.job_id)
        sys.exit()

    handler.create_obs_handler(access_config)

    # default run mode
    handler.run_job(session_config, args.local_code_path)

    # handler.output_url = "s3://0923/00lcm/result_dump/res/V212/"
    try:
        if args.single_server_mode:
            report_result_singlesever_mode(handler, session_config.train_instance_count)
        else:
            report_result(handler)
    except FileNotFoundError as e:
        print("error resport result failed. Exception:", e)
