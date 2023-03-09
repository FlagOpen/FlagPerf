#!/usr/bin/env /usr/bin/python3

"""
1. This script is used to get a summary info for FlagPerf
2. The preferred deployed location is /usr/local/bin/stat.py. 
   In the running-log directory, you type stat.py to get a summary for the running log.
3. output example
    log:./rank0.out.log
    {'converged': True,
     'e2e_time_minutes': 150.2,
     'elapsed_minutes': 150.2,
     'eval_accuracy': {'dist': {0.567799985408783: {'count': 2399,
                                                    'elapsed_minutes': 31.59},
                                0.7555999755859375: {'count': 2400,
                                                     'elapsed_minutes': 31.6},
                                0.7926999926567078: {'count': 2400,
                                                     'elapsed_minutes': 31.59},
                                0.7960000038146973: {'count': 2400,
                                                     'elapsed_minutes': 31.6},
                                0.8014000058174133: {'count': 1,
                                                     'elapsed_minutes': 0}},
                       'max': 0.8014000058174133,
                       'min': 0.567799985408783},
     'global_steps': {'max': 9600, 'min': 1, 'size': 9600},
     'loss': {'max': 8.568405151367188, 'min': 0.003969191107898951, 'size': 9600},
     'loss_scale': {'max': 4294967296, 'min': 512.0, 'size': 9600},
     'lr': {'max': 9.999538958045182e-06, 'min': 0.0, 'size': 9600},
     'num_trained_samples': {'max': 76800, 'min': 8, 'samples_per_second': 8.0},
     'train_finish_at': '2023-03-09 12:09:06',
     'train_start_at': '2023-03-09 09:38:57'}
"""
from pprint import pprint

import time
import json
from argparse import ArgumentParser
from collections import Counter

from datetime import datetime


def format_timestamp(timestamp: int) -> str:
    """format timestamp, return string notation"""
    if timestamp == 0:
        return "N/A"
    date_time = datetime.fromtimestamp(timestamp)
    time_str = date_time.strftime("%Y-%m-%d %H:%M:%S")
    return time_str


def stats(filepath: str) -> dict:
    prf = "[PerfLog] "
    loss = []
    lr = []

    loss_scale = []
    global_steps = []
    num_trained_samples = []
    eval_accuracy = []
    eval_acc_stat = {}  # key: eval_acc, value:[time_ms]

    e2e_time: float = 0
    converged: bool = False
    stat = {}

    train_start_ts: int = 0
    train_finished_at: int = 0
    last_ts = int(time.time())

    with open(filepath) as f:

        for line in f.readlines():

            if not line.startswith("[PerfLog]"):
                continue

            line = line[len(prf):]

            if line.find("FINISHED") > 0:
                obj = json.loads(line)
                last_ts = int(obj.get("metadata").get("time_ms") / 1000)
                train_finished_at = last_ts

            if line.find("INIT_START") > 0:
                obj = json.loads(line)
                train_start_ts = int(obj.get("metadata").get("time_ms") / 1000)

            if line.find("e2e_time") > 0:
                obj = json.loads(line)
                e2e_time = round(float(obj.get("value").get("e2e_time")) / 60, 1)
                converged = True

            if line.find('eval_accuracy') < 0:
                continue

            obj = json.loads(line)
            value = obj.get("value")

            if line.find("e2e_time") > 0:
                e2e_time = float(value.get("e2e_time"))

            loss.append(value.get("loss"))
            lr.append(value.get("learning_rate"))

            if value.get("loss_scale"):
                loss_scale.append(value.get("loss_scale"))

            global_steps.append(value.get("global_steps"))
            num_trained_samples.append(value.get("num_trained_samples"))

            eval_acc_val = value.get("eval_accuracy")

            eval_accuracy.append(eval_acc_val)
            metadata = obj.get("metadata")
            eval_acc_ts = metadata.get("time_ms")
            if eval_acc_val not in eval_acc_stat:
                eval_acc_stat[eval_acc_val] = []
            else:
                eval_acc_stat[eval_acc_val].append(eval_acc_ts)

        eval_accuracy_stat = Counter(eval_accuracy)
        eval_accuracy_stat_with_ts = {}

        for acc, count in eval_accuracy_stat.items():
            eval_accuracy_stat_with_ts[acc] = {"count": count}

            if len(eval_acc_stat[acc]) == 0:
                eval_accuracy_stat_with_ts[acc]["elapsed_minutes"] = 0
            else:
                eval_accuracy_stat_with_ts[acc]["elapsed_minutes"] = round(((max(eval_acc_stat[acc]) - min(
                    eval_acc_stat[acc])) / 1000) / 60, 2)

            stat = {
                "elapsed_minutes": round((int(last_ts) - train_start_ts) / 60, 1),
                "train_start_at": format_timestamp(train_start_ts),
                "train_finish_at": format_timestamp(train_finished_at),
                "converged": converged,
                "e2e_time_minutes": e2e_time,
                "eval_accuracy": {"min": min(eval_accuracy),
                                  "max": max(eval_accuracy),
                                  "dist": eval_accuracy_stat_with_ts},
                "loss": {"min": min(loss), "max": max(loss), "size": len(loss)},
                "lr": {"min": min(lr), "max": max(lr), "size": len(lr)},
                "global_steps": {"min": min(global_steps), "max": max(global_steps), "size": len(global_steps)},
                "num_trained_samples": {"min": min(num_trained_samples),
                                        "max": max(num_trained_samples),
                                        "samples_per_second": round(max(num_trained_samples) / max(global_steps), 1)},

            }

            if len(loss_scale) > 0:
                stat.update({"loss_scale": {"min": min(loss_scale), "max": max(loss_scale), "size": len(loss_scale)}})

    return stat


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, default="./rank0.out.log", help="log file path")
    args = parser.parse_args()
    f = args.file
    t = args.t
    print(f"log:{f}")
    stat = stats(f)
    pprint(stat)
