import json
import os
import sys

RESULT_PATH = sys.argv[1]
RANK_SIZE = sys.argv[2]

total_throughput = 0
accuracy = 0
for rank_id in range(int(RANK_SIZE)):
    file_name = "throughput_rank_{}".format(str(rank_id))
    log_path = os.path.join(RESULT_PATH, file_name)
    if not os.path.exists(log_path):
        print("{} file not exist".format(log_path))
    else:
        f = open(log_path, 'r')
        cur_throughput = float(f.read())
        print("{} file throught: {}".format(log_path, cur_throughput))
        total_throughput += cur_throughput
        f.close()

accuracy_file = os.path.join(RESULT_PATH, "eval_acc.log")
if not os.path.exists(accuracy_file):
    print("{} file not exist".format(accuracy_file))
else:
    with open(accuracy_file, 'rb') as fd:
        accuracy = float(fd.read())

print("throughput_ratio:{}".format(total_throughput))
print("accuracy:{}".format(accuracy))

result = {'throughput_ratio': total_throughput, 'accuracy': accuracy}
result_file = os.path.join(RESULT_PATH, "result.log")
with open(result_file, 'w') as f:
    json.dump(result, f)

try:
    import ais_utils
    ais_utils.set_result("training", "throughput_ratio", total_throughput)
    ais_utils.set_result("training", "accuracy", float(accuracy))
    ais_utils.set_result("training", "result", "OK")
except:
    sys.exit()
