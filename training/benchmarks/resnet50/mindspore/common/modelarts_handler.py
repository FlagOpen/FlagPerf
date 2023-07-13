from re import S
from urllib.parse import urlparse
from obs import ObsClient, model
from modelarts.session import Session
from modelarts.estimator import JOB_STATE, Estimator
import time
import os

import logging
logging.basicConfig(level = logging.DEBUG,format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_config_value(config, key):
    return None if config.get(key) == "" else config.get(key)

def continue_waiting(job_info):
    print("waiting for task, status %s, total time: %d(s)" % (JOB_STATE[job_info['status']], job_info['duration'] / 1000))

def exit_by_failure(job_info):
    print("task failed, status %s, please check log on obs, exit" % (JOB_STATE[job_info['status']]))
    raise RuntimeError('failed')

func_table = {
    0: continue_waiting,
    1: continue_waiting,
    2: continue_waiting,
    3: exit_by_failure,
    4: continue_waiting,
    5: exit_by_failure,
    6: exit_by_failure,
    7: continue_waiting,
    8: continue_waiting,
    9: exit_by_failure,
    11: exit_by_failure,
    12: exit_by_failure,
    13: exit_by_failure,
    14: exit_by_failure,
    15: continue_waiting,
    16: exit_by_failure,
    17: exit_by_failure,
    18: continue_waiting,
    19: continue_waiting,
    20: continue_waiting,
    21: exit_by_failure,
    22: exit_by_failure
}

# 调试需要 超时后停止
def wait_for_job_timeout(job_instance):
    count = 0
    while True:
        time.sleep(10)
        job_info = job_instance.get_job_info()
        if job_info['status'] == 10:
            print("task succeeded, total time %d(s)" % (job_info['duration'] / 1000))
            break
        func_table[job_info['status']](job_info)
        count = count + 1
        print("modelarts run time count:{}".format(count))
        if count == 6:
            print("modelarts run match:{} 10 so exit >>>>>>>".format(count))
            status = job_instance.stop_job_version()
            #status = job_instance.delete_job()
            raise RuntimeError('failed')
            break

try:
    import moxing as mox
    moxing_import_flag = True
except:
    moxing_import_flag = False

class modelarts_handler():
    def __init__(self):
        self.output_url = None
        self.job_log_prefix = None

    def sync_job_log(self, session_config):
        dstpath = os.path.join(os.getenv("BASE_PATH", "./"), "log")
        if not os.path.exists(dstpath):
            print("dstpath:{} not exist no get log")
            return
        for id in range(session_config.train_instance_count):
            logurl = self.job_log_prefix + '-' + str(id) + '.log'
            logname = os.path.basename(logurl)
            logpath = os.path.join(dstpath, logname)
            if self.session.obs.is_obs_path_exists(logurl):
                self.session.obs.download_file(logurl, logpath)
                #print("logurl:{} sync log to dstpath:{}".format(logurl, logpath))

    def wait_for_job(self, job_instance, session_config):
        count = 0
        while True:
            time.sleep(10)
            count = count + 1
            if count > 10:
                count = 10
                self.sync_job_log(session_config)
            job_info = job_instance.get_job_info()
            if job_info['status'] == 10:
                self.sync_job_log(session_config)
                print("task succeeded, total time %d(s)" % (job_info['duration'] / 1000))
                break
            func_table[job_info['status']](job_info)

    def create_obs_output_dirs(self, output_url):
        if moxing_import_flag == True:
            dstpath = output_url.replace("s3:", "obs:", 1)
            logger.info("create obs outdir mox mkdir:{}".format(dstpath))
            mox.file.make_dirs(dstpath)
        else:
            bucket_name = output_url[5:].split('/')[0]
            sub_dir = output_url.replace(f"s3://{bucket_name}/", "", 1)
            logger.debug('create obs output{} subdir:{} bucket:{}'.format(output_url, sub_dir, bucket_name))
            resp = self.obsClient.putContent(bucket_name, sub_dir, content=None)
            if resp.status < 300:
                logger.debug('obs put content request ok')
            else:
                logger.warn('errorCode:{} msg:{}'.format(resp.errorCode, resp.errorMessage))
                raise RuntimeError('failed')

    def create_obs_handler(self, access_config):
        if moxing_import_flag == False:
            # 创建 obs登录句柄
            self.obsClient = ObsClient(access_key_id=access_config.access_key,
                secret_access_key=access_config.secret_access_key, server=access_config.server)

    def create_session(self, access_config):
        # 如下配置针对计算中心等专有云 通用云不需要设置
        if access_config.get("iam_endpoint") != "" and access_config.get("iam_endpoint") != None \
            and access_config.get("obs_endpoint") != "" and access_config.get("obs_endpoint") != None \
            and access_config.get("modelarts_endpoint") != "" and access_config.get("modelarts_endpoint") != None:
            Session.set_endpoint(iam_endpoint=access_config.iam_endpoint, obs_endpoint=access_config.obs_endpoint, \
                modelarts_endpoint=access_config.modelarts_endpoint, region_name=access_config.region_name)
        # 创建modelarts句柄
        self.session = Session(access_key=access_config.access_key,
            secret_key=access_config.secret_access_key,
            project_id=access_config.project_id,
            region_name=access_config.region_name)

    def print_train_instance_types(self):
        algo_info = Estimator.get_train_instance_types(modelarts_session=self.session)
        print("get valid train_instance_types:{}".format(algo_info))

    def stop_new_versions(self, session_config):
        base_job_list_info = Estimator.get_job_list(modelarts_session=self.session, per_page=10, page=1, order="asc", search_content=session_config.job_name)
        if base_job_list_info == None or base_job_list_info.get("job_total_count", 0) == 0:
            print("find no match version return")
            return
        else:
            pre_version_id = base_job_list_info["jobs"][0].get("version_id")
            job_id = base_job_list_info["jobs"][0].get("job_id")
            job_status = base_job_list_info["jobs"][0].get("status")
            estimator = Estimator(modelarts_session=self.session, job_id=job_id, version_id=pre_version_id)
            if JOB_STATE[job_status] == "JOBSTAT_INIT" \
                or JOB_STATE[job_status] == "JOBSTAT_IMAGE_CREATING" \
                or JOB_STATE[job_status] == "JOBSTAT_SUBMIT_TRYING" \
                or JOB_STATE[job_status] == "JOBSTAT_DEPLOYING" \
                or JOB_STATE[job_status] == "JOBSTAT_WAITING" \
                or JOB_STATE[job_status] == "JOBSTAT_RUNNING":
                status = estimator.stop_job_version()
                print("jobname:{} jobid:{} preversionid:{} jobstatus:{} stop status:{}".format(
                    session_config.job_name, job_id, pre_version_id, JOB_STATE[job_status], status))
            else:
                print("jobname:{} jobid:{} preversionid:{} jobstatus:{} no need stop".format(
                    session_config.job_name, job_id, pre_version_id, JOB_STATE[job_status]))
            return

    def get_job_name_next_new_version(self, session_config):
        base_job_list_info = Estimator.get_job_list(modelarts_session=self.session, per_page=10, page=1, order="asc", search_content=session_config.job_name)
        if base_job_list_info == None or base_job_list_info.get("job_total_count", 0) == 0:
            return 1
        else:
            pre_version_id = base_job_list_info["jobs"][0].get("version_id")
            job_id = base_job_list_info["jobs"][0].get("job_id")
            estimator = Estimator(modelarts_session=self.session, job_id=job_id, version_id=pre_version_id)
            job_info = estimator.get_job_info()
            pre_version_id = job_info.get("version_name", "V0")[1:]
            return int(pre_version_id)+1

    def get_obs_url_content(self, obs_url):
        if moxing_import_flag == True:
            dsturl = obs_url.replace("s3:", "obs:", 1)
            with mox.file.File(dsturl, 'r') as f:
                file_str = f.read()
                return file_str
        else:
            bucket_name = obs_url[5:].split('/')[0]
            obs_sub_path = obs_url.replace(f"s3://{bucket_name}/", "", 1)
            resp = self.obsClient.getObject(bucket_name, obs_sub_path, loadStreamInMemory=True)
            if resp.status < 300: 
                logger.debug('request ok')
                return resp.body.buffer.decode("utf-8")
            else:
                raise RuntimeError('obs get object ret:{} url:{} bucket:{} path:{}'.format(resp.status, obs_url, bucket_name, obs_sub_path))


    def update_code_to_obs(self, session_config, localpath):
        # 待完善 验证
        if moxing_import_flag == True:
            dstpath = "obs:/" + session_config.code_dir
            logger.info("mox update loaclpath:{} dstpath:{}".format(localpath, dstpath))
            mox.file.copy_parallel(localpath, dstpath)
        else:
            bucket_name = session_config.code_dir.split('/')[1]
            sub_dir = "/".join(session_config.code_dir.strip("/").split('/')[1:])
            logger.info("update code localpath:{} codepath:{} bucket:{} subdir:{}".format(
                localpath, session_config.code_dir, bucket_name, sub_dir))
            resp = self.obsClient.putFile(bucket_name, sub_dir, localpath)

    def create_modelarts_job(self, session_config, output_url):
        jobdesc = session_config.job_description_prefix + "_jobname_" + session_config.job_name + "_" + str(session_config.train_instance_type) + "_"  + str(session_config.train_instance_count)
        estimator = Estimator(modelarts_session=self.session,
                            framework_type=session_config.framework_type,
                            framework_version=session_config.framework_version,
                            code_dir=session_config.code_dir,
                            boot_file=session_config.boot_file,
                            log_url=output_url[4:],
                            hyperparameters=session_config.hyperparameters,
                            output_path=output_url[4:],
                            pool_id = get_config_value(session_config, "pool_id"),
                            train_instance_type = get_config_value(session_config, "train_instance_type"),
                            train_instance_count=session_config.train_instance_count,
                            nas_type = get_config_value(session_config, "nas_type"),
                            nas_share_addr = get_config_value(session_config, "nas_share_addr"),
                            nas_mount_path = get_config_value(session_config, "nas_mount_path"),
                            job_description=jobdesc,
                            user_command = None)

        base_job_list_info = Estimator.get_job_list(modelarts_session=self.session, per_page=10, page=1, order="asc", search_content=session_config.job_name)
        if base_job_list_info == None or base_job_list_info.get("job_total_count", 0) == 0:
            logger.debug("new create inputs:{} job_name:{}".format(session_config.inputs, session_config.job_name))
            job_instance = estimator.fit(inputs=session_config.inputs, wait=False, job_name=session_config.job_name)
        else:
            job_id = base_job_list_info["jobs"][0].get("job_id")
            pre_version_id = base_job_list_info["jobs"][0].get("version_id")
            logger.debug("new versions job_id:{} pre_version_id:{}".format(job_id, pre_version_id))
            job_instance = estimator.create_job_version(job_id=job_id, pre_version_id=pre_version_id, inputs=session_config.inputs, wait=False, job_desc=jobdesc)

        print("inputs:{} job_name:{} ret instance:{}".format(session_config.inputs, session_config.job_name, job_instance))
        job_info = job_instance.get_job_info()
        if not job_info['is_success']:
            logger.error("failed to run job on modelarts, msg %s" % (job_info['error_msg']))
            raise RuntimeError('failed')

        self.job_log_prefix = "obs:/" + output_url[4:] + job_info["resource_id"]  + "-job-" + session_config.job_name

        print("create sucess job_id:{} resource_id:{} version_name:{} create_time:{}".format(
            job_info["job_id"], job_info["resource_id"], job_info["version_name"], job_info["create_time"]))
        return job_instance

    def run_job(self, session_config, localpath):
        logger.debug("session config:{}".format(session_config))
    
        self.print_train_instance_types()

        # 获取job_name的next 版本号
        next_version_id = self.get_job_name_next_new_version(session_config)
        # 生成输出路径
        self.output_url = os.path.join("s3:/{}".format(session_config.out_base_url), "V{}".format(next_version_id), "")
        logger.debug("output_url:{}".format(self.output_url))
        self.create_obs_output_dirs(self.output_url)

        # 更新代码到obs上
        self.update_code_to_obs(session_config, localpath)

        job_instance = self.create_modelarts_job(session_config, self.output_url)
        self.wait_for_job(job_instance, session_config)