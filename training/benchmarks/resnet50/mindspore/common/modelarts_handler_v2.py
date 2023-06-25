import logging
import os
import time

from modelarts.estimatorV2 import JOB_STATE, Estimator
from modelarts.session import Session
from modelarts.train_params import InputData, OutputData, TrainingFiles
from obs import ObsClient

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_config_value(config, key):
    return None if config.get(key) == "" else config.get(key)


try:
    import moxing as mox
    moxing_import_flag = True
except Exception:
    moxing_import_flag = False


class modelarts_handler():
    RESP_OK = 300
    OBS_PATH_HEAD = "obs:/"

    def __init__(self):
        self.output_url = None
        self.job_log_prefix = None
        self.job_name = None
        self.job_instance = None
        self.session_config = None
        self.bucket_name = None

    def sync_job_log(self, session_config):
        dstpath = os.path.join(os.getenv("BASE_PATH", "./"), "log")
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        for id in range(session_config.train_instance_count):
            logurl = self.job_log_prefix + '-' + str(id) + '.log'
            logname = os.path.basename(logurl)
            logpath = os.path.join(dstpath, logname)
            if self.session.obs.is_obs_path_exists(logurl):
                self.session.obs.download_file(logurl, logpath)

    def wait_for_job(self):
        count = 0
        while True:
            time.sleep(10)
            count = count + 1
            if count % 10 == 0:
                self.sync_job_log(self.session_config)
            job_info = self.job_instance.get_job_info()

            phase = job_info['status']['phase']
            if phase == "Completed":
                self.sync_job_log(self.session_config)
                logger.info("task succeeded, total time %d(s)" % (job_info['status']['duration'] / 1000))
                break
            elif phase in ['Failed', 'Abnormal', 'Terminated']:
                print("task failed, phase %s, please check log on obs, exit" % (job_info['status']['phase']))
                raise RuntimeError('job failed')
            else:
                print("waiting for task, phase %s, total time: %d(s), actual training time: %d(s) "
                      % (job_info['status']['phase'], 10 * count, job_info['status']['duration'] / 1000))

    def create_obs_output_dirs(self, output_url):
        if moxing_import_flag:
            dstpath = self.OBS_PATH_HEAD + output_url
            logger.info("create obs outdir mox mkdir:{}".format(dstpath))
            mox.file.make_dirs(dstpath)
        else:
            sub_dir = output_url.replace(f"/{self.bucket_name}/", "", 1)
            logger.debug('create obs output{} subdir:{} bucket:{}'.format(output_url, sub_dir, self.bucket_name))
            resp = self.obsClient.putContent(self.bucket_name, sub_dir, content=None)
            if resp.status < self.RESP_OK:
                logger.debug('obs put content request ok')
            else:
                logger.warn('create obs folder failed. errorCode:{} msg:{}'.format(resp.errorCode, resp.errorMessage))
                raise RuntimeError('create obs folder failed')

    def create_obs_handler(self, access_config):
        if not moxing_import_flag:
            # Create OBS login handle
            self.obsClient = ObsClient(access_key_id=access_config.access_key,
                                       secret_access_key=access_config.secret_access_key, server=access_config.server)

    def create_session(self, access_config):
        # 如下配置针对计算中心等专有云 通用云不需要设置
        if access_config.get("iam_endpoint") != "" and access_config.get("iam_endpoint") is not None \
            and access_config.get("obs_endpoint") != "" and access_config.get("obs_endpoint") is not None \
            and access_config.get("modelarts_endpoint") != "" and access_config.get("modelarts_endpoint") is not None:
            Session.set_endpoint(iam_endpoint=access_config.iam_endpoint, obs_endpoint=access_config.obs_endpoint,
                                 modelarts_endpoint=access_config.modelarts_endpoint,
                                 region_name=access_config.region_name)
        # Create modelars handle
        self.session = Session(access_key=access_config.access_key,
                               secret_key=access_config.secret_access_key,
                               project_id=access_config.project_id,
                               region_name=access_config.region_name)

    def print_train_instance_types(self):
        algo_info = Estimator.get_train_instance_types(self.session)
        print("get valid train_instance_types:{}".format(algo_info))

    def stop_job(self, job_id):
        job_info = Estimator.control_job_by_id(session=self.session, job_id=job_id)
        print("job stop status: {}".format(job_info["status"]["phase"]))

    def get_obs_url_content(self, obs_url):
        if moxing_import_flag:
            dsturl = self.OBS_PATH_HEAD + obs_url
            with mox.file.File(dsturl, 'r') as f:
                file_str = f.read()
                return file_str
        else:
            obs_sub_path = obs_url.replace(f"/{self.bucket_name}/", "", 1)
            resp = self.obsClient.getObject(self.bucket_name, obs_sub_path, loadStreamInMemory=True)
            if resp.status < self.RESP_OK:
                logger.debug('request ok')
                return resp.body.buffer.decode("utf-8")
            else:
                raise RuntimeError('obs get object ret:{} url:{} bucket:{} \
                                   path:{}'.format(resp.status, obs_url, self.bucket_name, obs_sub_path))

    def update_code_to_obs(self, localpath):
        if moxing_import_flag:
            dstpath = self.OBS_PATH_HEAD + self.session_config.code_dir
            logger.info("mox update loaclpath:{} dstpath:{}".format(localpath, dstpath))
            mox.file.copy_parallel(localpath, dstpath)
        else:
            sub_dir = "/".join(self.session_config.code_dir.strip("/").split('/')[1:])
            logger.info("update code localpath:{} codepath:{} bucket:{} subdir:{}".format(
                localpath, self.session_config.code_dir, self.bucket_name, sub_dir))
            print("bucket_name:{} sub_dir: {} localpath:{}".format(self.bucket_name, sub_dir, localpath))
            self.obsClient.putFile(self.bucket_name, sub_dir, localpath)

    def create_modelarts_job(self, output_url):
        jobdesc = self.session_config.job_description_prefix + "_jobname_" + self.job_name + "_" +\
            str(self.session_config.train_instance_type) + "_" + str(self.session_config.train_instance_count)

        output_list = [OutputData(obs_path=self.OBS_PATH_HEAD + self.session_config.out_base_url + self.job_name + "/",
                                  name="train_url")]

        estimator = Estimator(session=self.session,
                              framework_type=self.session_config.framework_type,
                              framework_version=self.session_config.framework_version,
                              training_files=TrainingFiles(code_dir=self.OBS_PATH_HEAD + self.session_config.code_dir,
                                                           boot_file=self.OBS_PATH_HEAD + self.session_config.boot_file),
                              log_url=self.OBS_PATH_HEAD + output_url,
                              parameters=self.session_config.parameters,
                              outputs=output_list,
                              pool_id=get_config_value(self.session_config, "pool_id"),
                              train_instance_type=get_config_value(self.session_config, "train_instance_type"),
                              train_instance_count=self.session_config.train_instance_count,
                              job_description=jobdesc,
                              user_command=None)

        logger.debug("new create inputs:{} job_name:{}".format(self.session_config.inputs, self.job_name))
        inut_list = [InputData(obs_path=self.OBS_PATH_HEAD + self.session_config.inputs, name="data_url")]
        try:
            job_instance = estimator.fit(inputs=inut_list, wait=False, job_name=self.job_name)
        except Exception as e:
            logger.error("failed to create job on modelarts, msg %s" % (e))
            raise RuntimeError('creat job failed')

        logger.debug("inputs:{} job_name:{} ret instance:{}".format(inut_list, self.job_name, job_instance))
        job_info = job_instance.get_job_info()
        print("\njob_info: {}\n".format(job_info))

        if 'error_msg' in job_info.keys():
            logger.error("failed to run job on modelarts, error_msg: %s error_code:\
                %s error_solution: %s" % (job_info['error_msg'], job_info['error_code'], job_info['error_solution']))
            raise RuntimeError('creat job failed')

        self.job_log_prefix = self.OBS_PATH_HEAD + output_url + "modelarts-job-" + job_info['metadata']['id'] + '-worker'
        print("create job sucess. job_id:{}  job name:{} create_time:{} job_log_prefix:{}".format(
              job_info["metadata"]["id"],  job_info["metadata"]["name"], job_info["metadata"]["create_time"],
              self.job_log_prefix))

        return job_instance

    def run_job(self, session_config, localpath):
        logger.debug("session config:{}".format(self.session_config))
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
        self.session_config = session_config
        self.job_name = self.session_config.job_name + timestr
        self.print_train_instance_types()
        # modelarts path end with '/'，or report error ModelArts.2791
        self.output_url = os.path.join(self.session_config.out_base_url, self.job_name, "")
        self.bucket_name = self.session_config.out_base_url.split('/')[1]
        logger.debug("output_url:{}".format(self.output_url))
        self.create_obs_output_dirs(self.output_url)

        # update code to obs
        self.update_code_to_obs(localpath)

        self.job_instance = self.create_modelarts_job(self.output_url)
        self.wait_for_job()
