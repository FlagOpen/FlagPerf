'''Some helper functions for starting tasks in container.'''
import os
CURR_PATH = os.path.abspath(os.path.dirname(__file__))


def _get_model_path(vendor, model_name, framework):
    '''Return the model path according to vendor or None if it doesn't exist.
    '''
    model_path = os.path.join(CURR_PATH + "/../benchmarks/", model_name, framework)
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        return None
    return model_path


def get_config_dir_file(task_args):
    '''Return config path and file path in vendor's dir, or None if the config
       file does not exist.
    '''
    config_file = task_args.extern_config_file
    config_dir = os.path.join(CURR_PATH + "/../" + task_args.vendor + "/",
                              task_args.model_name + "-" + task_args.framework + "/config/")
    config_dir = os.path.abspath(config_dir)
    if not os.path.isfile(os.path.join(config_dir, config_file)):
        return None, None
    return config_dir, config_file


def get_train_script_path(task_args):
    '''Return training script path, or None if it does not exist.'''
    model_path = _get_model_path(task_args.vendor, task_args.model_name, task_args.framework)
    if model_path is None:
        return None
    train_script_path = os.path.join(model_path, task_args.train_script)
    if not os.path.isfile(train_script_path):
        return None
    return train_script_path


def get_extern_module_dir(task_args):
    '''Return extern module dir or None if something wrong.'''
    extern_module_dir = os.path.join(CURR_PATH + "/../" + task_args.vendor,
                                     task_args.model_name + "-" + task_args.framework + "/extern/")
    extern_module_dir = os.path.abspath(extern_module_dir)
    if not os.path.isdir(extern_module_dir):
        return None
    return extern_module_dir


def write_pid_file(pid_file_path, pid_file):
    '''Write pid file for watching the process later.
       In each round of case, we will write the current pid in the same path.
    '''
    pid_file_path = os.path.join(pid_file_path, pid_file)
    if os.path.exists(pid_file_path):
        os.remove(pid_file_path)
    file_d = open(pid_file_path, "w")
    file_d.write("%s\n" % os.getpid())
    file_d.close()
