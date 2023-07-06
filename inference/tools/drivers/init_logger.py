from loguru import logger
import sys


def init_logger(config):
    logger.remove()
    """
    define "EVENTS", using logger.log("EVENT",msg) to log
    #21 means just important than info(#20), less than warning(#30)
    """
    logger.level("Init Begin", no=21)
    logger.level("Init End", no=21)
    logger.level("Export Begin", no=21)
    logger.level("Export End", no=21)
    logger.level("Model Forward Begin", no=21)
    logger.level("Model Forward End", no=21)
    logger.level("Vendor Compile Begin", no=21)
    logger.level("Vendor Compile End", no=21)
    logger.level("Vendor Inference Begin", no=21)
    logger.level("Vendor Inference End", no=21)

    logdir = config.log_dir
    logfile = logdir + "/run_inference.out.log"
    logger.add(logfile, level=config.loglevel)

    logger.add(sys.stdout, level=config.loglevel)
