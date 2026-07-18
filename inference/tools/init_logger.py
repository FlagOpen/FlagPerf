# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from loguru import logger
import sys


def init_logger(config):
    logger.remove()
    """
    define "EVENTS", using logger.log("EVENT",msg) to log
    #21 means just important than info(#20), less than warning(#30)
    Finish Info is more important than error(#40)
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
    logger.level("Finish Info", no=50)

    logdir = config.log_dir
    logfile = logdir + "/container.out.log"
    logger.add(logfile, level=config.loglevel)

    logger.add(sys.stdout, level=config.loglevel)
