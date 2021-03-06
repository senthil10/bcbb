"""Utility functionality for logging.
"""
import os
import sys
import datetime
import logging
import redis
from logbook.queues import RedisHandler
import logbook
from bcbio import utils

LOG_NAME = "nextgen_pipeline"

logger = logging.getLogger(LOG_NAME)
logger2 = logbook.Logger(LOG_NAME)
logger2.level = logbook.INFO


def setup_logging(config):
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        log_dir = config.get("log_dir", None)
        if log_dir:
            logfile = os.path.join(utils.safe_makedir(log_dir),
                                   "{0}.log".format(LOG_NAME))
            handler = logging.FileHandler(logfile)
            handler.setFormatter(formatter)
            logger.addHandler(handler)


def create_log_handler(config, batch_records=False):
    log_dir = config.get("log_dir", None)
    email = config.get("email", None)
    rabbitmq = config.get("rabbitmq_logging", None)
    redis = config.get("redis_handler", None)
    handlers = []

    if log_dir:
        utils.safe_makedir(log_dir)
        handlers.append(logbook.FileHandler(os.path.join(log_dir, "%s.log" % LOG_NAME)))
    else:
        handlers.append(logbook.StreamHandler(sys.stdout))

    if email:
        smtp_host = config.get("smtp_host", None)
        smtp_port = config.get("smtp_port", 25)
        if smtp_host is not None:
            smtp_host = [smtp_host, smtp_port]

        email = email.split(",")
        handlers.append(logbook.MailHandler(email[0], email, server_addr=smtp_host,
                                      format_string=u'''Subject: [BCBB pipeline] {record.extra[run]} \n\n {record.message}''',
                                      level='INFO', bubble=True))
    if rabbitmq:
        from logbook.queues import RabbitMQHandler
        handlers.append(RabbitMQHandler(rabbitmq["url"], queue=rabbitmq["log_queue"], bubble=True))

    if redis:
        try:
            redis_host = config.get('redis_handler').get('host')
            redis_port = int(config.get('redis_handler').get('port'))
            redis_key = config.get('redis_handler').get('key')
            redis_password = config.get('redis_handler').get('password')
            redis_handler = RedisHandler(host=redis_host, port=redis_port,
                key=redis_key, password=redis_password)
            handlers.append(redis_handler)
        except:
            logger2.warn("Failed loading Redis handler, please check your \
                    configuration and the connectivity to your Redis Database")

    if config.get("debug", False):
        for handler in handlers:
            handler.level = logbook.DEBUG
        logger2.level = logbook.DEBUG

    else:
        for handler in handlers:
            handler.level = logbook.INFO

    return logbook.NestedSetup(handlers)


class RecordProgress:
    """A simple interface for recording progress of the parallell
       workflow and outputting timestamp files
    """

    def __init__(self, work_dir, force_overwrite=False):
        self.step = 0
        self.dir = work_dir
        self.fo = force_overwrite

    def tag_progress(self, record):
        record.extra["progress"] = self.step

    def progress(self, action):
        self.step += 1
        self._timestamp_file(action)
        with logbook.Processor(self.tag_progress):
            logger2.info(action)

    def dummy(self):
        with logbook.Processor(self.tag_progress):
            logger2.info("HACK")

    def _action_fname(self, action):
        return os.path.join(self.dir, "{s:02d}_{act}.txt".format(s=self.step, act=action))

    def _timestamp_file(self, action):
        """Write a timestamp to the specified file, either appending or
        overwriting an existing file
        """
        fname = self._action_fname(action)
        mode = "w"
        if utils.file_exists(fname) and not self.fo:
            mode = "a"
        with open(fname, mode) as out_handle:
            out_handle.write("{}\n".format(datetime.datetime.now().isoformat()))
