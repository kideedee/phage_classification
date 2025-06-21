from logger.phg_cls_log import experiment_log as log


def start_experiment(experiment_name, timestamp):
    log.info("========================================================START========================================================")
    log.info(f"Experiment name: {experiment_name}")
    log.info(f"Experiment timestamp: {timestamp}")