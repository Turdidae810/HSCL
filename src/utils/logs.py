import logging


def set_arg_log(arg):
    logging.info('--------args----------')
    for k in list(vars(arg).keys()):
        logging.info('%s: %s' % (k, vars(arg)[k]))
    logging.info('--------args----------\n')
