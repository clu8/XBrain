from tqdm import tqdm


class Logger(object):
    def __init__(self, filename):
        self.f = open(filename, 'w')

    def log_print(self, log):
        tqdm.write(log)
        self.f.write('{}\n'.format(log))

    def close(self):
        self.f.close()
