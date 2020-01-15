import os
import io
import sys
from pathlib import Path
from datetime import datetime
import traceback

import logging
from tqdm import tqdm
from termcolor import colored
from tensorboardX import SummaryWriter
import numpy as np
import csv

from settings import *


class TqdmHandler(io.StringIO):
    def __init__(self): super().__init__()

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        print(self.buf, end='\r')


class Logger:
    def __init__(self, name, verbose=True, args=None):
        self.args = args
        logger = logging.getLogger(name)
        writer = None
        self.make_dirs()
        if not logger.handlers:
            format = logging.Formatter("[%(name)s|%(levelname)s] %(asctime)s > %(message)s")
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(format)
            logger.addHandler(streamHandler)
            logger.setLevel(args.verbose)

            if verbose:
                filename = os.path.join(self.log_dir, name + '.txt')
                fileHandler = logging.FileHandler(filename, mode="w")
                fileHandler.setFormatter(format)
                logger.addHandler(fileHandler)
                writer = SummaryWriter(self.log_dir)

        self.logger = logger
        self.writer = writer
        sys.excepthook = self.excepthook

    def log(self, msg, lvl="INFO"):
        lvl, color = self.get_level_color(lvl)
        self.logger.log(lvl, colored(msg, color))

    def add_level(self, name, lvl, color='white'):
        if name not in LOG_LEVELS.keys() and lvl not in LOG_LEVELS.values():
            LOG_LEVELS[name] = {'lvl': lvl, 'color': color}
            logging.addLevelName(lvl, name)
        else:
            raise AssertionError("log level already exists")

    def get_level_color(self, lvl):
        assert isinstance(lvl, str)
        lvl_num = LOG_LEVELS[lvl]['lvl']
        color = LOG_LEVELS[lvl]['color']
        return lvl_num, color

    def make_dirs(self):
        self.log_dir = os.path.join(PROJECT_ROOT, LOG_DIR, self.args.tag,
                                    datetime.now().strftime("%Y%m%d%H%M%S"))
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def excepthook(self, type_, value_, traceback_):
        e = "{}: {}".format(type_.__name__, value_)
        tb = "".join(traceback.format_exception(type_, value_, traceback_))
        self.log(e, "ERROR")
        self.log(tb, "DEBUG")

    def scalar_summary(self, info, step, lvl="INFO", filename='values.csv'):
        assert isinstance(info, dict), "data must be a dictionary"
        # flush to terminal
        if self.args.verbose <= LOG_LEVELS[lvl]['lvl']:
            key2str = {}
            for key, val in info.items():
                if isinstance(val, float):
                    valstr = "%-8.3g" %(val,)
                else:
                    valstr = str(val)
                key2str[self._truncate(key)] = self._truncate(valstr)

            if len(key2str) == 0:
                self.log("empty key-value dict", 'WARNING')
                return

            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

            dashes = '  ' + '-'*(keywidth + valwidth + 7)
            lines = [dashes]
            for key, val in key2str.items():
                lines.append('  | %s%s | %s%s |' %(
                    key,
                    ' '*(keywidth - len(key)),
                    val,
                    ' '*(valwidth - len(val))
                ))
            lines.append(dashes)
            print('\n'.join(lines))

        # flush to csv
        if self.log_dir is not None:
            filepath = Path(os.path.join(self.log_dir, filename))
            if not filepath.is_file():
                with open(filepath, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step'] + list(info.keys()))

            with open(filepath, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([step] + list(info.values()))

        # Flush to tensorboard
        if self.writer is not None:
            for k, v in info.items():
                self.writer.add_scalar(k, v, step)

    def progress(self, iterable, length=None):
        if length is None:
            assert hasattr(iterable, '__len__')
            length = len(iterable)
        return tqdm(iterable, total=length, file=TqdmHandler(), ncols=100, ascii=False)

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s
