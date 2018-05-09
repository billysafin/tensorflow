#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from glob import glob

learned_log_dir = "../log/checkpoint"

LABELS = {
    0   : 'A'
    , 1 : 'B'
    , 2 : 'C'
    , 3 : 'D'
}

def get_latest_modified_file_path(dirname, ext=''):
    files = '*' + ext
    target = os.path.join(dirname, files)
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    latest_modified_file_path = sorted(files, key=lambda files: files[1])[-1]
    return latest_modified_file_path[0]
