#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from kiem_tra_ho_so import main

if __name__ == "__main__":
    main()
