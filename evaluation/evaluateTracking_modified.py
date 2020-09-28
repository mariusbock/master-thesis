# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""
Compute metrics for trackers using MOTChallenge ground-truth data. Added slight modifications to this file
to fit workflow (compare to original evaluateTracking.py).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import io
import logging
import os
import sys
from tempfile import NamedTemporaryFile
import time

import motmetrics as mm


def compare_dataframes(gts, ts, vsflag='', iou=0.5):
    """Builds accumulator for each sequence."""
    accs = []
    anas = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Evaluating %s...', k)
            if vsflag != '':
                fd = io.open(vsflag + '/' + k + '.log', 'w')
            else:
                fd = ''
            acc, ana = mm.utils.CLEAR_MOT_M(gts[k][0], tsacc, gts[k][1], 'iou', distth=iou, vflag=fd)
            if fd != '':
                fd.close()
            accs.append(acc)
            anas.append(ana)
            names.append(k)
        else:
            logging.warning('No ground truth for %s, skipping.', k)

    return accs, anas, names


def parseSequences(seqmap):
    """Loads list of sequences from file."""
    assert os.path.isfile(seqmap), 'Seqmap %s not found.' % seqmap
    fd = io.open(seqmap)
    res = []
    for row in fd.readlines():
        row = row.strip()
        if row == '' or row == 'name' or row[0] == '#':
            continue
        res.append(row)
    fd.close()
    return res


def parseEvals(evalmap):
    """Loads list of eval files from file."""
    assert os.path.isfile(evalmap), 'Seqmap %s not found.' % evalmap
    fd = io.open(evalmap)
    res = []
    for row in fd.readlines():
        row = row.strip()
        if row == '' or row == 'name' or row[0] == '#':
            continue
        res.append(row)
    fd.close()
    return res


def generateSkippedGT(gtfile, skip, fmt):
    """Generates temporary ground-truth file with some frames skipped."""
    del fmt  # unused
    tf = NamedTemporaryFile(delete=False, mode='w')
    with io.open(gtfile) as fd:
        lines = fd.readlines()
        for line in lines:
            arr = line.strip().split(',')
            fr = int(arr[0])
            if fr % (skip + 1) != 1:
                continue
            pos = line.find(',')
            newline = str(fr // (skip + 1) + 1) + line[pos:]
            tf.write(newline)
    tf.close()
    tempfile = tf.name
    return tempfile


def evaluateTracking_modGT(files_path, seqinfo_path, seqmap, log='', loglevel='info', fmt='mot15-2D', solver='', skip=0, iou=0.5):
    """
    Main function that was slightly modified to take in different parameters and ground truth file suiting project's
    data layout and returns summary object for further processing
    """

    # pylint: disable=missing-function-docstring
    # pylint: disable=too-many-locals

    loglevel = getattr(logging, loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if not solver == '':
        mm.lap.default_solver = solver

    seqs = parseSequences(seqmap)

    gtfiles = [os.path.join(files_path, i, 'gt.txt') for i in seqs]
    tsfiles = [os.path.join(files_path, i, '%s.txt' % i) for i in seqs]

    for gtfile in gtfiles:
        if not os.path.isfile(gtfile):
            logging.error('gt File %s not found.', gtfile)
            sys.exit(1)
    for tsfile in tsfiles:
        if not os.path.isfile(tsfile):
            logging.error('res File %s not found.', tsfile)
            sys.exit(1)

    logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
    for seq in seqs:
        logging.info('\t%s', seq)
    logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
    logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
    logging.info('Loading files.')

    if skip > 0 and 'mot' in fmt:
        for i, gtfile in enumerate(gtfiles):
            gtfiles[i] = generateSkippedGT(gtfile, skip, fmt=fmt)

    gt = OrderedDict(
        [(f.split('/')[-2].split('_')[0], (mm.io.loadtxt(f, fmt=fmt), os.path.join(seqinfo_path, f.split('/')[-2].split('_')[0], 'seqinfo.ini'))) for i, f in
         enumerate(gtfiles)])
    ts = OrderedDict([(f.split('/')[-1].split('.')[0], mm.io.loadtxt(f, fmt=fmt)) for i, f in enumerate(tsfiles)])

    mh = mm.metrics.create()
    st = time.time()
    accs, analysis, names = compare_dataframes(gt, ts, log, 1. - iou)
    logging.info('adding frames: %.3f seconds.', time.time() - st)

    logging.info('Running metrics')

    summary = mh.compute_many(accs, anas=analysis, names=names, metrics=mm.metrics.motchallenge_metrics,
                              generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')
    return summary


def evaluateTracking(pred_path, data_path, seqmap, log='', loglevel='info', fmt='mot15-2D', solver='', skip=0, iou=0.5):
    """
    Main function that was slightly modified to take in different parameters suiting project's data layout
    and returns summary object for further processing
    """
    # pylint: disable=missing-function-docstring
    # pylint: disable=too-many-locals

    loglevel = getattr(logging, loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if not solver == '':
        mm.lap.default_solver = solver

    seqs = parseSequences(seqmap)

    gtfiles = [os.path.join(data_path, i, 'gt/gt.txt') for i in seqs]
    tsfiles = [os.path.join(pred_path, i, '%s.txt' % i) for i in seqs]

    for gtfile in gtfiles:
        if not os.path.isfile(gtfile):
            logging.error('gt File %s not found.', gtfile)
            sys.exit(1)
    for tsfile in tsfiles:
        if not os.path.isfile(tsfile):
            logging.error('res File %s not found.', tsfile)
            sys.exit(1)

    logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
    for seq in seqs:
        logging.info('\t%s', seq)
    logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
    logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
    logging.info('Loading files.')

    if skip > 0 and 'mot' in fmt:
        for i, gtfile in enumerate(gtfiles):
            gtfiles[i] = generateSkippedGT(gtfile, skip, fmt=fmt)

    gt = OrderedDict(
        [(f.split('/')[-3].split('_')[0], (mm.io.loadtxt(f, fmt=fmt), os.path.join(data_path, f.split('/')[-3].split('_')[0], 'seqinfo.ini'))) for i, f in
         enumerate(gtfiles)])
    ts = OrderedDict([(f.split('/')[-1].split('.')[0], mm.io.loadtxt(f, fmt=fmt)) for i, f in enumerate(tsfiles)])

    mh = mm.metrics.create()
    st = time.time()
    accs, analysis, names = compare_dataframes(gt, ts, log, 1. - iou)
    logging.info('adding frames: %.3f seconds.', time.time() - st)

    logging.info('Running metrics')

    summary = mh.compute_many(accs, anas=analysis, names=names, metrics=mm.metrics.motchallenge_metrics,
                              generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')
    return summary
