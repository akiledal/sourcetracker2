#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
# www.biota.com
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

#Fixes "Invalid DISPLAY variable" error
plt.switch_backend('Agg')

def plot_heatmap(mpm, cm=plt.cm.viridis, xlabel='Sources', ylabel='Sinks',
                 title='Mixing Proportions (as Fraction)'):
    '''Make a basic mixing proportion histogram.'''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(mpm, vmin=0, vmax=1.0, cmap=cm, annot=True, linewidths=.5,
                ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax
