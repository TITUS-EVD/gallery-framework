import ROOT
import sys
import os

from root_numpy import root2array
import pandas
import numpy
from matplotlib import pyplot as plt


samples = dict()
samples.update(
    {'nue': "/data/argoneut/larsoft/nue_cc_sim/nue_cc_stage0_larlite_ana.root"})
samples.update(
    {'numu': "/data/argoneut/larsoft/numu_cc_sim/numu_stage0_larlite_ana.root"})
samples.update(
    {'xing': "/data/argoneut/larsoft/xing_muons/xing_muons_stage0_larlite_ana.root"})
samples.update(
    {'anu_sim': "/data/argoneut/larsoft/anu_sim/anu_sim_stage0_larlite_ana.root"})

colors = dict()
colors.update(
    {'nue': 'blue', 'numu': "red", 'xing': 'green', "anu_sim": 'black'})

labels = dict()
labels.update({'nue': 'CC Electrons',
               'numu': "CC Muons", 'xing': 'Crossing Muons', "anu_sim": 'Anti Nu Sim'})


def main():

    data_arrays = dict()

    for _key in samples:
        _file = samples[_key]

        # Open the file and get the histograms needed

        a = pandas.DataFrame(root2array(_file))
        data_arrays.update({_key: a})

    # plot_n_tracks(data_arrays)
    # plot_n_clusters(data_arrays)
    # plot_n_vertexes(data_arrays)
    # plot_n_long_tracks(data_arrays)
    plot_percent_hits_of_longest_track(data_arrays)


def plot_n_tracks(data_arrays):

    bins = list(numpy.arange(0, 10, 1))
    bins.append(100)
    bins = numpy.asarray(bins)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    # Make histograms of the data
    fig, ax = plt.subplots(figsize=(10, 7))

    for _key in data_arrays:

        n_tracks, bin_edges = numpy.histogram(data_arrays[_key]['_n_tracks'],
                                              bins, density=False)

        N = sum(n_tracks)
        n_tracks = (1.0*n_tracks) / N
        # n_tracks = list(n_tracks)
        x_centers = range(0, len(n_tracks))

        n_tracks = numpy.append(n_tracks, 0)
        x_centers.append(10)

        ax.plot(x_centers, 100*n_tracks, color=colors[_key],
                ls='steps-post',
                label=labels[_key], linewidth=3)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    plt.ylim([0, 100])

    plt.xlabel("Number of Tracks", fontsize=20)
    plt.ylabel("Percent of Events", fontsize=20)
    plt.legend(fontsize=20)

    plt.grid(True)
    plt.show()


def plot_n_long_tracks(data_arrays):

    bins = list(numpy.arange(0, 10, 1))
    bins.append(100)
    bins = numpy.asarray(bins)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    # Make histograms of the data
    fig, ax = plt.subplots(figsize=(10, 7))

    for _key in data_arrays:

        n_long_tracks, bin_edges = numpy.histogram(data_arrays[_key]['_n_long_tracks'],
                                              bins, density=False)

        N = sum(n_long_tracks)
        n_long_tracks = (1.0*n_long_tracks) / N
        # n_long_tracks = list(n_long_tracks)
        x_centers = range(0, len(n_long_tracks))

        n_long_tracks = numpy.append(n_long_tracks, 0)
        x_centers.append(10)

        ax.plot(x_centers, 100*n_long_tracks, color=colors[_key],
                ls='steps-post',
                label=labels[_key], linewidth=3)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    plt.ylim([0, 100])

    plt.xlabel("Number of Tracks > 15cm", fontsize=20)
    plt.ylabel("Percent of Events", fontsize=20)
    plt.legend(fontsize=20)

    plt.grid(True)
    plt.show()

def plot_percent_hits_of_longest_track(data_arrays):

    bins = list(numpy.arange(0, 1.01, 0.05))
    bins = numpy.asarray(bins)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    # Make histograms of the data
    fig, ax = plt.subplots(figsize=(10, 7))

    for _key in data_arrays:

        n_long_tracks, bin_edges = numpy.histogram(data_arrays[_key]['_percent_hits_of_longest_track'],
                                              bins, density=False)

        N = sum(n_long_tracks)
        n_long_tracks = (1.0*n_long_tracks) / N
        # n_long_tracks = list(n_long_tracks)
        x_centers = range(0, len(n_long_tracks))

        n_long_tracks = numpy.append(n_long_tracks, 0)
        x_centers.append(1.01)

        ax.plot(bin_edges, 100*n_long_tracks, color=colors[_key],
                ls='steps-post',
                label=labels[_key], linewidth=3)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    plt.ylim([0, 100])

    plt.xlabel("Percentage of Hits ass'd with longest track", fontsize=20)
    plt.ylabel("Percent of Events", fontsize=20)
    plt.legend(fontsize=20)

    plt.grid(True)
    plt.show()


def plot_n_vertexes(data_arrays):

    bins = list(numpy.arange(0, 10, 1))
    bins.append(100)
    bins = numpy.asarray(bins)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    # Make histograms of the data
    fig, ax = plt.subplots(figsize=(10, 7))

    for _key in data_arrays:

        n_vertexes, bin_edges = numpy.histogram(data_arrays[_key]['_n_vertexes'],
                                              bins, density=False)

        N = sum(n_vertexes)
        n_vertexes = (1.0*n_vertexes) / N
        # n_vertexes = list(n_vertexes)
        x_centers = range(0, len(n_vertexes))

        n_vertexes = numpy.append(n_vertexes, 0)
        x_centers.append(10)

        ax.plot(x_centers, 100*n_vertexes, color=colors[_key],
                ls='steps-post',
                label=labels[_key], linewidth=3)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    plt.ylim([0, 100])

    plt.xlabel("Number of Vertexes", fontsize=20)
    plt.ylabel("Percent of Events", fontsize=20)
    plt.legend(fontsize=20)

    plt.grid(True)
    plt.show()


def plot_n_clusters(data_arrays):

    bins = list(numpy.arange(0, 10, 1))
    bins.append(100)
    bins = numpy.asarray(bins)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    # Make histograms of the data
    fig, ax = plt.subplots(figsize=(10, 7))

    for _key in data_arrays:

        n_clusters, bin_edges = numpy.histogram(
            data_arrays[_key]['_n_clusters'],
            bins, density=False)

        N = sum(n_clusters)
        n_clusters = (1.0*n_clusters) / N
        # n_clusters = list(n_clusters)
        x_centers = range(0, len(n_clusters))

        n_clusters = numpy.append(n_clusters, 0)
        x_centers.append(10)

        ax.plot(x_centers, 100*n_clusters, color=colors[_key],
                ls='steps-post',
                label=labels[_key], linewidth=3)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    plt.ylim([0, 100])

    plt.xlabel("Number of Clusters", fontsize=20)
    plt.ylabel("Percent of Events", fontsize=20)
    plt.legend(fontsize=20)

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
