# src/visualization/stats_visualizer.py
import matplotlib.pyplot as plt

class StatsVisualizer:
    def visualize(self, stats):
        plt.bar(stats.keys(), stats.values())
        plt.savefig('data/outputs/stats.png')