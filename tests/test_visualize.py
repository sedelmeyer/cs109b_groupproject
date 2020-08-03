import os
import unittest
import tempfile

import pandas as pd
import matplotlib.pyplot as plt

import caproj.visualize


class TestPlots(unittest.TestCase):
    """Test basic plotting functions"""

    def setUp(self):
        """Set up data for tests"""
        self.data = pd.DataFrame({"x": [0, 1, 2, 3], "y": [2, 3, 4, 5]})

    def test_save_plot(self):
        """Test save_plot function"""
        with tempfile.TemporaryDirectory() as tmp:
            plt.plot(self.data["x"], self.data["y"])
            fp = os.path.join(tmp, "test.png")
            caproj.visualize.save_plot(plt_object=plt, savepath=fp)
            self.assertTrue(os.path.exists(fp))

    def test_save_plot_none(self):
        """Test save_plot function passes with no savepath"""
        with tempfile.TemporaryDirectory():
            plt.plot(self.data["x"], self.data["y"])
            caproj.visualize.save_plot(plt_object=plt, savepath=None)
            pass
