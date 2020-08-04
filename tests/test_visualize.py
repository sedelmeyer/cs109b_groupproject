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
        """Ensure save_plot function saves image file"""
        with tempfile.TemporaryDirectory() as tmp:
            plt.plot(self.data["x"], self.data["y"])
            fp = os.path.join(tmp, "test.png")
            caproj.visualize.save_plot(plt_object=plt, savepath=fp)
            self.assertTrue(os.path.exists(fp))

    def test_save_plot_none(self):
        """Ensure save_plot function does not save when savepath is set to None"""
        with tempfile.TemporaryDirectory():
            plt.plot(self.data["x"], self.data["y"])
            caproj.visualize.save_plot(plt_object=plt, savepath=None)
            pass

    def test_set_savepath_return_path_true(self):
        """Ensure set_savepath generates filepath when return_path set to True"""
        dirpath = "testdir/test"
        filename = "testfile.jpg"
        return_path = True
        savepath = caproj.visualize.set_savepath(
            dirpath=dirpath, filename=filename, return_path=return_path
        )
        filepath = os.path.join(dirpath, filename)
        self.assertEqual(filepath, savepath)

    def test_set_savepath_return_path_false(self):
        """Ensure set_savepath returns None when return_path set to False"""
        dirpath = "testdir/test"
        filename = "testfile.jpg"
        return_path = False
        savepath = caproj.visualize.set_savepath(
            dirpath=dirpath, filename=filename, return_path=return_path
        )
        self.assertEqual(None, savepath)
