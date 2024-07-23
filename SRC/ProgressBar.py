# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:59:14 2023
Edited on Mon Jul 22 2024
@author: DEscobar-Salce
"""

import time

class ProgressBar:
    """
    A class to display a progress bar with elapsed and remaining time information in the console.

    Attributes:
        bar_length (int): Length of the progress bar (default is 50 characters).
        start_time (float): Time when the progress bar starts, initialized to None.
    """

    def __init__(self, bar_length=50):
        """
        Initialize the ProgressBar class with the specified bar length.

        Parameters:
            bar_length (int): Length of the progress bar (default is 50 characters).
        """
        self.bar_length = bar_length
        self.start_time = None

    def display(self, iteration, total):
        """
        Display a progress bar with time information in the console.

        Parameters:
            iteration (int): Current iteration (0-indexed).
            total (int): Total iterations.
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        # Calculate progress and the bar display
        progress = (iteration + 1) / total
        arrow = '=' * int(round(progress * self.bar_length) - 1) + '>'
        spaces = ' ' * (self.bar_length - len(arrow))
        
        # Calculate elapsed time and estimated remaining time
        elapsed_time = time.time() - self.start_time
        remaining_time = (elapsed_time / progress) * (1 - progress)
        
        # Convert elapsed and remaining time to minutes and seconds
        elapsed_min, elapsed_sec = divmod(elapsed_time, 60)
        remaining_min, remaining_sec = divmod(remaining_time, 60)
        
        # Print the progress bar with time information
        print('\r[{0}] {1}% - Elapsed: {2:.0f}m{3:.0f}s - Remaining: {4:.0f}m{5:.0f}s'.format(
            arrow + spaces, 
            int(round(progress * 100)),
            elapsed_min,
            elapsed_sec,
            remaining_min,
            remaining_sec
        ), end='')
