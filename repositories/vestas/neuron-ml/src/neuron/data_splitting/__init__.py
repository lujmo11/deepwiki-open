"""Module containing classes for splitting data for training and evaluation.

All concrete evaluators should implement the DataSplitter interface.

The DataSplitter used in a training run is chosen at runtime based on the configuration using the
`get_data_splitter` function
"""
