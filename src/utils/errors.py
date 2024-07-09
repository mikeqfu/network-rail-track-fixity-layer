"""
This module specifies customised errors.
"""


class InvalidSubsectionLength(Exception):
    """
    Indicate invalid subsection length.

    This class is utilised in the method
    :meth:`TrackMovement.split_section()<src.shaft.TrackMovement.split_section>`.
    The error would be raised if the ``unit_length`` exceeds the length of ``section``.
    """
    pass
