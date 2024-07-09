"""
Demonstration of a simple random forest model.
"""

import os

from pyhelpers.settings import pd_preferences

from src.modeller.prototype import TrackMovementEstimator

if __name__ == '__main__':
    pd_preferences(precision=8)

    # Specify element, direction and subsection length over which the average movement is calculated
    element = 'Left Top'
    direction = 'Up'
    subsect_len = 10

    # Create an instance of the estimator for track fixity
    tmp = TrackMovementEstimator(element=element, direction=direction, subsect_len=subsect_len)

    # Integrate data of track movements and influencing factors
    tmp.integrate_data(element=element, direction=direction, subsect_len=subsect_len, verbose=True)

    # Make data sets for creating a random forest (RF) model
    tmp.get_training_test_sets(test_size=0.2, random_state=1)

    # Train an RF model
    tmp.classifier(n_estimators=300, max_depth=15, oob_score=True, n_jobs=os.cpu_count() - 1)

    """
    Mean accuracy: 49.96%

                        Importance
    Curvature               0.3887
    Cant                    0.3725
    Max speed               0.2016
    Underline bridges       0.0095
    Overline bridges        0.0073
    Max axle load           0.0067
    Retaining walls         0.0060
    Tunnels                 0.0058
    Stations                0.0019
    """

    # View the confusion matrix for the trained RF model
    tmp.view_confusion_matrix()

    # Normalise the confusion matrix over the predicted labels
    tmp.view_confusion_matrix(normalise='pred')
