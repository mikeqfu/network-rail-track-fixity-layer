"""
This module provides a prototype machine-learning-based probabilistic tool
for predicting track movement.
"""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from pyhelpers._cache import _print_failure_msg
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.shaft.feature import FeatureCollator
from src.utils.general import paired_next, save_plot


class TrackMovementEstimator(FeatureCollator):
    """
    Prototype machine-learning-based probabilistic model for predicting track movement.

    This class implements a prototype model designed to predict track movement and analyse
    track movement using the `random forest <https://en.wikipedia.org/wiki/Random_forest>`_
    algorithm. The model is trained and tested on a data set processed by the
    :mod:`~src.shaft` module, where the :class:`~src.shaft.movement.TrackMovement` class calculates
    the track movement (serving as the target variable in the model) and the
    :class:`~src.shaft.feature.FeatureCollator` class gathers information of relevant
    explanatory variables (forming the feature data set for the model).
    """

    #: The names of explanatory variables (a.k.a. features) for model specification.
    FEATURE_NAMES: list = [
        'Curvature',
        'Cant',
        'Max speed',
        'Max axle load',
        'Overline bridges',
        'Underline bridges',
        'Retaining walls',
        'Tunnels',
        'Stations']
    
    #: The name of the target variable.
    TARGET_NAME: str = 'lateral_displacement_mean'

    def __init__(self, element, direction, subsect_len=10, elr=None, target_name=None,
                 feature_names=None, db_instance=None, random_state=0, verbose=True):
        # noinspection PyShadowingNames
        """
        :param element: Element of rail head, such as left or right top of rail, or running edge.
        :type element: str
        :param direction: Railway direction, such as up or down direction.
        :type direction: str
        :param subsect_len: Length (in metres) of a subsection for which movement is calculated;
            defaults to ``10``.
        :type subsect_len: int
        :param elr: Engineer's Line Reference(s); defaults to ``None``.
        :type elr: str | list | None
        :param target_name: Name of the target variable;
            when ``target_name=None`` (default), it takes the class attribute
            :py:attr:`~src.modeller.prototype.TrackMovementEstimator.TARGET_NAME`.
        :type target_name: str | None
        :param feature_names: Names of explanatory variables (features) for model specification;
            defaults to ``None``.
        :type feature_names: list | None
        :param db_instance: PostgreSQL database instance; defaults to ``None``.
        :type db_instance: TrackFixityDB | None
        :param random_state: A random seed number; defaults to ``0``.
        :type random_state: int | None
        :param verbose: Whether to print relevant information to the console; defaults to ``True``.
        :type verbose: bool | int

        :ivar pandas.DataFrame | None data_set: The data set for developing the prototype
            machine learning model for estimating/predicting track movement; defaults to ``None``.
        :ivar list valid_target_names: List of valid target variable names.
        :ivar str target_name: The name of the target variable.
        :ivar numpy.ndarray | None target_edges: Critical values based on which the target data is
            categorised into different classes; defaults to ``None``.
        :ivar LabelEncoder label_encoder: An encoder that labels the target data.
        :ivar pandas.Series target_data: Data of track movement to be predicted;
            defaults to ``None``.
        :ivar pandas.Series target_data_labels: Labelled ``target_data``; defaults to ``None``.
        :ivar list labels: Descriptive texts for ``target_data_labels``; defaults to ``None``.
        :ivar list feature_names: Names of explanatory variables (features) for model specification.
        :ivar pandas.DataFrame feature_data: Data of features collated to predict the
            ``target_data_labels``.
        :ivar int | None random_state: A random seed number.

        :ivar pandas.DataFrame | None X_train: Feature data of a training set; defaults to ``None``.
        :ivar pandas.DataFrame | None y_train: Target data of a training set; defaults to ``None``.
        :ivar pandas.Series | None X_test: Feature data of a test set; defaults to ``None``.
        :ivar pandas.Series | None y_test: Target data of a test set; defaults to ``None``.

        :ivar RandomForestClassifier estimator: The prototype random forest model;
            defaults to ``None``.
        :ivar RandomForestClassifier best_estimator: The best random forest model
            (if a model selection procedure is implemented); defaults to ``None``.
        :ivar pandas.DataFrame best_estimator_params:
            The parameters of the best random forest model; defaults to ``None``.
        :ivar float score: The mean accuracy of the ``estimator`` making predictions
            on a given test data; defaults to ``None``.
        :ivar pandas.DataFrame feature_importance: Relative importance of different features
            based on a trained model; defaults to ``None``.
        :ivar pandas.DataFrame confusion_matrix: Relative importance of different features
            based on a trained model; defaults to ``None``.

        **Examples**::

            >>> from src.modeller.prototype import TrackMovementEstimator
            >>> tme = TrackMovementEstimator(element='Left Top', direction='Up', subsect_len=10)
            Initialising the feature collator ... Done.
            Initialising the estimator ... Done.
            >>> tme.target_name
            'lateral_displacement_mean'
            >>> tme.feature_names
            ['Curvature',
             'Cant',
             'Max speed',
             'Max axle load',
             'Overline bridges',
             'Underline bridges',
             'Retaining walls',
             'Tunnels',
             'Stations']
        """

        super().__init__(db_instance=db_instance, elr=elr, verbose=verbose)

        if verbose:
            print("Initialising the estimator", end=" ... ")

        try:
            self.load_movement(element=element, direction=direction, subsect_len=subsect_len)

            self.element = element
            self.direction = direction
            self.subsect_len = subsect_len

            self.data_set = None

            if target_name is None:
                self.target_name = self.TARGET_NAME
            else:
                valid_target_names = [
                    y for y in list(self.track_movement.values())[0].columns if 'displacement' in y]
                assert target_name in valid_target_names
                self.target_name = target_name

            self.target_edges = None
            self.label_encoder = LabelEncoder()
            self.target_data = None
            self.target_data_ = None
            self.target_data_labels = None

            self.labels = []

            self.feature_names = self.FEATURE_NAMES if feature_names is None else feature_names

            self.feature_data = None

            self.random_state = random_state

            self.X_train = None
            self.y_train = None
            self.X_test = None
            self.y_test = None

            self.y_test_pred = None

            self.estimator = None
            self.best_estimator = None
            self.best_estimator_params = None
            self.score = None
            self.feature_importance = None

            self.confusion_matrix = None

            if verbose:
                print("Done.")

        except Exception as e:
            _print_failure_msg(e)

    def label_target(self, target_data, percentiles=None, as_dataframe=False, ret_dat=False,
                     **kwargs):
        # noinspection PyShadowingNames
        """
        Categorise the model target (e.g. lateral displacement of the top of a rail) into classes.

        This method classifies the target data into six categories for use in a random forest model,
        which handles categorical data. The target data is divided into ranges based on the
        specified percentiles and labelled with values from 0 to 5. By default, the boundaries of
        these intervals are determined by the 10th, 25th, 50th, 75th and 95th percentiles.

        :param percentiles: Percentiles used to calculate the descriptive statistics of
            ``target_data``; defaults to ``[.25, .5, .75]`` if ``None``.
        :type percentiles: list[float] | None
        :param target_data: The continuous target data to be categorised
            (e.g. lateral displacement of the top of a rail).
        :type target_data: pandas.Series
        :param as_dataframe: Whether the labelled target is formatted as a DataFrame;
            defaults to ``False``.
        :type as_dataframe: bool
        :param ret_dat: If ``True``, returns the processed data; defaults to ``False``.
        :type ret_dat: bool
        :param kwargs: [Optional] additional parameters for the `pandas.cut()`_ function,
            excluding ``include_lowest``, ``duplicates`` and ``retbins``.
        :return: Encoded target data as a pandas Series or DataFrame.
        :rtype: pandas.Series | pandas.DataFrame

        .. _`pandas.DataFrame.describe()`:
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
        .. _`pandas.cut()`:
            https://pandas.pydata.org/docs/reference/api/pandas.cut.html

        **Examples**::

            >>> from modeller import TrackMovementEstimator
            >>> element = 'Left Top'
            >>> direction = 'Up'
            >>> subsect_len = 10
            >>> tme = TrackMovementEstimator(element, direction, subsect_len)
            Initialising the feature collator ... Done.
            Initialising the estimator ... Done.
            >>> tme.integrate_data(element, direction, subsect_len, verbose=True)
            Calculating track movement ... Done.
            Collating features:
              ballasts (src. Ballast summary) ... Done.
              structures (src. CARRS and OPAS) ... Done.
            Finalising the data integration ... Done.
            >>> tme.track_movement_.head()
                                                      subsection  ...  Stations
            0  LINESTRING Z (399428.96 653473.9 34.442, 39942...  ...         0
            1  LINESTRING Z (399434.5201166851 653482.2116640...  ...         0
            2  LINESTRING Z (399440.0608295194 653490.5362699...  ...         0
            3  LINESTRING Z (399445.5700898784 653498.8817273...  ...         0
            4  LINESTRING Z (399451.0456262118 653507.2493444...  ...         0
            [5 rows x 46 columns]
            >>> tme.target_name
            'lateral_displacement_mean'
            >>> target_data = tme.track_movement_[tme.target_name]
            >>> target_data.head()
            0   -0.002879
            1   -0.002754
            2   -0.001841
            3   -0.002032
            4   -0.003821
            Name: lateral_displacement_mean, dtype: float64
            >>> tme.label_target(target_data)
            >>> tme.target_data_labels.head()
            0    2
            1    2
            2    3
            3    3
            4    1
            Name: lateral_displacement_mean, dtype: int64
            >>> tme.get_descriptive_labels()
            >>> tme.labels
            ['≤ -4.48 mm',
             '(-4.48, -3.5] mm',
             '(-3.5, -2.51] mm',
             '(-2.51, 0.0] mm',
             '> 0.0 mm']
        """

        if percentiles is None:
            percentiles_ = [.25, .5, .75]  # [.10, .25, .5, .75, .95]
        else:
            percentiles_ = copy.copy(percentiles)
        stats = target_data.describe(percentiles=percentiles_)
        stats_pctl = stats.loc[['min'] + ["{:.0%}".format(x) for x in percentiles_] + ['max']]

        if 0 not in stats_pctl.values:
            # stats_pctl = stats_pctl.append(pd.Series(0.0, index=['0'])).sort_values()
            stats_pctl = pd.concat([stats_pctl, pd.Series(0.0, index=['0'])], axis=0).sort_values()

        self.target_data_, self.target_edges = pd.cut(
            target_data, bins=stats_pctl.values, include_lowest=True, duplicates='drop',
            retbins=True, **kwargs)

        movement_classes = self.label_encoder.fit_transform(self.target_data_)

        if as_dataframe:
            target_data_labels = pd.DataFrame(movement_classes, columns=[self.target_name])
        else:
            target_data_labels = pd.Series(movement_classes, name=self.target_name)

        self.target_data_labels = target_data_labels

        if ret_dat:
            return target_data_labels

    def get_descriptive_labels(self, ret_dat=False):
        """
        Retrieve descriptive texts for the numeric target labels.

        This method provides human-readable descriptions for the numeric labels used in the
        classification of track movement.

        :param ret_dat: If ``True``, returns the processed data; defaults to ``False``.
        :type ret_dat: bool
        :return: Descriptive class labels for the calculated track movement, if ``ret_dat=True``.
        :rtype: list

        .. seealso::

            - Examples for the method
              :meth:`~src.modeller.prototype.TrackMovementEstimator.label_target`.
        """

        self.labels = []

        # noinspection PyTypeChecker
        target_edges = list(paired_next(np.round(self.target_edges * 1000, 2)))

        for i, x in enumerate(target_edges):
            if i == 0:
                self.labels.append(f'≤ {x[1]} mm')
            elif i == len(target_edges) - 1:
                self.labels.append(f'> {x[0]} mm')
            else:
                self.labels.append(f'({x[0]}, {x[1]}] mm')

        # cls_labels_dict = dict(zip(range(len(self.labels)), self.labels))

        if ret_dat:
            return self.labels

    def get_training_test_sets(self, random_state=None, ret_dat=False, verbose=True, **kwargs):
        """
        Create and return training and test data sets.

        This method splits the data set into training and test sets for model development and
        evaluation.

        :param random_state: A random seed number for reproducibility; defaults to ``None``.
        :type random_state: int | None
        :param ret_dat: If ``True``, returns the created data sets; defaults to ``False``.
        :type ret_dat: bool
        :param verbose: Whether to print relevant information in console; defaults to ``True``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the
            `sklearn.model_selection.train_test_split`_ function.
        :return: Feature and target data of both the training and test set.
        :rtype: tuple[pandas.DataFrame, pandas.Series, pandas.DataFrame, pandas.Series]

        .. _`sklearn.model_selection.train_test_split`:
            https://scikit-learn.org/stable/modules/generated/
            sklearn.model_selection.train_test_split.html

        **Examples**:

            >>> from src.modeller import TrackMovementEstimator
            >>> element = 'Left Top'
            >>> direction = 'Up'
            >>> subsect_len = 10
            >>> tme = TrackMovementEstimator(element, direction, subsect_len)
            Initialising the feature collator ... Done.
            Initialising the estimator ... Done.
            >>> tme.get_training_test_sets()
            Splitting into training and test sets ... Done.
            The data is now ready for modelling.
            >>> tme.X_train.head()
                  Curvature    Cant  Max speed  ...  Retaining walls  Tunnels  Stations
            67    -0.000849  -73.00       75.0  ...                0        0         0
            7626   0.000000    0.00      125.0  ...                0        0         0
            5554  -0.000675 -140.00      100.0  ...                0        0         0
            4226  -0.000163  -42.83      105.0  ...                0        0         0
            350    0.000107   13.28       90.0  ...                0        0         0
            [5 rows x 9 columns]
            >>> tme.y_train.head()
            59      3
            6788    0
            4897    3
            3631    3
            342     4
            Name: lateral_displacement_mean, dtype: int64
            >>> tme.X_test.head()
                  Curvature    Cant  Max speed  ...  Retaining walls  Tunnels  Stations
            5845  -0.000529 -134.48      110.0  ...                0        0         0
            178    0.000000    0.00       95.0  ...                0        0         0
            5249   0.000000    0.00      110.0  ...                0        0         0
            2472  -0.001412 -125.00       75.0  ...                0        0         0
            1239   0.001314  134.00       80.0  ...                0        0         0
            [5 rows x 9 columns]
            >>> tme.y_test.head()
            5133    3
            170     2
            4622    2
            2031    2
            1082    2
            Name: lateral_displacement_mean, dtype: int64
        """

        if random_state is not None:
            self.random_state = random_state

        if self.track_movement_ is None:
            self.integrate_data(
                element=self.element, direction=self.direction, subsect_len=self.subsect_len,
                ret_dat=False, verbose=verbose)

        if verbose:
            print("Splitting into training and test sets", end=" ... ")

        try:
            # feature_names, target_name = map(copy.copy, [self.feature_names, self.target_name])
            self.data_set = self.track_movement_[self.feature_names + [self.target_name]].dropna()

            self.feature_data = self.data_set[self.feature_names]

            self.target_data = self.data_set[self.target_name]
            self.label_target(target_data=self.target_data)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.feature_data, self.target_data_labels, random_state=self.random_state,
                **kwargs)

        except Exception as e:
            _print_failure_msg(e)

        if verbose:
            if all(x is not None for x in [self.X_train, self.X_test, self.y_train, self.y_test]):
                print("Done.\nThe data is now ready for modelling.")

        if ret_dat:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def classifier(self, random_state=None, verbose=1, n_jobs=0, **kwargs):
        """
        Train a random forest model for estimating/predicting track movement.

        This method initialises and trains a random forest classifier using the provided parameters.

        :param random_state: A random seed number for reproducibility; defaults to ``None``.
        :type random_state: int | None
        :param verbose: Whether to print relevant information in console, or the level of
            verbosity for the console output; defaults to ``1``.
        :type verbose: int | bool
        :param n_jobs: Number of CPU cores to use for training the classifier.
            When ``n_jobs=0`` (default), it uses one less than the total number of available cores.
        :type n_jobs: int
        :param kwargs: [Optional] additional parameters for the
            `sklearn.ensemble.RandomForestClassifier`_ class.

        .. _`sklearn.ensemble.RandomForestClassifier`:
            https://scikit-learn.org/stable/modules/generated/
            sklearn.ensemble.RandomForestClassifier.html

        **Examples**::

            >>> from modeller import TrackMovementEstimator
            >>> element = 'Left Top'
            >>> direction = 'Up'
            >>> subsect_len = 10
            >>> tme = TrackMovementEstimator(element, direction, subsect_len)
            Initialising the feature collator ... Done.
            Initialising the estimator ... Done.
            >>> tme.integrate_data(element, direction, subsect_len, verbose=True)
            Calculating track movement ... Done.
            Collating features:
              ballasts (src. Ballast summary) ... Done.
              structures (src. CARRS and OPAS) ... Done.
            Finalising the data integration ... Done.
            >>> list(tme.track_movement.keys())
            ['Up_LeftTopOfRail_201910_202004']
            >>> tme.track_movement['Up_LeftTopOfRail_201910_202004'].head()
                                                  subsection  ...  vertical_displacement_b_abs_max
            0     LINESTRING Z (399428.96 653473.9 34.442...  ...                        -0.003203
            1     LINESTRING Z (399434.5201166851 653482....  ...                        -0.002794
            2     LINESTRING Z (399440.0608295194 653490....  ...                        -0.003202
            3     LINESTRING Z (399445.5700898784 653498....  ...                        -0.001803
            4     LINESTRING Z (399451.0456262118 653507....  ...                        -0.001201
            [5 rows x 37 columns]
            >>> tme.get_training_test_sets(test_size=0.2, random_state=1)
            >>> tme.classifier(n_estimators=300, max_depth=15, oob_score=True)
            Mean accuracy: 49.96%

                                Importance
            Curvature              0.3887
            Cant                   0.3725
            Max speed              0.2016
            Underline bridges      0.0095
            Overline bridges       0.0073
            Max axle load          0.0067
            Retaining walls        0.0060
            Tunnels                0.0058
            Stations               0.0019
            >>> from pyhelpers.settings import mpl_preferences
            >>> mpl_preferences(backend='TkAgg')
            >>> # tme.view_confusion_matrix(save_as=".svg", verbose=True)
            >>> tme.view_confusion_matrix()

        .. figure:: ../_images/tme_view_confusion_matrix.*
            :name: tfe_view_confusion_matrix
            :align: center
            :width: 100%

            A view of the confusion matrix for the example random forest model.

        .. code-block:: python

            >>> # Normalise the confusion matrix over the predicted labels
            >>> # tme.view_confusion_matrix(normalise='pred', save_as=".svg", verbose=True)
            >>> tme.view_confusion_matrix(normalise='pred')

        .. figure:: ../_images/tme_view_confusion_matrix_norm_pred.*
            :name: tme_view_confusion_matrix_norm_pred
            :align: center
            :width: 100%

            A view of the confusion matrix (normalised over the predicted labels)
            for the example random forest model.
        """

        if random_state:
            self.random_state = random_state

        if self.X_train is None or self.X_test is None:
            self.get_training_test_sets(random_state=self.random_state)

        # X_train, X_test = map(lambda dat: dat.to_numpy('float32'), [self.X_train, self.X_test])
        # y_train, y_test = map(lambda dat: dat.to_numpy('float32'), [self.y_train, self.y_test])

        kwargs.update({
            'verbose': verbose,
            'random_state': self.random_state,
            'n_jobs': os.cpu_count() - 1 if n_jobs == 0 else n_jobs,
        })
        rf = RandomForestClassifier(**kwargs)

        if verbose:
            print("\nModel training in process ... ")

        rf.fit(self.X_train, self.y_train)

        self.estimator = rf

        self.best_estimator = copy.copy(self.estimator)
        self.best_estimator_params = pd.DataFrame(
            self.best_estimator.get_params().items(),
            columns=['param_name', 'param_value']).set_index('param_name')

        if verbose:
            print("Done.")

        if verbose:
            print("\nTesting accuracy ... ")
        self.score = self.estimator.score(self.X_test, self.y_test)
        if verbose:
            print("Done.")
            print("\nMean accuracy: %.2f%%" % (self.score * 100))

        self.feature_importance = pd.DataFrame.from_dict(
            data=dict(zip(self.X_train.columns, rf.feature_importances_)), orient='index',
            columns=['Importance'])
        self.feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
        if verbose:
            print("\n", self.feature_importance)

    def view_confusion_matrix(self, normalise=None, cmap='YlGnBu', add_caption=False, save_as=None,
                              dpi=600, verbose=False, **kwargs):
        """
        Create a visual representation of the confusion matrix for the current random forest model.
    
        :param normalise: Options include ``'pred'``, ``'true'`` and ``'all'``:
    
                - ``'pred'``: Normalise the confusion matrix over the predicted labels (columns).
                - ``'true'``: Normalise over the true labels (rows).
                - ``'all'``: Normalise over the entire population.
    
            When ``normalise=None`` (default), the confusion matrix will not be normalised.
            See also the function `sklearn.metrics.confusion_matrix`_.
        :type normalise: str | None
        :param cmap: Colour map for the heatmap; defaults to ``'YlGnBu'``.
        :type cmap: str
        :param add_caption: Whether to add a caption below the figure; defaults to ``False``.
        :type add_caption: bool
        :param save_as: File format to save the visualisation; defaults to ``None``.
        :type save_as: str | None
        :param dpi: DPI (dots per inch) for saving the image; defaults to ``600``.
        :type dpi: int
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters for the `seaborn.heatmap`_ function.
    
        .. _`sklearn.metrics.confusion_matrix`:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        .. _`seaborn.heatmap`:
            https://seaborn.pydata.org/generated/seaborn.heatmap.html
    
        .. seealso::
    
            - Examples for the method :meth:`~src.modeller.TrackFixityEstimator.classifier`.
        """

        if len(self.labels) == 0:
            self.get_descriptive_labels()

        # if self.y_test_pred is None:
        if self.best_estimator.verbose == 2:
            print(f"\nWith the following RF model parameters:\n\n{self.best_estimator_params}\n")
        self.y_test_pred = self.best_estimator.predict(self.X_test)

        self.confusion_matrix = confusion_matrix(
            y_true=self.y_test, y_pred=self.y_test_pred, normalize=normalise)

        if normalise == 'pred':
            colour_bar_kws = {'label': 'Percentage of predictions (for precision)'}
            norm_info = ' (normalised over the predictions)'
            fn_suffix = '_norm_pred'
        elif normalise == 'true':
            colour_bar_kws = {'label': 'Percentage of classes in data (for recall)'}
            norm_info = ' (normalised over the actual displacement)'
            fn_suffix = '_norm_true'
        elif normalise == 'all':
            colour_bar_kws = {'label': 'Percentage (normalized over all)'}
            norm_info = ' (normalised over all)'
            fn_suffix = '_norm_all'
        else:  # None
            colour_bar_kws = {'label': 'Frequency'}
            norm_info = ''
            fn_suffix = ''

        fig = plt.figure(figsize=(len(self.labels) * 2, len(self.labels)), constrained_layout=True)
        ax = fig.add_subplot()

        # seaborn.set(font_scale=1.2)
        tick_labels = [x.replace(' mm', '') for x in self.labels]
        # noinspection PyTestUnpassedFixture
        ax_ = seaborn.heatmap(
            data=self.confusion_matrix, cmap=cmap, annot=True, fmt='.2%' if normalise else 'd',
            xticklabels=tick_labels, yticklabels=tick_labels, cbar_kws=colour_bar_kws, ax=ax,
            annot_kws={"fontsize": 14}, **kwargs)
        cbar = ax_.figure.axes[-1]
        cbar.yaxis.label.set_size(16)
        cbar.yaxis.axes.tick_params(labelsize=14)

        ax.set_xlabel('Prediction (mm)', fontsize=16, weight='bold', labelpad=8)
        ax.set_ylabel('Calculated displacement (mm)', fontsize=16, weight='bold', labelpad=2)
        ax.tick_params(axis='both', labelsize=15, labelrotation=0)
        ax.xaxis.tick_top()
        ax.xaxis.set(label_position='top')

        if add_caption:
            fig.text(
                x=0.5, y=-0.1, s=f'Confusion matrix{norm_info} on the test set',
                horizontalalignment='center', fontsize=16, fontweight='bold',
                transform=ax.transAxes)

        if save_as:
            save_plot(
                fig, filename=f"tme_view_confusion_matrix{fn_suffix}", save_as=save_as,
                path_to_img_dir="docs\\source\\_images", dpi=dpi, verbose=verbose)
