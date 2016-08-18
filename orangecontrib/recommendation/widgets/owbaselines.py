from PyQt4.QtGui import QApplication

from Orange.widgets import settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

from orangecontrib.recommendation import GlobalAvgLearner, ItemAvgLearner, \
    UserAvgLearner, UserItemBaselineLearner
from orangecontrib.recommendation.utils import format_data


class OWBaselines(OWBaseLearner):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "Baselines"
    description = 'This widget contains the following baseline models: ' \
                  'Global average, User average, Item average and ' \
                  'User-Item average.'
    icon = "icons/user-item-baseline.svg"
    priority = 80

    MODEL_NAMES = ['Global average', 'User average', 'Item average',
                   'User-Item average']
    LEARNERS_LIST = [GlobalAvgLearner, ItemAvgLearner, UserAvgLearner,
                     UserItemBaselineLearner]
    G_AVG, U_AVG, I_AVG, UI_AVG = 0, 1, 2, 3

    learner_class = settings.Setting(G_AVG)
    LEARNER = LEARNERS_LIST[G_AVG]

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Learner")

        gui.radioButtons(box, self, "learner_class",
                         btnLabels=self.MODEL_NAMES,
                         callback=self.settings_changed)
        self._set_name_learner()

    def _set_name_learner(self):
        # Set learner and text
        if self.learner_class == self.G_AVG:
            self.LEARNER = GlobalAvgLearner
            self.name_line_edit.setText(self.name + ' (Global average)')
        elif self.learner_class == self.U_AVG:
            self.LEARNER = UserAvgLearner
            self.name_line_edit.setText(self.name + ' (User average)')
        elif self.learner_class == self.I_AVG:
            self.LEARNER = ItemAvgLearner
            self.name_line_edit.setText(self.name + ' (Item average)')
        elif self.learner_class == self.UI_AVG:
            self.LEARNER = UserItemBaselineLearner
            self.name_line_edit.setText(self.name + ' (User-Item average)')
        else:
            raise TypeError('Unknown learner class')

    def settings_changed(self):
        self._set_name_learner()
        super().settings_changed()

    def _check_data(self):
        self.valid_data = False

        if self.data is not None:
            try:  # Check ratings data
                valid_ratings = format_data.check_data(self.data)
            except Exception as e:
                valid_ratings = False
                print('Error checking rating data: ' + str(e))

            if not valid_ratings:  # Check if it's valid
                self.Error.data_error(
                    "Data not valid for rating models.")
            else:
                self.valid_data = True

        return self.valid_data

    def update_learner(self):
        self._check_data()

        # If our method returns 'False', it could be because there is no data.
        # But when cross-validating, a learner is required, as the data is in
        # the widget Test&Score
        if self.valid_data or self.data is None:
            super().update_learner()

    def update_model(self):
        self._check_data()
        super().update_model()

if __name__ == '__main__':
    app = QApplication([])
    widget = OWBaselines()
    widget.show()
    app.exec()