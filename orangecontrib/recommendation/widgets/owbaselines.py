from PyQt4.QtGui import QApplication

from Orange.widgets import settings
from Orange.widgets import gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

from orangecontrib.recommendation import GlobalAvgLearner, ItemAvgLearner, \
    UserAvgLearner, UserItemBaselineLearner


class OWBaselines(OWBaseLearner):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    base_name = "Baselines"
    name = base_name
    description = 'This widget contains the following baseline models: ' \
                  'Global average, User average, Item average and ' \
                  'User-Item average.'
    icon = "icons/user-item-baseline.svg"
    priority = 80

    MODEL_NAMES = ['Global average', 'User average', 'Item average',
                   'User-Item average']
    LEARNER_CLASS = [GlobalAvgLearner, ItemAvgLearner, UserAvgLearner,
                     UserItemBaselineLearner]
    G_AVG, U_AVG, I_AVG, UI_AVG = 0, 1, 2, 3

    learner_class = settings.Setting(G_AVG)
    LEARNER = LEARNER_CLASS[G_AVG]

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Learner")

        gui.radioButtons(box, self, "learner_class",
                         btnLabels=self.MODEL_NAMES,
                         callback=self.create_learner)

    def create_learner(self):
        if self.learner_class == self.G_AVG:
            self.LEARNER = GlobalAvgLearner
            self.name = self.base_name + ' (Global average)'
        elif self.learner_class == self.U_AVG:
            self.LEARNER = UserAvgLearner
            self.name = self.base_name + ' (User average)'
        elif self.learner_class == self.I_AVG:
            self.LEARNER = ItemAvgLearner
            self.name = self.base_name + ' (Item average)'
        elif self.learner_class == self.UI_AVG:
            self.LEARNER = UserItemBaselineLearner
            self.name = self.base_name + ' (User-Item average)'
        else:
            raise TypeError('Unknown learner class')

        return self.LEARNER()

if __name__ == '__main__':
    app = QApplication([])
    widget = OWBaselines()
    widget.show()
    app.exec()