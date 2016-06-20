from PyQt4.QtGui import QApplication

from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

from orangecontrib.recommendation import UserAvgLearner

class OWUserAvg(OWBaseLearner):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "User Average"
    description = 'Uses the average rating value of a user to make predictions.'
    icon = "icons/mywidget.svg"
    priority = 80

    LEARNER = UserAvgLearner

if __name__ == '__main__':
    app = QApplication([])
    widget = UserAvgLearner()
    widget.show()
    app.exec()