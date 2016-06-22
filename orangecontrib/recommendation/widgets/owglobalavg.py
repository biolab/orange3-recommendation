from PyQt4.QtGui import QApplication

from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

from orangecontrib.recommendation import GlobalAvgLearner

class OWGlobalAvg(OWBaseLearner):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "Global Average"
    description = 'Uses the average rating value of all ratings to make predictions'
    icon = "icons/average.svg"
    priority = 80

    LEARNER = GlobalAvgLearner

if __name__ == '__main__':
    app = QApplication([])
    widget = GlobalAvgLearner()
    widget.show()
    app.exec()