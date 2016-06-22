from PyQt4.QtGui import QApplication

from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

from orangecontrib.recommendation import ItemAvgLearner

class OWItemAvg(OWBaseLearner):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "Item Average"
    description = 'Uses the average rating value of an item to make predictions.'
    icon = "icons/item-avg.svg"
    priority = 80

    LEARNER = ItemAvgLearner

if __name__ == '__main__':
    app = QApplication([])
    widget = ItemAvgLearner()
    widget.show()
    app.exec()