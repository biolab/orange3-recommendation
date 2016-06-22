from PyQt4.QtGui import QApplication

from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

from orangecontrib.recommendation import UserItemBaselineLearner

class OWUserItemBaseline(OWBaseLearner):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "User-Item Baseline"
    description = 'This model takes the bias of users and items plus the ' \
                  'global average to make predictions.'
    icon = "icons/organization.svg"
    priority = 80

    LEARNER = UserItemBaselineLearner

if __name__ == '__main__':
    app = QApplication([])
    widget = UserItemBaselineLearner()
    widget.show()
    app.exec()