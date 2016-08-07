from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication

from Orange.widgets import settings
from Orange.widgets import gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

from orangecontrib.recommendation import CLiMFLearner


class OWCLIMF(OWBaseLearner):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "CLiMF"
    description = 'This model is focused on improving top-k recommendations' \
                  ' through ranking by directly maximizing the Mean Reciprocal'\
                  ' Rank (MRR)'
    icon = "icons/climf.svg"
    priority = 80

    LEARNER = CLiMFLearner

    num_factors = settings.Setting(5)
    num_iter = settings.Setting(25)
    learning_rate = settings.Setting(0.005)
    lmbda = settings.Setting(0.02)

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Parameters")
        self.base_estimator = CLiMFLearner()

        gui.spin(box, self, "num_factors", 1, 10000,
                 label="Number of latent factors:",
                 alignment=Qt.AlignRight, callback=self.settings_changed)

        gui.spin(box, self, "num_iter", 1, 10000,
                 label="Number of iterations:",
                 alignment=Qt.AlignRight, callback=self.settings_changed)

        gui.doubleSpin(box, self, "learning_rate", minv=1e-4, maxv=1e+5,
                       step=1e-5, label="Learning rate:", decimals=5,
                       alignment=Qt.AlignRight, controlWidth=90,
                       callback=self.settings_changed)

        gui.doubleSpin(box, self, "lmbda",  minv=1e-4, maxv=1e+4, step=1e-4,
                       label="Regularization:", decimals=4,
                       alignment=Qt.AlignRight, controlWidth=90,
                       callback=self.settings_changed)

    def create_learner(self):
        return self.LEARNER(
            num_factors=self.num_factors,
            num_iter=self.num_iter,
            learning_rate=self.learning_rate,
            lmbda=self.lmbda
        )

    def get_learner_parameters(self):
        return (("Number of latent factors", self.num_factors),
                ("Number of iterations", self.num_iter),
                ("Learning rate", self.learning_rate),
                ("Regularization", self.lmbda))

if __name__ == '__main__':
    app = QApplication([])
    widget = OWCLIMF()
    widget.show()
    app.exec()