from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication

from Orange.widgets import settings
from Orange.widgets import gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

from orangecontrib.recommendation import CLiMFLearner

class OWBRISMF(OWBaseLearner):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "CLiMF"
    description = 'This model is focused on improving top-k recommendations' \
                  ' through ranking by directly maximizing the Mean Reciprocal'\
                  ' Rank (MRR)'
    icon = "icons/climf.svg"
    priority = 80

    LEARNER = CLiMFLearner

    K = settings.Setting(5)
    steps = settings.Setting(25)
    alpha = settings.Setting(0.005)
    beta = settings.Setting(0.02)

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Parameters")
        self.base_estimator = CLiMFLearner()

        gui.spin(box, self, "K", 1, 10000,
                 label="Latent factors:",
                 alignment=Qt.AlignRight, callback=self.settings_changed)

        gui.spin(box, self, "steps", 1, 10000,
                 label="Number of iterations:",
                 alignment=Qt.AlignRight, callback=self.settings_changed)

        gui.doubleSpin(box, self, "alpha", minv=1e-4, maxv=1e+5, step=1e-5,
                   label="Learning rate:", decimals=5, alignment=Qt.AlignRight,
                   controlWidth=90, callback=self.settings_changed)

        gui.doubleSpin(box, self, "beta",  minv=1e-4, maxv=1e+4, step=1e-4,
                       label="Regularization factor:", decimals=4,
                       alignment=Qt.AlignRight,
                       controlWidth=90, callback=self.settings_changed)

    def create_learner(self):
        return self.LEARNER(
            K=self.K,
            steps=self.steps,
            alpha=self.alpha,
            beta=self.beta
        )

    def get_learner_parameters(self):
        return (("Latent factors", self.K),
                ("Number of iterations", self.steps),
                ("Learning rate", self.alpha),
                ("Regularization factor", self.beta))

if __name__ == '__main__':
    app = QApplication([])
    widget = OWBRISMF()
    widget.show()
    app.exec()