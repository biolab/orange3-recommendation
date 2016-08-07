from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication

from Orange.data import Table
from Orange.widgets import settings
from Orange.widgets import gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

from orangecontrib.recommendation import SVDPlusPlusLearner


class OWSVDPlusPlus(OWBaseLearner):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "SVD++"
    description = 'Matrix factorization model which makes use of implicit ' \
                  'feedback information'
    icon = "icons/svdplusplus.svg"
    priority = 80

    LEARNER = SVDPlusPlusLearner

    inputs = [("Feedback information", Table, "set_feedback")]

    outputs = [("P", Table),
               ("Q", Table),
               ("Y", Table)]

    num_factors = settings.Setting(5)
    num_iter = settings.Setting(25)
    learning_rate = settings.Setting(0.005)
    bias_learning_rate = settings.Setting(0.005)
    lmbda = settings.Setting(0.02)
    bias_lmbda = settings.Setting(0.02)
    feedback = None

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Parameters")
        self.base_estimator = SVDPlusPlusLearner()

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

        gui.doubleSpin(box, self, "bias_learning_rate", minv=1e-4, maxv=1e+5,
                       step=1e-5, label="Learning rate:", decimals=5,
                       alignment=Qt.AlignRight, controlWidth=90,
                       callback=self.settings_changed)

        gui.doubleSpin(box, self, "lmbda", minv=1e-4, maxv=1e+4, step=1e-4,
                       label="Regularization:", decimals=4,
                       alignment=Qt.AlignRight, controlWidth=90,
                       callback=self.settings_changed)

        gui.doubleSpin(box, self, "bias_lmbda", minv=1e-4, maxv=1e+4, step=1e-4,
                       label="Bias regularization:", decimals=4,
                       alignment=Qt.AlignRight, controlWidth=90,
                       callback=self.settings_changed)

    def create_learner(self):
        return self.LEARNER(
            num_factors=self.num_factors,
            num_iter=self.num_iter,
            learning_rate=self.learning_rate,
            bias_learning_rate=self.bias_learning_rate,
            lmbda=self.lmbda,
            bias_lmbda=self.bias_lmbda,
            feedback=self.feedback
        )

    def get_learner_parameters(self):
        return (("Number of latent factors", self.num_factors),
                ("Number of iterations", self.num_iter),
                ("Learning rate", self.learning_rate),
                ("Bias learning rate", self.bias_learning_rate),
                ("Regularization", self.lmbda),
                ("Bias regularization", self.bias_lmbda))

    def update_model(self):
        super().update_model()

        P = None
        Q = None
        Y = None
        if self.valid_data:
            P = self.model.getPTable()
            Q = self.model.getQTable()
            Y = self.model.getYTable()

        self.send("P", P)
        self.send("Q", Q)
        self.send("Y", Y)

    def set_feedback(self, feedback):
        self.feedback = feedback
        self.update_learner()

if __name__ == '__main__':
    app = QApplication([])
    widget = OWSVDPlusPlus()
    widget.show()
    app.exec()