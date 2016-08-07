from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication

from Orange.data import Table
from Orange.widgets import settings
from Orange.widgets import gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

from orangecontrib.recommendation import TrustSVDLearner


class OWTrustSVD(OWBaseLearner):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "TrustSVD"
    description = 'Matrix factorization model which makes use of implicit ' \
                  'feedback information and social trust'
    icon = "icons/trustsvd.svg"
    priority = 80

    LEARNER = TrustSVDLearner

    MISSING_DATA_WARNING = 0

    inputs = [("Trust information ", Table, "set_trust")]

    outputs = [("P", Table),
               ("Q", Table),
               ("Y", Table),
               ("W", Table)]

    num_factors = settings.Setting(5)
    num_iter = settings.Setting(25)
    learning_rate = settings.Setting(0.005)
    bias_learning_rate = settings.Setting(0.005)
    lmbda = settings.Setting(0.02)
    bias_lmbda = settings.Setting(0.02)
    social_lmbda = settings.Setting(0.02)
    trust = None

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Parameters")

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

        gui.doubleSpin(box, self, "social_lmbda", minv=1e-4, maxv=1e+4,
                       step=1e-4, label="Social regularization:", decimals=4,
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
            trust=self.trust
        )

    def get_learner_parameters(self):
        return (("Number of latent factors", self.num_factors),
                ("Number of iterations", self.num_iter),
                ("Learning rate", self.learning_rate),
                ("Bias learning rate", self.bias_learning_rate),
                ("Regularization", self.lmbda),
                ("Bias regularization", self.bias_lmbda),
                ("Social regularization", self.social_lmbda))

    def update_learner(self):
        if self.trust is None:
            self.warning(self.MISSING_DATA_WARNING,
                         "Trust data input is needed.")
            return
        else:
            self.warning(self.MISSING_DATA_WARNING)
        super().update_learner()

    def update_model(self):
        super().update_model()

        P = None
        Q = None
        Y = None
        W = None
        if self.valid_data:
            P = self.model.getPTable()
            Q = self.model.getQTable()
            Y = self.model.getYTable()
            W = self.model.getWTable()

        self.send("P", P)
        self.send("Q", Q)
        self.send("Y", Y)
        self.send("W", W)

    def set_trust(self, trust):
        self.trust = trust
        self.update_learner()


if __name__ == '__main__':
    app = QApplication([])
    widget = OWTrustSVD()
    widget.show()
    app.exec()