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

    inputs = [("Trust information ", Table, "set_trust"),
              ("Feedback information", Table, "set_feedback")]

    outputs = [("P", Table),
               ("Q", Table),
               ("Y", Table),
               ("W", Table)]

    K = settings.Setting(5)
    steps = settings.Setting(25)
    alpha = settings.Setting(0.07)
    beta = settings.Setting(0.1)
    beta_trust = settings.Setting(0.05)
    trust = None
    feedback = None

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Parameters")

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

        gui.doubleSpin(box, self, "beta_trust", minv=1e-4, maxv=1e+4, step=1e-4,
                       label="Regularization factor (Trust):", decimals=4,
                       alignment=Qt.AlignRight,
                       controlWidth=90, callback=self.settings_changed)

    def create_learner(self):
        return self.LEARNER(
            K=self.K,
            steps=self.steps,
            alpha=self.alpha,
            beta=self.beta,
            beta_trust=self.beta_trust,
            trust=self.trust,
            feedback=self.feedback
        )

    def get_learner_parameters(self):
        return (("Latent factors", self.K),
                ("Number of iterations", self.steps),
                ("Learning rate", self.alpha),
                ("Regularization factor", self.beta),
                ("Regularization factor (Trust)", self.beta_trust))

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

    def set_feedback(self, feedback):
        self.feedback = feedback
        self.update_learner()


if __name__ == '__main__':
    app = QApplication([])
    widget = OWTrustSVD()
    widget.show()
    app.exec()