from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.widgets import settings, gui
from Orange.widgets.utils.signals import Output, Input
from Orange.widgets.utils.widgetpreview import WidgetPreview

from orangecontrib.recommendation import SVDPlusPlusLearner
from orangecontrib.recommendation.utils import format_data
from orangecontrib.recommendation.widgets.utils.owconcurrentlearner import OWConcurrentLearner
import orangecontrib.recommendation.optimizers as opt


class OWSVDPlusPlus(OWConcurrentLearner):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "SVD++"
    description = 'Matrix factorization model which makes use of implicit ' \
                  'feedback information'
    icon = "icons/svdplusplus.svg"
    priority = 80

    LEARNER = SVDPlusPlusLearner

    class Inputs(OWConcurrentLearner.Inputs):
        feedback = Input("Feedback information", Table)

    class Outputs(OWConcurrentLearner.Outputs):
        p = Output("P", Table, explicit=True)
        q = Output("Q", Table, explicit=True)
        y = Output("Y", Table, explicit=True)

    # Parameters (general)
    num_factors = settings.Setting(10)
    num_iter = settings.Setting(15)
    learning_rate = settings.Setting(0.01)
    bias_learning_rate = settings.Setting(0.01)
    lmbda = settings.Setting(0.1)
    bias_lmbda = settings.Setting(0.1)
    feedback = None

    # Seed (Random state)
    RND_SEED, FIXED_SEED = range(2)
    seed_type = settings.Setting(RND_SEED)
    random_seed = settings.Setting(42)

    # SGD optimizers
    class _Optimizer:
        SGD, MOMENTUM, NAG, ADAGRAD, RMSPROP, ADADELTA, ADAM, ADAMAX = range(8)
        names = ['Vanilla SGD', 'Momentum', "Nesterov momentum", 'AdaGrad',
                 'RMSprop', 'AdaDelta', 'Adam', 'Adamax']

    opt_type = settings.Setting(_Optimizer.SGD)
    momentum = settings.Setting(0.9)
    rho = settings.Setting(0.9)
    beta1 = settings.Setting(0.9)
    beta2 = settings.Setting(0.999)

    def add_main_layout(self):
        # hbox = gui.hBox(self.controlArea, "Settings")

        # Frist groupbox (Common parameters)
        box = gui.widgetBox(self.controlArea, "Parameters")

        gui.spin(box, self, "num_factors", 1, 10000,
                 label="Number of latent factors:",
                 alignment=Qt.AlignRight, callback=self.settings_changed)

        gui.spin(box, self, "num_iter", 1, 10000,
                 label="Number of iterations:",
                 alignment=Qt.AlignRight, callback=self.settings_changed)

        gui.doubleSpin(box, self, "learning_rate", minv=1e-5, maxv=1e+5,
                       step=1e-5, label="Learning rate:", decimals=5,
                       alignment=Qt.AlignRight, controlWidth=90,
                       callback=self.settings_changed)

        gui.doubleSpin(box, self, "bias_learning_rate", minv=1e-5, maxv=1e+5,
                       step=1e-5, label="     Bias learning rate:", decimals=5,
                       alignment=Qt.AlignRight, controlWidth=90,
                       callback=self.settings_changed)

        gui.doubleSpin(box, self, "lmbda", minv=1e-4, maxv=1e+4, step=1e-4,
                       label="Regularization:", decimals=4,
                       alignment=Qt.AlignRight, controlWidth=90,
                       callback=self.settings_changed)

        gui.doubleSpin(box, self, "bias_lmbda", minv=1e-4, maxv=1e+4, step=1e-4,
                       label="     Bias regularization:", decimals=4,
                       alignment=Qt.AlignRight, controlWidth=90,
                       callback=self.settings_changed)

        # Second groupbox (SGD optimizers)
        box = gui.widgetBox(self.controlArea, "SGD optimizers")

        gui.comboBox(box, self, "opt_type", label="SGD optimizer: ",
            items=self._Optimizer.names, orientation=Qt.Horizontal,
            addSpace=4, callback=self._opt_changed)

        _m_comp = gui.doubleSpin(box, self, "momentum", minv=1e-4, maxv=1e+4,
                                 step=1e-4, label="Momentum:", decimals=4,
                                 alignment=Qt.AlignRight, controlWidth=90,
                                 callback=self.settings_changed)

        _r_comp = gui.doubleSpin(box, self, "rho", minv=1e-4, maxv=1e+4,
                                 step=1e-4, label="Rho:", decimals=4,
                                 alignment=Qt.AlignRight, controlWidth=90,
                                 callback=self.settings_changed)

        _b1_comp = gui.doubleSpin(box, self, "beta1", minv=1e-5, maxv=1e+5,
                                  step=1e-4, label="Beta 1:", decimals=5,
                                  alignment=Qt.AlignRight, controlWidth=90,
                                  callback=self.settings_changed)

        _b2_comp = gui.doubleSpin(box, self, "beta2", minv=1e-5, maxv=1e+5,
                                  step=1e-4, label="Beta 2:", decimals=5,
                                  alignment=Qt.AlignRight, controlWidth=90,
                                  callback=self.settings_changed)
        gui.rubber(box)
        self._opt_params = [_m_comp, _r_comp, _b1_comp, _b2_comp]
        self._show_right_optimizer()

        # Third groupbox (Random state)
        box = gui.widgetBox(self.controlArea, "Random state")
        rndstate = gui.radioButtons(box, self, "seed_type",
                                    callback=self.settings_changed)
        gui.appendRadioButton(rndstate, "Random seed")
        gui.appendRadioButton(rndstate, "Fixed seed")
        ibox = gui.indentedBox(rndstate)
        self.spin_rnd_seed = gui.spin(ibox, self, "random_seed", -1e5, 1e5,
                                      label="Seed:", alignment=Qt.AlignRight,
                                      callback=self.settings_changed)
        self.settings_changed()  # Update (extra) settings

    def settings_changed(self):
        # Enable/Disable Fixed seed control
        self.spin_rnd_seed.setEnabled(self.seed_type == self.FIXED_SEED)
        super().settings_changed()

    def _show_right_optimizer(self):
        enabled = [[False, False, False, False],  # SGD
                   [True, False, False, False],  # Momentum
                   [True, False, False, False],  # NAG
                   [False, False, False, False],  # AdaGrad
                   [False, True, False, False],  # RMSprop
                   [False, True, False, False],  # AdaDelta
                   [False, False, True, True],  # Adam
                   [False, False, True, True],  # Adamax
                ]
        mask = enabled[self.opt_type]
        for spin, enabled in zip(self._opt_params, mask):
            [spin.box.hide, spin.box.show][enabled]()

    def _opt_changed(self):
        self._show_right_optimizer()
        self.settings_changed()

    def select_optimizer(self):
        if self.opt_type == self._Optimizer.MOMENTUM:
            return opt.Momentum(momentum=self.momentum)

        elif self.opt_type == self._Optimizer.NAG:
            return opt.NesterovMomentum(momentum=self.momentum)

        elif self.opt_type == self._Optimizer.ADAGRAD:
            return opt.AdaGrad()

        elif self.opt_type == self._Optimizer.RMSPROP:
            return opt.RMSProp(rho=self.rho)

        elif self.opt_type == self._Optimizer.ADADELTA:
            return opt.AdaDelta(rho=self.rho)

        elif self.opt_type == self._Optimizer.ADAM:
            return opt.Adam(beta1=self.beta1, beta2=self.beta2)

        elif self.opt_type == self._Optimizer.ADAMAX:
            return opt.Adamax(beta1=self.beta1, beta2=self.beta2)

        else:
            return opt.SGD()

    def create_learner(self):
        # Set random state
        if self.seed_type == self.FIXED_SEED:
            seed = self.random_seed
        else:
            seed = None

        return self.LEARNER(
            num_factors=self.num_factors,
            num_iter=self.num_iter,
            learning_rate=self.learning_rate,
            bias_learning_rate=self.bias_learning_rate,
            lmbda=self.lmbda,
            bias_lmbda=self.bias_lmbda,
            feedback=self.feedback,
            optimizer=self.select_optimizer(),
            random_state=seed
        )

    def get_learner_parameters(self):
        return (("Number of latent factors", self.num_factors),
                ("Number of iterations", self.num_iter),
                ("Learning rate", self.learning_rate),
                ("Bias learning rate", self.bias_learning_rate),
                ("Regularization", self.lmbda),
                ("Bias regularization", self.bias_lmbda),
                ("SGD optimizer", self._Optimizer.names[self.opt_type]))

    def _check_data(self):
        self.valid_data = False

        if self.data is not None:
            try:  # Check ratings data
                valid_ratings = format_data.check_data(self.data)
            except Exception as e:
                valid_ratings = False
                print('Error checking rating data: ' + str(e))

            if not valid_ratings:  # Check if it's valid
                self.Error.data_error(
                    "Data not valid for rating models.")
            else:
                self.valid_data = True

        return self.valid_data

    def update_learner(self):
        self._check_data()

        # If our method returns 'False', it could be because there is no data.
        # But when cross-validating, a learner is required, as the data is in
        # the widget Test&Score
        if self.valid_data or self.data is None:
            super().update_learner()

    def commit(self):
        self.Outputs.model.send(self.model)

        P = None
        Q = None
        Y = None
        if self.valid_data and self.model is not None:
            P = self.model.getPTable()
            Q = self.model.getQTable()
            Y = self.model.getYTable()

        self.Outputs.p.send(P)
        self.Outputs.q.send(Q)
        self.Outputs.y.send(Y)

    @Inputs.feedback
    def set_feedback(self, feedback):
        self.feedback = feedback
        if self.auto_apply:
            self.apply()


if __name__ == '__main__':
    WidgetPreview(OWSVDPlusPlus).run()
