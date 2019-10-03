import copy
from types import SimpleNamespace as namespace

from Orange.widgets import gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin


class Result(namespace):
    model = None


def run(data, learner, max_iter, state: TaskState):
    def callback(iteration):
        nonlocal state
        state.set_progress_value(int(iteration / max_iter * 100))
        return state.is_interruption_requested()

    _learner = copy.copy(learner)
    _learner.callback = callback

    model = _learner(data)
    return Result(model=model)


class OWConcurrentLearner(OWBaseLearner, ConcurrentWidgetMixin, openclass=True):
    def __init__(self, preprocessors=None):
        ConcurrentWidgetMixin.__init__(self)
        OWBaseLearner.__init__(self, preprocessors=preprocessors)

    def setup_layout(self):
        super().setup_layout()
        self._cancel_btn = gui.button(self.apply_button, self, "Cancel",
                                      callback=self.cancel)
        self._cancel_btn.setEnabled(False)

    def update_model(self):
        self.show_fitting_failed(None)
        self.model = None
        if self.check_data() and self._check_data():
            self._cancel_btn.setEnabled(True)
            self.start(run, self.data, self.learner, self.num_iter)

    def cancel(self):
        super().cancel()
        self.model = None
        self._cancel_btn.setEnabled(False)
        self.commit()

    def on_done(self, result: Result):
        self.model = result.model
        self.model.name = self.learner_name or self.name
        self.model.instances = self.data
        self._cancel_btn.setEnabled(False)
        self.commit()

    def on_exception(self, ex: Exception):
        self.show_fitting_failed(ex)
        self._cancel_btn.setEnabled(False)
        self.commit()

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()
