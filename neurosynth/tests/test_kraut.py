import os
import logging
import os.path as op

try:
    from testkraut.testcase import TestFromSPEC, discover_specs, template_case, TemplateTestCase

    # this block is optional
    if 'TESTKRAUT_LOGGER_VERBOSE' in os.environ:
        lgr = logging.getLogger('testkraut')
        console = logging.StreamHandler()
        lgr.addHandler(console)
        cfg = os.environ['TESTKRAUT_LOGGER_VERBOSE']
        if cfg == 'debug':
            lgr.setLevel(logging.DEBUG)
        else:
            lgr.setLevel(logging.INFO)

    class TestKrautTests(TestFromSPEC):
        __metaclass__ = TemplateTestCase
        search_dirs = [os.path.join(os.path.dirname(__file__), 'data')]

        @template_case(discover_specs([op.join(op.dirname(__file__),
                                               'kraut')]))
        def _run_spec_test(self, spec_filename):
            return TestFromSPEC._run_spec_test(self, spec_filename)

except ImportError:
    pass
