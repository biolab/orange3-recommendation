import unittest

def run_tests():
    testmodules = [
        'test_global_avg',
        'test_item_avg',
        'test_user_avg',
        'test_user_item_baseline',
        'test_brismf',
        'test_climf',
        'test_svdplusplus',
        'test_trustsvd',
        'test_chunks'
    ]

    # Build paths
    base_path = 'orangecontrib.recommendation.tests.coverage.'
    testmodules = [base_path + m for m in testmodules]

    suite = unittest.TestSuite()
    for t in testmodules:
        try:
            # If the module defines a suite() function, call it to get the suite
            mod = __import__(t, globals(), locals(), ['suite'])
            suitefn = getattr(mod, 'suite')
            suite.addTest(suitefn())
        except (ImportError, AttributeError):
            # else, just load all the test cases from the module.
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

    unittest.TextTestRunner().run(suite)

if __name__ == "__main__":
    pass
    run_tests()