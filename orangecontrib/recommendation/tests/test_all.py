import unittest

if __name__ == "__main__":
    testmodules = [
        'test_global_avg',
        'test_item_avg',
        'test_user_avg',
        'test_user_item_baseline',
        'test_brismf'
    ]

    suite = unittest.TestSuite()
    for t in testmodules:
        try:
            # If the module defines a suite() function, call it to get the suite.
            mod = __import__(t, globals(), locals(), ['suite'])
            suitefn = getattr(mod, 'suite')
            suite.addTest(suitefn())
        except (ImportError, AttributeError):
            # else, just load all the test cases from the module.
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

    unittest.TextTestRunner().run(suite)