import unittest
from core.input_module.input_processor import InputProcessor
from core.input_module.input_listener import InputListener

class testListener(InputListener):
    def __init__(self):
        super(testListener, self).__init__("OBJ Test listner")
    def on_input_received(self, data):
        return


class inputModuleTestTdd(unittest.TestCase):
    def test_input_module_register_method(self):
        size = 1
        inputTest = InputProcessor()
        t = testListener()
        r = inputTest.register_input_listener(t)
        self.assertEqual(len(r), size)
        self.assertIsInstance(r[0].getId(), str)

    def test_input_module_unregister_method(self):
        inputTest = InputProcessor()
        t = testListener()
        inputTest.register_input_listener(t)
        r = inputTest.unregister_input_listener(t)
        self.assertIsInstance(r.getId(), str)

    def test_input_module_process_file(self):
        inputTest = InputProcessor()
        t = testListener()
        inputTest.register_input_listener(t)
        r = inputTest.process_one("test.mid")
        self.assertTrue(r)

    def test_input_module_process_file_not_exist(self):
        inputTest = InputProcessor()
        t = testListener()
        inputTest.register_input_listener(t)
        r = inputTest.process_one("test1.mid")
        self.assertFalse(r)



if __name__ == '__main__':
    unittest.main()