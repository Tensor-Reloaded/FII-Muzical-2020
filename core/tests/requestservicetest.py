import unittest
import requests

class serverTestTdd(unittest.TestCase):
    
    def test_get(self):
        self.url = 'http://localhost:{port}/'.format(port=32008)
        response = requests.get(self.url)
        self.assertTrue(response.ok)

    def test_post(self):
        self.url = 'http://localhost:{port}/'.format(port=32008)
        response = requests.post(self.url)
        self.assertTrue(response.ok)
