import unittest
from shallowing_NN_v2 import add_partial_score_to_file
import json
import numpy as np
import os

class TestAddPartialScoreToFile(unittest.TestCase):

    def test_add_partial_score_to_file(self):
        if os.path.exists('test.txt'):
            os.remove('test.txt')
        score = [11, 22, 33]
        for i in range(10):
            for j in range(10):
                add_partial_score_to_file(score, 'test.txt', i)

        file = open('test.txt', 'r')
        dictionary = json.loads(file.read())
        file.close()

        for key in dictionary.keys():
            self.assertListEqual(dictionary[key]['accuracy'], [score[2]] * 10)
            self.assertListEqual(dictionary[key]['loss'], [score[1]] * 10)

        if os.path.exists('test.txt'):
            os.remove('test.txt')
