# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 2022

@author: tdrumond

Module containing utility functions to implement a Mastermind game.

The main element is the MastermindMatch class, that allows to instantiate
a match of the game having a certain secret code generated at random.
This class plays the role of the codemaker, allowing to check if a guess
is correct and rating how close a guess is to the secret code.
"""
from random import choice
from typing import List

# Possible colors for codes in in the game
_colors = ['blue', 'red', 'green', 'yellow', 'orange', 'violet']
_colors_to_int = dict([(c, i) for i, c in enumerate(_colors)])


def get_possible_colors():
    """Getter function to read the array of possible colors"""
    return _colors


def generate_random_secret(size) -> List[str]:
    """Generate a random secret of a given size"""
    secret = [choice(_colors) for _ in range(size)]
    return secret


class MastermindMatch:
    """Class to instantiate a mastermind game match with a random secret code.
    A MastermindMatch object plays the role of the codemaker player,
    while the code instantiating the class typically plays the role of code
    guesser.
    """

    def __init__(self,
                 secret_size=4,
                 correct_color_points=1,
                 correct_position_points=3):
        """Instantiates a mastermind guess with a random secret code

        A match can be created by calling:
        match = MastermindMatch()

        Args:
            secret_size (int, optional): defines the size of the secred.
            Defaults to 4.
            correct_color_points (int, optional): points awarded for a correct
            color at the wrong position. Defaults to 1.
            correct_position_points (int, optional): points awarded for a
            correct color at the right position. Defaults to 3.
        """
        self._secret = generate_random_secret(secret_size)
        self.correct_color_points = correct_color_points
        self.correct_position_points = correct_position_points

    def is_correct(self, guess: List[str]) -> bool:
        """Checks whether a guess matches the secret code

        Args:
            guess (list[str]): a mastermind guess as a list of color strings

        Returns:
            bool: True if the guess matches the secret code, False otherwise
        """
        return guess == self._secret

    def generate_random_guess(self):
        return generate_random_secret(len(self._secret))

    def rate_guess(self, guess: List[str]):
        """Gives a numeric score for a given guess proportional to how close
        it is to the secret code (higher is better)

        Args:
            guess (list[srt]): a mastermind guess as a list of color strings

        Returns:
            int or float: the computed score
        """
        correct_position = 0
        correct_colors = 0
        for i, color in enumerate(guess):
            if self._secret[i] == color:
                correct_position += 1
            elif color in self._secret:
                correct_colors += 1
        score = correct_colors*self.correct_color_points + \
            correct_position * self.correct_position_points
        return score

    def secret_size(self):
        """Returns the size of the secret code"""
        return len(self._secret)

    def max_score(self):
        """Returns the maximum possible score under the defined point
        schedule for this instance"""
        return self.correct_position_points * len(self._secret)


def encode_guess(guess: List[str]) -> List[int]:
    """Encode a guess in a list of integest corresponding to the color postion
    int the list of valid colors

    Args:
        guess (list[str]): a mastermind guess as a list of color strings

    Returns:
        list[int]: a mastermind guess as a list of integers
    """
    return [_colors_to_int[c] for c in guess]
