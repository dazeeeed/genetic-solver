# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 2022

@author: agademer & tdrumond

Module containing utility functions to instantiate a traveling 
salesperson problem.
This module is accompanied of a cities.txt file, containing a list of
2D coordinates representing different cities.
"""

import matplotlib.pyplot as plt
from random import shuffle
from typing import List, Dict, Tuple, Optional
from collections.abc import Iterable, Mapping

Coordinates = Tuple[int, int]

def load_cities(filename) -> Dict[str,Coordinates]:
    """ load the cities list from a text file """
    with open(filename) as file:
        nbCities = int(file.readline())
        cities = {}
        for _ in range(nbCities):
            city_name, x, y = file.readline().split(";")
            cities[city_name]=(int(x), int(y))
        return cities


def default_road(cities:Dict) -> List:
    """ Default road: all the cities in the order of the text file """
    return list(cities.keys())


def draw_cities(cities:Dict, road=Optional[Iterable[str]]):
    """ Plot the cities and the trajectory """
    x_cords, y_coords = tuple(zip(*cities.values()))
    plt.figure()
    plt.scatter(x_cords, y_coords, color="red")
    if road is not None:
        road_coordinates = [cities[c] for c in road]
        x_cords, y_coords = list(zip(*road_coordinates))
        plt.plot(x_cords, y_coords)
        for city_name in road:
            plt.annotate(
                city_name, 
                cities[city_name],
                xytext=(4, -1), 
                textcoords='offset points')
    plt.gca().set_aspect('equal')
    plt.show()


def distance(city1:Coordinates, city2:Coordinates) -> float:
    """ Euclidian distance between two cities """
    return ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5


def road_length(cities:Dict[str, Coordinates], road:Iterable[str]) -> float:
    """ Calculate the length of the road """
    road_coords = [cities[c] for c in road]
    total = 0
    for i in range(len(road_coords)-1):
        total += distance(road_coords[i], road_coords[i+1])
    total += distance(road_coords[-1], road_coords[0])
    return total


if __name__ == '__main__':
    cities = load_cities("cities.txt")
    print(cities)
    road = default_road(cities)
    shuffle(road)
    print(road)
    draw_cities(cities, road)
    print(road_length(cities, road))
