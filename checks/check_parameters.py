from __future__ import print_function, division
import imgaug as ia
# TODO ForceSign
from imgaug.parameters import (
    Binomial, Choice, DiscreteUniform, Poisson, Normal, Laplace, ChiSquare,
    Weibull, Uniform, Beta, Deterministic, Clip, Discretize, Multiply, Add,
    Divide, Power, Absolute, RandomSign, Positive, Negative,
    SimplexNoise, FrequencyNoise, Sigmoid
)
import numpy as np


def main():
    params = [
        ("Binomial(0.1)", Binomial(0.1)),
        ("Choice", Choice([0, 1, 2])),
        ("Choice with p", Choice([0, 1, 2], p=[0.1, 0.2, 0.7])),
        ("DiscreteUniform(0, 10)", DiscreteUniform(0, 10)),
        ("Poisson(0)", Poisson(0)),
        ("Poisson(5)", Poisson(5)),
        ("Discretize(Poisson(5))", Discretize(Poisson(5))),
        ("Normal(0, 1)", Normal(0, 1)),
        ("Normal(1, 1)", Normal(1, 1)),
        ("Normal(1, 2)", Normal(0, 2)),
        ("Normal(Choice([-1, 1]), 2)", Normal(Choice([-1, 1]), 2)),
        ("Discretize(Normal(0, 1.0))", Discretize(Normal(0, 1.0))),
        ("Positive(Normal(0, 1.0))", Positive(Normal(0, 1.0))),
        ("Positive(Normal(0, 1.0), mode='reroll')", Positive(Normal(0, 1.0), mode="reroll")),
        ("Negative(Normal(0, 1.0))", Negative(Normal(0, 1.0))),
        ("Negative(Normal(0, 1.0), mode='reroll')", Negative(Normal(0, 1.0), mode="reroll")),
        ("Laplace(0, 1.0)", Laplace(0, 1.0)),
        ("Laplace(1.0, 3.0)", Laplace(1.0, 3.0)),
        ("Laplace([-1.0, 1.0], 1.0)", Laplace([-1.0, 1.0], 1.0)),
        ("ChiSquare(1)", ChiSquare(1)),
        ("ChiSquare([1, 6])", ChiSquare([1, 6])),
        ("Weibull(0.5)", Weibull(0.5)),
        ("Weibull((1.0, 3.0))", Weibull((1.0, 3.0))),
        ("Uniform(0, 10)", Uniform(0, 10)),
        ("Beta(0.5, 0.5)", Beta(0.5, 0.5)),
        ("Deterministic(1)", Deterministic(1)),
        ("Clip(Normal(0, 1), 0, None)", Clip(Normal(0, 1), minval=0, maxval=None)),
        ("Multiply(Uniform(0, 10), 2)", Multiply(Uniform(0, 10), 2)),
        ("Add(Uniform(0, 10), 5)", Add(Uniform(0, 10), 5)),
        ("Absolute(Normal(0, 1))", Absolute(Normal(0, 1))),
        ("RandomSign(Poisson(1))", RandomSign(Poisson(1))),
        ("RandomSign(Poisson(1), 0.9)", RandomSign(Poisson(1), 0.9))
    ]

    params_arithmetic = [
        ("Normal(0, 1.0)", Normal(0.0, 1.0)),
        ("Normal(0, 1.0) + 5", Normal(0.0, 1.0) + 5),
        ("5 + Normal(0, 1.0)", 5 + Normal(0.0, 1.0)),
        ("5 + Normal(0, 1.0)", Add(5, Normal(0.0, 1.0), elementwise=True)),
        ("Normal(0, 1.0) * 10", Normal(0.0, 1.0) * 10),
        ("10 * Normal(0, 1.0)", 10 * Normal(0.0, 1.0)),
        ("10 * Normal(0, 1.0)", Multiply(10, Normal(0.0, 1.0), elementwise=True)),
        ("Normal(0, 1.0) / 10", Normal(0.0, 1.0) / 10),
        ("10 / Normal(0, 1.0)", 10 / Normal(0.0, 1.0)),
        ("10 / Normal(0, 1.0)", Divide(10, Normal(0.0, 1.0), elementwise=True)),
        ("Normal(0, 1.0) ** 2", Normal(0.0, 1.0) ** 2),
        ("2 ** Normal(0, 1.0)", 2 ** Normal(0.0, 1.0)),
        ("2 ** Normal(0, 1.0)", Power(2, Normal(0.0, 1.0), elementwise=True))
    ]

    params_noise = [
        ("SimplexNoise", SimplexNoise()),
        ("Sigmoid(SimplexNoise)", Sigmoid(SimplexNoise())),
        ("SimplexNoise(linear)", SimplexNoise(upscale_method="linear")),
        ("SimplexNoise(nearest)", SimplexNoise(upscale_method="nearest")),
        ("FrequencyNoise((-4, 4))", FrequencyNoise(exponent=(-4, 4))),
        ("FrequencyNoise(-2)", FrequencyNoise(exponent=-2)),
        ("FrequencyNoise(2)", FrequencyNoise(exponent=2))
    ]

    images_params = [param.draw_distribution_graph() for (title, param) in params]
    images_arithmetic = [param.draw_distribution_graph() for (title, param) in params_arithmetic]
    images_noise = [param.draw_distribution_graph(size=(1000, 10, 10)) for (title, param) in params_noise]

    ia.imshow(np.vstack(images_params))
    ia.imshow(np.vstack(images_arithmetic))
    ia.imshow(np.vstack(images_noise))


if __name__ == "__main__":
    main()
