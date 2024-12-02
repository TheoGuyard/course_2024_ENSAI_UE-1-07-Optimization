import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def display_gradient_descent(
    function, coord_descent, x_min=-2, x_max=2, y_min=-2, y_max=2, tick=0.1
):
    """
    Plot a 3D wireframe of the parameter function and a line representing the
    gradient descent.
    :param function: the function to optimize
    :type function: function
    :param coord_descent: the coordinate of the gradient descent point
    :type coord_descent: np.array
    :type x_min: float
    :param x_max: the max of the x axis
    :type x_max: float
    :param y_min: the min of the x axis
    :type y_min: float
    :param y_max: the max of the x axis
    :type y_max: float
    :param tick: the space between to point
    :type tick: float
    :return: None
    :rtype:
    """
    fig = plt.figure()
    # ax = fig.gca(projection="3d")
    ax = fig.add_subplot(projection="3d")

    x_min = min(coord_descent[0][0], x_min)
    y_min = min(coord_descent[0][1], y_min)
    # Generate the 2D grid
    X = np.arange(x_min, x_max, tick)
    Y = np.arange(y_min, y_max, tick)
    X, Y = np.meshgrid(X, Y)

    # Compute the Z axis
    Z = []
    for i in range(len(X)):
        Z.append(
            np.fromiter(
                (
                    function.function_definition(np.array([X[i, j], Y[i, j]]))
                    for j in range(len(X[i]))
                ),
                X.dtype,
            )
        )
    Z = np.asarray(Z)

    # Plot the wireframe.
    ax.plot_wireframe(
        X, Y, Z, rcount=(x_max - x_min) // tick, ccount=(y_max - y_min) // tick
    )
    f_x = []
    for i in range(len(coord_descent)):
        f_x.append(function.function_definition(coord_descent[i]))
    f_x = np.asarray(f_x)
    ax.plot(
        coord_descent[::100, 0],
        coord_descent[::100, 1],
        f_x[::100],
        color="red",
        marker="+",
    )

    plt.show()


def contour(
    studied_function,
    x_min=-2,
    x_max=2,
    y_min=-2,
    y_max=2,
    levels=[0.1, 1, 2, 4, 9, 16, 25, 36, 49, 64, 81, 100],
):
    """

    :param studied_function: the function to display
    :type studied_function: AbstractFunction
    :param x_min: the x min to plot
    :type x_min: float
    :param x_max: the x max to plot
    :type x_max: float
    :param y_min: the y min to plot
    :type y_min: to plot
    :param y_max: the y max to plot
    :type y_max: to plot
    :param levels: the contour level
    :type levels: List of float
    :return: None
    :rtype: None
    """

    # Generate the 2D grid
    X = np.linspace(x_min, x_max, 100)
    Y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(X, Y)
    # Compute the Z axis
    Z = []
    for i in range(len(X)):
        Z.append(
            np.fromiter(
                (
                    studied_function.function_definition(np.array([X[i, j], Y[i, j]]))
                    for j in range(len(X))
                ),
                X.dtype,
            )
        )
    Z = np.asarray(Z)
    c = plt.contour(X, Y, Z, levels)
    plt.show()


def contour_and_gradient(
    studied_function,
    coord_descent,
    x_min=-2,
    x_max=2,
    y_min=-2,
    y_max=2,
    levels=[0.1, 1, 2, 4, 9, 16, 25, 36, 49, 64, 81, 100, 125, 160, 200, 250],
):

    x_min = min(coord_descent[0][0], 0)
    y_min = min(coord_descent[0][1], 0)
    # Generate the 2D grid
    X = np.linspace(x_min, x_max, 100)
    Y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(X, Y)
    # Compute the Z axis
    Z = []
    for i in range(len(X)):
        Z.append(
            np.fromiter(
                (
                    studied_function.function_definition(np.array([X[i, j], Y[i, j]]))
                    for j in range(len(X))
                ),
                X.dtype,
            )
        )
    Z = np.asarray(Z)
    c = plt.contour(X, Y, Z, levels)
    plt.plot(
        coord_descent[1:-1:100, 0],
        coord_descent[1:-1:100, 1],
        "o--",
        c="red",
    )
    plt.plot(coord_descent[1, 0], coord_descent[1, 1], "+", c="blue", markersize=15)
    plt.plot(coord_descent[-1, 0], coord_descent[-1, 1], "+", c="green", markersize=15)
    plt.show()


def display_3d_surface(
    studied_function, x_min=-2, x_max=2, y_min=-2, y_max=2, tick=0.1
):
    """
    Function to display a function in 3D. It's only straightfoward technical
    code.
    :param studied_function: the function you want to plot
    :type studied_function: AbstractFunction
    :param x_min: the min of the x axis
    :type x_min: float
    :param x_max: the max of the x axis
    :type x_max: float
    :param y_min: the min of the x axis
    :type y_min: float
    :param y_max: the max of the x axis
    :type y_max: float
    :param tick: the space between to point
    :type tick: float
    :return: None
    :rtype:
    """

    # Creation og the 3D canvas
    fig = plt.figure()
    # ax = fig.gca(projection="3d")
    ax = fig.add_subplot(projection="3d")

    # Generate the 2D grid
    X = np.arange(x_min, x_max, tick)
    Y = np.arange(y_min, y_max, tick)
    X, Y = np.meshgrid(X, Y)

    # Compute the Z axis
    # Ok this code is strange, but numpy ask a 2D vector for Z so I have to
    # compture it

    # Create a array with the same shape as X or Y
    Z = np.zeros(X.shape)
    # Iterate threw the (x,y) plane
    for i in range(len(X)):
        # Maybe the more complicate code here. I skip a for loop and decide to
        # fromiter. It applies a function for each element of a iterable and
        # return an array.
        Z[i, :] = np.fromiter(
            (
                studied_function.function_definition([X[i, j], Y[i, j]])
                for j in range(len(X[i]))
            ),
            X.dtype,
        )
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # Show the plot
    plt.show()
