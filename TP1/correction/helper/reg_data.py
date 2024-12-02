import numpy as np
from numpy.random import randn, rand


def new_data(n=1000, d=2, file_name="helper/data.csv"):
    """Génère les exemples.

    Génère une matrice *Z* de dimension *n x d* et une matrice *Y* de dimension *n x 1*.
    Z contient les coordonnées de n points, chaque ligne correspond à un point.
    Y contient la valeur associée à chacun des points.
    Les données sont sauvegardées dans le fichier "data.csv".

    :param n: nombre d'exemples
    :type n: entier
    :param d: nombre de dimensions
    :type d: entier
    :return: 2-uplet (Z,Y)
    """
    Z = np.matrix(rand(n, d))
    theta = np.matrix(randn(d, 1))

    np.savetxt(
        file_name, np.concatenate((Z, Z * theta), axis=1), delimiter=","
    )


def get_data(file_name="helper/data.csv"):
    """Génère les exemples.

    Génère une matrice *Z* de dimension *n x d* et une matrice *Y* de dimension *n x 1*.
    Z contient les coordonnées de n points, chaque ligne correspond à un point.
    Y contient la valeur associée à chacun des points.

    :param n: nombre d'exemples
    :type n: entier
    :param d: nombre de dimensions
    :type d: entier
    :return: 2-uplet (Z,Y)
    """
    try:
        data = np.matrix(np.loadtxt(file_name, delimiter=","))
    except IOError as e:
        if str(e) != file_name + " not found.":
            raise e
        print("WARNING:", file_name, "not found, generating new data.")
        new_data(file_name=file_name)
        data = np.matrix(np.loadtxt(file_name, delimiter=","))
    return data[:, 0:-1], data[:, -1]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
