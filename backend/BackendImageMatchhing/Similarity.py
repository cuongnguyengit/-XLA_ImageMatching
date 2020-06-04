from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import numpy as np

class Similarity:
    @staticmethod
    def cosine(A, B):
        return cosine_distances(A, B)[0][0]

    @staticmethod
    def euclidean(A, B):
        return euclidean_distances(A, B)[0][0]

    @staticmethod
    def distance_list(A, list_B, cosine=True):
        if cosine:
            return cosine_distances(A, list_B)[0]
        else:
            return euclidean_distances(A, list_B)[0]



if __name__ == '__main__':
    A = np.random.randn(4).reshape(1, 4)
    B = np.random.randn(4).reshape(1, 4)
    C = np.random.randn(8).reshape(2, 4)
    print(A)
    print(B)
    print(C)
    cos = Similarity.cosine(A, B)
    eu = Similarity.euclidean(A, B)
    print(cos)
    print(eu)
    cos = Similarity.distance_list(A, C, cosine=True)
    print(cos)
    cos = Similarity.distance_list(A, C, cosine=False)
    print(cos)
