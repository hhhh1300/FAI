import numpy as np
import sklearn.decomposition
import matplotlib.pyplot as plt

#https://medium.com/ai%E5%8F%8D%E6%96%97%E5%9F%8E/preprocessing-data-%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90-pca-%E5%8E%9F%E7%90%86%E8%A9%B3%E8%A7%A3-afe1fd044d4f
"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None
        
        self.pca = sklearn.decomposition.PCA(n_components=n_components)

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        # self.pca.fit(X)
        # print(self.pca.components_)
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        c = X.transpose() @ X
        eig_val, eig_vec = np.linalg.eig(c)
        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(X.shape[1])]
        eig_pairs.sort(key=lambda x : x[0], reverse=True)
        self.components = np.array([ele[1] for ele in eig_pairs[:self.n_components]]).transpose().astype(np.float32)
        # print(self.components)
    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        X = (self.components.transpose() @ (X - self.mean).transpose()).transpose()
        # X = self.pca.transform(X)
        return X

    def reconstruct(self, X):
        #TODO: 2%
        # X = self.pca.transform(np.reshape(X, (1, -1)))
        # X = self.pca.inverse_transform(np.reshape(X, (1, -1)))
        # return X
        return (self.components @ self.transform(X).transpose()).transpose() + self.mean
    
    def plot(self):
        plt.subplot(151)
        plt.imshow(np.reshape(self.mean, (61, 80)))
        plt.subplot(152)
        plt.imshow(np.reshape(self.components[:,0], (61, 80)))
        plt.subplot(153)
        plt.imshow(np.reshape(self.components[:,1], (61, 80)))
        plt.subplot(154)
        plt.imshow(np.reshape(self.components[:,2], (61, 80)))
        plt.subplot(155)
        plt.imshow(np.reshape(self.components[:,3], (61, 80)))
        plt.show()
