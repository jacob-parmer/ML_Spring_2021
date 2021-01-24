"""
Author: Jacob Parmer
Date: Jan 24, 2021

Most (basically all) of this code was taken from Sebastian Raschka's 2014 article:
'Implementating a Principal Component Analysis (PCA)' - 
https://sebastianraschka.com/Articles/2014_pca_step_by_step.html

"""


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

class PCA3D:

    def __init__(self, seed):
        np.random.seed(seed)
        
        mu_vec1 = np.array([0,0,0])
        cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T

        mu_vec2 = np.array([1,1,1])
        cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T

        self.all_samples = np.concatenate((self.class1_sample, self.class2_sample), axis=1)

        self.fig = plt.figure(figsize=(8,8))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def display_samples(self):
        plt.rcParams['legend.fontsize'] = 10

        self.ax.plot(self.class1_sample[0,:], self.class1_sample[1,:], self.class1_sample[2,:], 'o', markersize=8, color='blue', 
                alpha=0.5, label='class1')
        self.ax.plot(self.class2_sample[0,:], self.class2_sample[1,:], self.class2_sample[2,:], '^', markersize=8, color='red',
                alpha=0.5, label='class2')

        plt.title('Samples for class 1 and class 2')
        self.ax.legend(loc='upper right')

        plt.show()

    def get_mean_vector(self):
        mean_x = np.mean(self.all_samples[0,:])
        mean_y = np.mean(self.all_samples[1,:])
        mean_z = np.mean(self.all_samples[2,:])

        mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

        return mean_vector

    def get_scatter_matrix(self, mean_vector):
        scatter_matrix = np.zeros((3,3))
        for i in range(self.all_samples.shape[1]):
            scatter_matrix += (self.all_samples[:,i].reshape(3,1) - mean_vector).dot((
                               self.all_samples[:,i].reshape(3,1) - mean_vector).T)

        return scatter_matrix

    def get_covariance_matrix(self):
        cov_mat = np.cov([self.all_samples[0,:], self.all_samples[1,:], self.all_samples[2,:]])
        return cov_mat

    def get_eigens(self, matrix):
        eig_val, eig_vec = np.linalg.eig(matrix)
        return eig_val, eig_vec

    def sort_eigens(self, eig_val, eig_vec):
        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        return eig_pairs

    def reduce_eigens(self, eig_pairs):
        matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
        return matrix_w

    def transform_samples(self, matrix_w):
        transformed = matrix_w.T.dot(self.all_samples)
        return transformed

    def display_transformed_samples(self, transformed):
        plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
        plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
        plt.xlim([-4,4])
        plt.ylim([-4,4])
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.legend()
        plt.title('Transformed samples with class labels')

        plt.show()

    def PCA(self):
        covariance_matrix = self.get_covariance_matrix()
        eig_val, eig_vec = self.get_eigens(covariance_matrix)
        eig_pairs = self.sort_eigens(eig_val, eig_vec)
        matrix_w = self.reduce_eigens(eig_pairs)
        transformed = self.transform_samples(matrix_w)
        
        return transformed

def main():

    pca = PCA3D(seed=8675309)
    pca.display_samples()
    
    transformed = pca.PCA()

    pca.display_transformed_samples(transformed)

    return

if __name__ == "__main__":
    main()
