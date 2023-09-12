# -*- coding: utf-8 -*-

# Author:
#    Antonio López Martínez-Carrasco <antoniolopezmc1995@gmail.com>

"""This file contains the implementation of the algorithm of Trace-Based Clustering.
"""

from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from seaborn import heatmap
import matplotlib.pyplot as plt

class TraceBasedClustering (object):
    """This class represents the algorithm of Trace-Based Clustering.

    :type k: int
    :param k: Number of partitions to generate (from 'number_of_clusters = 2' to 'number_of_clusters = k').
    :type pandas_dataframe: pandas.DataFrame
    :param pandas_dataframe: Input dataset. It is VERY IMPORTANT to respect the following conditions:
        (1) the dataset must be a pandas dataframe;
        (2) the dataset must not contain missing values;
        (3) for each attribute, all its values must be of the same type.
    :type clustering_algorithm: str
    :param clustering_algorithm: Algorithm of clustering used inside Trace-Based Clustering. Possible values: 'kmeans'.
    :type match_function: str
    :param match_function: Match function used inside Trace-Based Clustering. Possible values: 'jaccard2' (coeficient of Jaccard2), 'dice' (coeficient of Dice).
    :type random_seed: int, default=1
    :param random_seed: Seed for generation of random numbers.
    """

    def __init__(self, k, clustering_algorithm, match_function, random_seed=1):
        """Method to initialize an object of type 'TraceBasedClustering'.
        """
        available_values_for_clustering_algorithm = ['kmeans']
        available_values_for_match_function = ['jaccard2', 'dice']
        if type(k) is not int:
            raise TypeError("Parameter 'k' must be an integer (type 'int').")
        if type(clustering_algorithm) is not str:
            raise TypeError("Parameter 'clustering_algorithm' must be a string (type 'str').")
        if clustering_algorithm not in available_values_for_clustering_algorithm:
            raise ValueError("Parameter 'clustering_algorithm' is not valid (see documentation).")
        if type(match_function) is not str:
            raise TypeError("Parameter 'match_function' must be a string (type 'str').")
        if match_function not in available_values_for_match_function:
            raise ValueError("Parameter 'match_function' is not valid (see documentation).")
        if type(random_seed) is not int:
            raise TypeError("Parameter 'random_seed' must be an integer (type 'int').")
        self.k = k
        self.clusteringAlgorithm = clustering_algorithm
        self.matchFunction = match_function
        self.randomSeed = random_seed

    def generateSetOfPartitions(self, pandas_dataframe):
        """Method to generate all partitions (set of partitions) from partition with 'number_of_clusters = 2' to partition with 'number_of_clusters = k'.

        :type pandas_dataframe: pandas.DataFrame
        :param pandas_dataframe: Input dataset. It is VERY IMPORTANT to respect the following conditions:
          (1) the dataset must be a pandas dataframe;
          (2) the dataset must not contain missing values;
          (3) for each attribute, all its values must be of the same type.
        :rtype: dict
        :return: a dictionary where every pair key-value corresponding with a partition and its clusters. The key with the number 'n' corresponding with the partition with 'n' clusters (so, the keys go from 2 to k). A value is a list with the clusters of that partition and every cluster is a list with the names/numbers of the rows of the original pandas dataframe (passed by parameter).
        """
        if type(pandas_dataframe) is not DataFrame:
            raise TypeError("Parameter 'pandas_dataframe' must be a pandas DataFrame.")
        # Final dictionary.
        dictionary_of_partitions = dict()
        if self.clusteringAlgorithm == 'kmeans':
            # The random seed will be different in every calling to clustering algorithm.
            current_random_seed = self.randomSeed
            # Generate all partitions (from 'number_of_cluster=2' to 'number_of_cluster=k').
            for number_of_clusters in range(2, self.k+1):
                # Create and run KMeans.
                kmeans_alg = KMeans(n_clusters=number_of_clusters, random_state=current_random_seed).fit(pandas_dataframe)
                # For the current partition, generate all its clusters (from 0 to 'number_of_clusters - 1').
                list_of_clusters = [] # List of lists.
                for cluster in range(0,number_of_clusters):
                    sub_dataframe = pandas_dataframe[kmeans_alg.labels_ == cluster] # Dataframe with only the elements/rows of the original pandas dataframe in the cluster number 'cluster'.
                    list_of_clusters.append( sub_dataframe.index.tolist() ) # Get the list of indexes of the sub_dataframe and add them to the list 'list_of_clusters'. 
                # current_random_seed + 3
                current_random_seed = current_random_seed + 3
                # Add partition to the dictionary.
                dictionary_of_partitions[number_of_clusters] = list_of_clusters
        else:
            raise ValueError("The clustering algorithm (" + self.clusteringAlgorithm + ") is not valid (see documentation).")
        return dictionary_of_partitions

    def generateMatrixOfMatches(self, set_of_partitions, save_to_file=None):
        """Method to generate the matrix of matches (as a pandas DataFrame) from the set of partitions (returned by the function 'generateSetOfPartitions').

        :type set_of_partitions: dict
        :param set_of_partitions: the set of all partitions obtained from the original pandas DataFrame (value returned by the function 'generateSetOfPartitions').
        :type save_to_file: str, default=None
        :param save_to_file: path for saving the matrix of matches (as a .csv file). IMPORTANT: the values (they are tuples) will be stored as strings.       
        :rtype: pandas.DataFrame
        :return: a pandas DataFrame where the values are tuples. A value in the column Y and in the row X (indexed with [X,Y]) is a tuple with 2 elements: (1) the value of match between a specific cluster of the partition with more clusters (partition with 'number_of_clusters = k'), indicated by the column, and the cluster with the highest value of match with another previous specific partition, indicated by the row, and (2) the label indicating the cluster with the highest value of match of that previous specific partition. For example: the value in ['partition9', 'partition20_cluster5'] correspond with a tuple of the form (a, b), where 'a' is the value of match between the cluster 5 of the partition 20 and the cluster with the highest value of match with the partition 9 (for example, cluster 7) and 'b' is the label 'cluster7'.
        """
        if type(set_of_partitions) is not dict:
            raise TypeError("Parameter 'set_of_partitions' must be a python dictionary (type 'dict').")
        # Final pandas DataFrame.
        matrix_of_matches = DataFrame()
        # Iterate over the partition with more clusters ('number_of_clusters_of_partition_k = k'): from cluster 0 to cluster 'number_of_clusters_of_partition_k - 1'.
        number_of_clusters_of_partition_k = self.k
        for cluster_of_partition_k in range(0,number_of_clusters_of_partition_k):
            # Get the cluster (names/numbers of the rows of the original pandas dataframe) as a python set (type 'set').
            cluster_of_partition_k_as_a_python_set = set(set_of_partitions[self.k][cluster_of_partition_k])
            # Temporal list (list of values of the new attribute of the final pandas dataframe).
            temporal_list = []
            # Iterate over the set of previous partitions to k: from partition 2 (with 2 clusters) to partition k-1 (with k-1 clusters).
            number_of_previous_partitions = self.k-1
            for current_partition in range(2,number_of_previous_partitions+1):
                # Maximum value of match between 'cluster_of_partition_k' and the current cluster of the current partition.
                max_value_of_match = -1 # Initially -1. IMPORTANT: This value is impossible and it will be always overwritten (in other case, some error has occurred).
                cluster_with_max_value_of_match = None # Initially None. This value is impossible and it will be always overwritten (in other case, some error has occurred).
                # Iterate over the current partition: from cluster 0 to cluster 'number_of_clusters_of_current_partition - 1'.
                number_of_clusters_of_current_partition = current_partition
                for current_cluster in range(0,number_of_clusters_of_current_partition):
                    # Get the cluster (names/numbers of the rows of the original pandas dataframe) as a python set (type 'set').
                    cluster_of_current_partition_as_a_python_set = set(set_of_partitions[current_partition][current_cluster])
                    ## COMPUTE THE MATCH FUNCTION ##
                    if (self.matchFunction == 'jaccard2'):
                        # Obtain the intersection between the elements of 'cluster_of_partition_k_as_a_python_set' and elements of 'cluster_of_current_partition_as_a_python_set'.
                        intersection = cluster_of_partition_k_as_a_python_set.intersection(cluster_of_current_partition_as_a_python_set)
                        # Match function.
                        value_of_match = len(intersection) / len(cluster_of_current_partition_as_a_python_set)
                        # Compare with the maximum match value.
                        if (value_of_match > max_value_of_match):
                            max_value_of_match = value_of_match
                            cluster_with_max_value_of_match = current_cluster
                    elif (self.matchFunction == 'dice'):
                        # Obtain the intersection between the elements of 'cluster_of_partition_k_as_a_python_set' and elements of 'cluster_of_current_partition_as_a_python_set'.
                        intersection = cluster_of_partition_k_as_a_python_set.intersection(cluster_of_current_partition_as_a_python_set)
                        # Match function.
                        value_of_match = (2*len(intersection)) / (len(cluster_of_partition_k_as_a_python_set)+len(cluster_of_current_partition_as_a_python_set))
                        # Compare with the maximum match value.
                        if (value_of_match > max_value_of_match):
                            max_value_of_match = value_of_match
                            cluster_with_max_value_of_match = current_cluster
                    else:
                        raise ValueError("The match function (" + self.matchFunction + ") is not valid (see documentation).")
                # Assign 'max_value_of_match' and 'cluster_with_max_value_of_match' to 'temporal_list'.
                temporal_list.append( (max_value_of_match, 'cluster'+str(cluster_with_max_value_of_match)) )
            # Add new attribute to 'matrix_of_matches'.
            matrix_of_matches['partition'+str(self.k)+'_cluster'+str(cluster_of_partition_k)] = Series(temporal_list)
        # Change the names of the rows to: 'partition2', 'partition3', ..., partition{k-1}.
        # - Currently, the names of the rows are: 0, 1, 2, ..., k-3.
        matrix_of_matches.rename(index=(lambda x: 'partition'+str(x+2)), inplace=True)
        # Save the matrix of matches (if it is the case).
        if save_to_file:
            matrix_of_matches.to_csv(save_to_file, index_label="row_name") # In this case, the row names are also stored (because they are not only numbers and contain important information).
        return matrix_of_matches

    def visualizeMatrixOfMatches(self, matrix_of_matches, figsize=None, save_to_file=None):
        """Method to visualize the matrix of matches returned by the function 'generateMatrixOfMatches' with a heat map.

        :type matrix_of_matches: pandas.DataFrame
        :param matrix_of_matches: the matrix of matches returned by the function 'generateMatrixOfMatches'.
        :type figsize: tuple
        :param figsize: Tuple of the form (width in inches, height in inches).
        :type save_to_file: str, default=None
        :param save_to_file: path for saving the heat map (as a .png file).
        :rtype: matplotlib.axes
        :return: heat map.
        """
        if type(matrix_of_matches) is not DataFrame:
            raise TypeError("Parameter 'matrix_of_matches' must be a pandas DataFrame.")
        # For each cell of the matrix of matches (tuple), we only need the first element (value of match).
        matrix = matrix_of_matches.applymap( lambda value : value[0] )
        # To avoid long names in the final heat map, we are going to change the names of the rows and the names of the columns.
        matrix.index = ['p'+str(i) for i in range(2,self.k)] # From partition 2 (with 2 clusters) to partition k-1 (with k-1 clusters).
        matrix.columns = ['p'+str(self.k)+'_c'+str(i) for i in range(0,self.k)] # From cluster 0 to cluster 'number_of_clusters_of_partition_k - 1' ('number_of_clusters_of_partition_k' = k).
        # Create the heat map.
        plt.clf() # If there are previous data, clean them.
        if (figsize != None):
            plt.figure(figsize=figsize)
        heat_map_axes = heatmap(matrix, cbar_kws = dict(use_gridspec=False,location="left"))
        heat_map_axes.set_position([heat_map_axes.get_position().x0, heat_map_axes.get_position().y0, heat_map_axes.get_position().width * 0.85, heat_map_axes.get_position().height * 0.85])
        heat_map_axes.xaxis.tick_top()
        heat_map_axes.yaxis.tick_right()
        for tick in heat_map_axes.get_xticklabels():
            tick.set_rotation(90)
        for tick in heat_map_axes.get_yticklabels():
            tick.set_rotation(0)
        # Save the heat map (if it is the case).
        if save_to_file:
            heat_map_axes.get_figure().savefig(save_to_file)
        return heat_map_axes

    def selectClustersByMeanMedian(self, matrix_of_matches, mean_greater_or_equal_than, median_greater_or_equal_than, criterion='mean_and_median'):
        """Method to select the most 'interesting' or 'valuable' clusters of the partition with more clusters (partition with 'number_of_clusters = k') using the mean and/or the median of the columns of 'matrix_of_matches'.

        :type matrix_of_matches: pandas.DataFrame
        :param matrix_of_matches: the matrix of matches returned by the function 'generateMatrixOfMatches'.
        :type mean_greater_or_equal_than: float
        :param mean_greater_or_equal_than: minimum mean that need to have a column of 'matrix_of_matches' to select the corresponding cluster.
        :type median_greater_or_equal_than: float
        :param median_greater_or_equal_than: minimum median that need to have a column of 'matrix_of_matches' to select the corresponding cluster.
        :type criterion: str, default='mean_and_median'
        :param criterion: measures to consider for selecting a cluster as 'interesting' or 'valuable'. Possible values: 'mean_and_median' (mean and median), 'mean_or_median' (mean or median), 'only_mean', 'only_median' and 'no_measure' (select all clusters of the partition with more clusters no matter tha mean or median).
        :rtype: list
        :return: a python list with the names of the most 'interesting' or 'valuable' clusters of the partition with more clusters (partition with 'number_of_clusters = k'). The elements of the list are tuples that contain: (1) the name of the 'interesting' or 'valuable' cluster, (2) the mean of its corresponding column of 'matrix_of_matches' and (3) the median of its corresponding column of 'matrix_of_matches'. 
        """
        available_values_for_criterion = ['mean_and_median', 'mean_or_median', 'only_mean', 'only_median', 'no_measure']
        if type(matrix_of_matches) is not DataFrame:
            raise TypeError("Parameter 'matrix_of_matches' must be a pandas DataFrame.")
        if type(mean_greater_or_equal_than) is not float:
            raise TypeError("Parameter 'mean_greater_or_equal_than' must be a float.")
        if type(median_greater_or_equal_than) is not float:
            raise TypeError("Parameter 'median_greater_or_equal_than' must be a float.")
        if type(criterion) is not str:
            raise TypeError("Parameter 'criterion' must be a string (type 'str').")
        if criterion not in available_values_for_criterion:
            raise ValueError("Parameter 'criterion' is not valid (see documentation).")
        # Final list.
        final_list = []
        # For each cell of the matrix of matches (tuple), we only need the first element (value of match).
        matrix = matrix_of_matches.applymap( lambda value : value[0] )
        # Iterator over the columns of 'matrix'.
        if (criterion == 'mean_and_median'):
            for column in matrix.columns:
                # Get the current column as a Series.
                current_column = matrix.loc[:, column]
                # Get mean and median.
                current_column_mean = current_column.mean()
                current_column_median = current_column.median()
                # Check the condition.
                if (current_column_mean >= mean_greater_or_equal_than) and (current_column_median >= median_greater_or_equal_than):
                    final_list.append( (column, "mean="+str(current_column_mean), "median="+str(current_column_median)) )
        elif (criterion == 'mean_or_median'):
            for column in matrix.columns:
                # Get the current column as a Series.
                current_column = matrix.loc[:, column]
                # Get mean and median.
                current_column_mean = current_column.mean()
                current_column_median = current_column.median()
                # Check the condition.
                if (current_column_mean >= mean_greater_or_equal_than) or (current_column_median >= median_greater_or_equal_than):
                    final_list.append( (column, "mean="+str(current_column_mean), "median="+str(current_column_median)) )
        elif (criterion == 'only_mean'):
            for column in matrix.columns:
                # Get the current column as a Series.
                current_column = matrix.loc[:, column]
                # Get mean and median.
                current_column_mean = current_column.mean()
                current_column_median = current_column.median()
                # Check the condition.
                if (current_column_mean >= mean_greater_or_equal_than):
                    final_list.append( (column, "mean="+str(current_column_mean), "median="+str(current_column_median)) )
        elif (criterion == 'only_median'):
            for column in matrix.columns:
                # Get the current column as a Series.
                current_column = matrix.loc[:, column]
                # Get mean and median.
                current_column_mean = current_column.mean()
                current_column_median = current_column.median()
                # Check the condition.
                if (current_column_median >= median_greater_or_equal_than):
                    final_list.append( (column, "mean="+str(current_column_mean), "median="+str(current_column_median)) )
        elif (criterion == 'no_measure'):
            for column in matrix.columns:
                # Get the current column as a Series.
                current_column = matrix.loc[:, column]
                # Get mean and median.
                current_column_mean = current_column.mean()
                current_column_median = current_column.median()
                # Add to the list.
                final_list.append( (column, "mean="+str(current_column_mean), "median="+str(current_column_median)) )
        else:
            raise ValueError("The criterion (" + criterion + ") is not valid (see documentation).")
        return final_list
