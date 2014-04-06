
import numpy as np
import logging
from time import time
from neurosynth.base.dataset import Dataset
from neurosynth.analysis import reduce as nsr
from neurosynth.base.mask import Mask
from neurosynth.base import imageutils
from sklearn import cluster
import os

logger = logging.getLogger('neurosynth.cluster')

class Clusterer:

    def __init__(self, dataset=None, algorithm=None, output_dir='.',  grid_scale=None,
            features=None, feature_threshold=0.0, global_mask=None, roi_mask=None, 
            distance_mask=None, min_voxels_per_study=None, min_studies_per_voxel=None, 
            **kwargs):
        """ Initialize Clusterer.
        Args:
            dataset: The dataset to use for clustering. Either a Dataset instance or a numpy
                array with voxels in rows and features in columns.
            algorithm: Optional algorithm to use for clustering. If None, an algorithm 
                must be passed to the cluster() method later.
            output_directory: Directory to use for writing all outputs.
            grid_scale: Optional integer. If provided, a 3D grid will be applied to the 
                image data, with values in all voxels in each grid cell being averaged 
                prior to clustering analysis. This is an effective means of dimension 
                reduction in cases where the data are otherwise too large for clustering.
            features: Optional features to use for selecting a subset of the studies in the 
                Dataset instance. If dataset is a numpy matrix, will be ignored.
            feature_threshold: float; the threshold to use for feature selection. Will be 
                ignored if features is None.
            global_mask: An image defining the space to use for all analyses. Only necessary
                if dataset is a numpy array.
            roi_mask: An image that determines which voxels to cluster. All non-zero voxels
                will be included in the clustering analysis. When roi_mask is None, all 
                voxels in the global_mask (i.e., the whole brain) will be clustered.
            distance_mask: An image defining the voxels to base the distance matrix 
                computation on. All non-zero voxels will be used to compute the distance
                matrix. For example, if the roi_mask contains voxels in only the insula, 
                and distance_mask contains voxels in only the cerebellum, then voxels in 
                the insula will be clustered based on the similarity of their coactvation 
                with all and only cerebellum voxels.
            min_voxels_per_study: An optional integer. If provided, all voxels with fewer 
                than this number of studies will be removed from analysis.
            min_studies_per_voxel: An optional integer. If provided, all studies with fewer 
                than this number of active voxels will be removed from analysis.
            **kwargs: Additional keyword arguments to pass to the clustering algorithm.

        """
        self.output_dir = output_dir

        if algorithm is not None:
            self._set_clustering_algorithm(algorithm, **kwargs)

        if isinstance(dataset, Dataset):

            self.dataset = dataset

            if global_mask is None:
                self.masker = dataset.volume

            if features is not None:
                data = self.dataset.get_ids_by_features(features, threshold=feature_threshold, 
                            get_image_data=True)
            else:
                data = self.dataset.get_image_data()

            if min_studies_per_voxel is not None:
                logger.info("Thresholding voxels based on number of studies.")
                sum_vox = data.sum(1)
                # Save the indices for later reconstruction
                active_vox = np.where(sum_vox > min_studies_per_voxel)[0]  
                n_active_vox = active_vox.shape[0]

            if min_voxels_per_study is not None:
                logger.info("Thresholding studies based on number of voxels.")
                sum_studies = data.sum(0)
                active_studies = np.where(sum_studies > min_voxels_per_study)[0]
                n_active_studies = active_studies.shape[0]

            if min_studies_per_voxel is not None:
                logger.info("Selecting voxels with more than %d studies." % min_studies_per_voxel)
                data = data[active_vox, :]

            if min_voxels_per_study is not None:
                logger.info("Selecting studies with more than %d voxels." % min_voxels_per_study)
                data = data[:, active_studies]

            self.data = data

        else:
            self.data = dataset

            if global_mask is None:
                raise ValueError("If dataset is a numpy array, a valid global_mask (filename, " +
                    "Mask instance, or nibabel image) must be passed.")

            if not isinstance(global_mask, Mask):
                global_mask = Mask(global_mask)
            self.masker = global_mask

        if grid_scale is not None:
            self.data, self.grid = nsr.apply_grid(self.data, masker=self.masker, scale=grid_scale, threshold=None)


    def create_distance_matrix(self, distance_metric='jaccard', figure_file=None, save_distance=None):
        """ Creates a distance matrix of each grid roi across studies in Neurosynth Dataset.
        Args:
            distance_metric: The distance metric to use; see scipy documentation for available 
                metrics. Defaults to Jaccard Distance.
            figure_file: Filename for output image of the clustered data. If None, no image is written.
            distance_file: Filename for output of the distance matrix. If None, matrix is not saved.
        """
        from sklearn.metrics.pairwise import pairwise_distances
        t = time()
        logger.info('Creating distance matrix using ' + distance_metric)
        dist = pairwise_distances(self.data, metric=distance_metric)
        logger.info('Distance matrix computation took %.1f seconds.' % (time()-t))
        if figure_file is not None:
            plt.imshow(dist,aspect='auto',interpolation='nearest')
            plt.savefig(figure_file)
        if distance_file is not None:
            np.savetxt(distance_file, dist)
        self.distance_matrix = dist


    def cluster(self, algorithm=None, n_clusters=10, save_images=True, **kwargs):
        """
        Args:
            algorithm: Optional clustering algorithm to use (see _set_clustering_algorithm
                for details). If None (default), use algorithm passed to the Clusterer
                instance at initialization.
            num_clusters: Number of clusters to extract. Can be an integer or a list
                of integers to iterate.
        """
        if algorithm is not None:
            self._set_clustering_algorithm(algorithm, **kwargs)
        elif not hasattr(self, 'clusterer'):
                raise ValueError("You must provide a valid clustering algorithm.")

        if isinstance(n_clusters, int):
            n_clusters = [n_clusters]

        clusterer = self.clusterer

        for k in n_clusters:
            # Set n_clusters for algorithms that allow it
            if hasattr(clusterer, 'n_clusters'):
                clusterer.n_clusters = k

            # Now figure out if we need to pass in raw data or a distance matrix.
            if False:
                if not hasattr(self, 'distance_matrix'):
                    self.create_distance_matrix()  # Fix
                X = self.distance_matrix
            else:
                X = self.data
            labels = clusterer.fit_predict(X)

            if save_images:
                self._create_cluster_images(labels)


    def _set_clustering_algorithm(self, algorithm, **kwargs):
        """ Set the algorithm to use in subsequent cluster analyses.
        Args:
            algorithm: The clustering algorithm to use. Either a string or an (uninitialized)
                scikit-learn clustering object. If string, must be one of 'ward', 'spectral', 
                'kmeans', or 'minik'.
        """
        if isinstance(algorithm, basestring):

            algs = {
                'ward': cluster.Ward,
                'spectral': cluster.SpectralClustering,
                'kmeans': cluster.KMeans,
                'minik': cluster.MiniBatchKMeans,
                'affprop': cluster.AffinityPropagation,
                'dbscan': cluster.DBSCAN
            }

            if algorithm not in algs.keys():
                raise ValueError("Invalid clustering algorithm name. Valid options are 'ward'," + 
                    "'spectral', 'kmeans', 'minik', 'affprop', or 'dbscan'.")

            algorithm = algs[algorithm]

        self.clusterer = algorithm(**kwargs)


    def plot_distance_by_cluster(self):
        ''' Creates a figure of distance matrix sorted by cluster solution. '''
        lab = pd.DataFrame(labels)
        lab.columns = ['cluster']
        lab['cluster'].sort()
        csort = list(lab['cluster'].index)
        orderedc = distance_matrix[:,csort]
        orderedc = orderedc[csort,:]
        plt.imshow(orderedc,aspect='auto',interpolation='nearest')
        plt.savefig(figname)

    def plot_silhouette_scores(self):
        pass

    def _create_cluster_images(self, labels):
        ''' Creates a Nifti image of reconstructed cluster labels. 
        Args:
            dataset: A pickled neurosynth dataset
            grid_image: A .nii grid image created using create_grid_image()
            labels: A vector of cluster labels 
            prefix: A string indicating path of output image 
        Outputs:
            Cluster_k.nii.gz: Will output a nifti image with cluster labels
        '''
        regions = self.masker.mask(self.grid)
        unique_regions = np.unique(regions)
        n_regions = unique_regions.size
        m = np.zeros(regions.size)
        for i in range(n_regions):
            m[regions == unique_regions[i]] = labels[i] + 1
        prefix = os.path.join(self.output_dir, 'ClusterImages')
        if not os.path.isdir(prefix):
            os.makedirs(prefix)
        outfile = os.path.join(prefix,'Cluster_k%d.nii.gz' % len(np.unique(labels)))
        imageutils.save_img(m, outfile, self.masker)
