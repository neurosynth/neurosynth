
import numpy as np
import logging
from time import time
from neurosynth.base.dataset import Dataset
from neurosynth.analysis import reduce as nsr
from neurosynth.analysis import meta
from neurosynth.base.mask import Masker
from neurosynth.base import imageutils
from sklearn import cluster
from sklearn.metrics.pairwise import pairwise_distances
import os
import re
from os.path import join, basename, isdir
from copy import deepcopy
from shutil import copyfile
import simplejson as json

logger = logging.getLogger('neurosynth.cluster')

class Clusterer:

    def __init__(self, dataset, method='studies', global_mask=None, roi_mask=None,
            reference_mask=None, features=None, feature_threshold=0.0,
            min_voxels_per_study=None, min_studies_per_voxel=None,
            voxel_parcellation='ward', n_parcels=500, distance_metric='correlation',
            clustering_method='ward', output_dir='.', prefix=None,
            parcellation_kwargs={}, clustering_kwargs={}):
        """ Initialize Clusterer.
        Args:
            dataset: The Dataset instance to use for clustering.
            global_mask: An image defining the space to use for all analyses. If None, the 
                mask found in the Dataset will be used.
            roi_mask: An image that determines which voxels to cluster. All non-zero voxels
                will be included in the clustering analysis. When roi_mask is None, all 
                voxels in the global_mask (i.e., the whole brain) will be clustered. roi_mask
                can be an image filename, a nibabel image, or an already-masked array with 
                the same dimensions as the global_mask.
            reference_mask: An image defining the voxels to base the distance matrix 
                computation on. All non-zero voxels will be used to compute the distance
                matrix. For example, if the roi_mask contains voxels in only the insula, 
                and reference_mask contains voxels in only the cerebellum, then voxels in 
                the insula will be clustered based on the similarity of their coactvation 
                with all and only cerebellum voxels.
            features: Optional features to use for selecting a subset of the studies in the 
                Dataset instance. If dataset is a numpy matrix, will be ignored.
            feature_threshold: float; the threshold to use for feature selection. Will be 
                ignored if features is None.
            min_voxels_per_study: An optional integer. If provided, all voxels with fewer 
                than this number of studies will be removed from analysis.
            min_studies_per_voxel: An optional integer. If provided, all studies with fewer 
                than this number of active voxels will be removed from analysis. 
            voxel_parcellation: Either a scikit-learn object with a fit_transform method,
                or the name of the parcellation method to use for reducing the dimensionality 
                of the reference mask. Valid options include:
                    None: no parcellation
                    'ward': spatially constrained hierarchical clustering; see Thirion
                        et al (2014)
                    'pca': principal component analysis
                    'grid': downsample the reference mask to an isometric grid
                Defaults to 'ward'. Note that parcellation will only be used if method 
                is set to 'coactivation' (i.e., it will be ignored by default).
            n_parcels: Number of parcels to request, if using a voxel_parcellation method.
                Meaning depends on parcellation algorithm.
            distance_metric: Optional string providing the distance metric to use for 
                computation of a distance matrix. When None, no distance matrix is computed
                and we assume that clustering will be done on the raw data. Valid options 
                are any of the strings accepted by sklearn's pairwise_distances method.
                Defaults to 'correlation'. Note that for some clustering methods (e.g., k-means), 
                no distance matrix will be computed, and this argument will be ignored.
            clustering_method: Algorithm to use for clustering. Must be one of 'ward', 'spectral',
                'agglomerative', 'dbscan', 'kmeans', or 'minik'. If None, can be set 
                later via set_algorithm() or cluster().
            output_directory: Directory to use for writing all outputs.
            prefix: Optional prefix to prepend to all outputted directories/files.
            parcellation_kwargs: Optional keyword arguments to pass to parcellation object.
            clustering_kwargs: Optional keyword arguments to pass to clustering object.
        """
        
        self.output_dir = output_dir
        self.prefix = prefix

        # Save all arguments for metadata output
        self.args = {}
        for a in (['clustering_method', 'output_dir', 'features', 'feature_threshold',
                    'global_mask', 'roi_mask', 'reference_mask', 'min_voxels_per_study',
                    'min_studies_per_voxel', 'distance_metric', 'voxel_parcellation',
                    'n_parcels'
                    ] + clustering_kwargs.keys() + parcellation_kwargs.keys()):
            self.args[a] = locals()[a]

        self.set_algorithm(clustering_method, **clustering_kwargs)

        self.dataset = dataset

        self.masker = dataset.masker if global_mask is None else Masker(global_mask)

        # Condition study inclusion on specific features
        if features is not None:
            data = self.dataset.get_ids_by_features(features, threshold=feature_threshold, 
                get_image_data=True)
        else:
            data = self.dataset.get_image_data()

        # Trim data based on minimum number of voxels or studies
        if min_studies_per_voxel is not None:
            logger.info("Thresholding voxels based on number of studies.")
            av = self.masker.unmask(data.sum(1) > min_studies_per_voxel, output='vector')
            self.masker.add(av)

        if min_voxels_per_study is not None:
            logger.info("Thresholding studies based on number of voxels.")
            active_studies = np.where(data.sum(0) > min_voxels_per_study)[0]
            data = data[:, active_studies]
        
        if method == 'coactivation':
            # Set reference voxels, defaulting to whole brain
            if reference_mask is not None:
                self.masker.add(reference_mask)
            ref_vox = self.masker.get_current_mask()
            self.reference_data = data[ref_vox,:]
            if reference_mask is not None: self.masker.remove(-1)

            # Dimensionality reduction
            if voxel_parcellation is not None:
                if hasattr(voxel_parcellation, 'fit_transform'):
                    self.reference_data = voxel_parcellation.fit_transform(self.reference_data.T).T
                elif voxel_parcellation == 'grid':
                        self.reference_data, _ = nsr.apply_grid(self.reference_data, masker=self.masker, scale=grid_scale, threshold=None)
                else:
                    vox = self.masker.get_current_mask()
                    self.reference_data = self.data[vox,:]
        
        # Set the voxels to cluster
        if roi_mask is not None: self.masker.add(roi_mask)
        self.roi_data = data[self.masker.get_current_mask(), :]
        # if roi_mask is not None: self.masker.remove(-1)
        if distance_metric is not None:
            self.create_distance_matrix(distance_metric=distance_metric)


    def create_distance_matrix(self, distance_metric='correlation', affinity=False, figure_file=None, 
                                distance_file=None):
        """ Creates a distance matrix of each grid roi across studies in Neurosynth Dataset.
        Args:
            distance_metric: The distance metric to use; see scipy documentation for available 
                metrics. Defaults to Jaccard Distance.
            affinity: If True, converts distance to affinity matrix (1 - distance).
            figure_file: Filename for output image of the clustered data. If None, no image is written.
            distance_file: Filename for output of the distance matrix. If None, matrix is not saved.
        """
        t = time()
        logger.info('Creating distance matrix using ' + distance_metric)
        Y = self.reference_data if hasattr(self, 'reference_data') else None
        dist = pairwise_distances(self.roi_data, Y=Y, metric=distance_metric)
        logger.info('Distance matrix computation took %.1f seconds.' % (time()-t))
        if figure_file is not None:
            plt.imshow(dist,aspect='auto',interpolation='nearest')
            plt.savefig(figure_file)
        if distance_file is not None:
            np.savetxt(distance_file, dist)
        if affinity:
            dist = 1.0 - dist
        self.distance_matrix = dist


    def cluster(self, algorithm=None, n_clusters=10, save_images=True, 
                precomputed_distances=False, bundle=False, coactivation_maps=False,
                **kwargs):
        """
        Args:
            algorithm: Algorithm to use for clustering. Must be one of 'ward', 'spectral',
                'agglomerative', 'dbscan', 'kmeans', or 'minik'. If None, uses the algorithm
                passed at initialization.
            n_clusters: Number of clusters to extract. Can be an integer or a list
                of integers to iterate.
            save_images: Boolean indicating whether or not to save images to file.
            precomputed_distances: Indicates whether or not to use precomputed distances in 
                the clustering. If True, the distance_matrix stored in the instance will be 
                used; when False (default), the raw data will be used.
            kwargs: Optional arguments to pass onto the scikit-learn clustering object.
        """
        if algorithm is not None:
            self.set_algorithm(algorithm, **kwargs)

        if isinstance(n_clusters, int):
            n_clusters = [n_clusters]

        clusterer = self.clusterer

        for k in n_clusters:
            # Set n_clusters for algorithms that allow it
            if hasattr(clusterer, 'n_clusters'):
                clusterer.n_clusters = k

            # Now figure out if we need to pass in raw data or a distance matrix.
            if precomputed_distances:
                if not hasattr(self, 'distance_matrix'):
                    raise ValueError("No precomputed distance matrix exists. Either set precomputed_distances to False, " +
                                    "or call the create_distance_matrix method before trying to cluster.")
                X = self.distance_matrix

                # Tell clusterer not to compute a distance/affinity matrix
                if hasattr(clusterer, 'affinity'):  # SpectralClustering and AffinityPropagation
                    clusterer.affinity = 'precomputed'
                elif hasattr(clusterer, 'metric'):  # DBSCAN
                    clusterer.metric = 'precomputed'

            else:
                X = self.roi_data

            labels = clusterer.fit_predict(X) + 1

            if save_images:
                self._create_cluster_images(labels, coactivation_maps)

            if bundle:
                # Generate metadata
                metadata = deepcopy(self.args)
                metadata.update({
                    'n_clusters': k,
                    'precomputed_distances': precomputed_distances,
                    })

                # Copy mask images
                for img in ['global_mask', 'roi_mask', 'reference_mask']:
                    if metadata[img] is not None:
                        ext = re.search('.nii(.gz)*$', metadata[img]).group()
                        copyfile(metadata[img], join(self.output_dir, img + ext))
                        metadata[img] = basename(metadata[img]) # Strip path

                # Write metadata
                json.dump(metadata, open(join(self.output_dir, 'metadata.json'), 'w'))



    def set_algorithm(self, algorithm, **kwargs):
        """ Set the algorithm to use in subsequent cluster analyses.
        Args:
            algorithm: The clustering algorithm to use. Either a string or an (uninitialized)
                scikit-learn clustering object. If string, must be one of 'ward', 'spectral', 
                'agglomerative', 'dbscan', 'kmeans', or 'minik'.
            kwargs: Additional keyword arguments to pass onto the scikit-learn clustering
                object.
        """

        self.algorithm = algorithm

        if isinstance(algorithm, basestring):

            algs = {
                'ward': cluster.Ward,
                'spectral': cluster.SpectralClustering,
                'agglomerative': cluster.AgglomerativeClustering,
                'kmeans': cluster.KMeans,
                'minik': cluster.MiniBatchKMeans,
                'affprop': cluster.AffinityPropagation,
                'dbscan': cluster.DBSCAN
            }

            if algorithm not in algs.keys():
                raise ValueError("Invalid clustering algorithm name. Valid options are 'ward'," + 
                    "'spectral', 'kmeans', 'minik', 'agglomerative', 'affprop', or 'dbscan'.")

            algorithm = algs[algorithm]

        self.clusterer = algorithm(**kwargs)


    # def plot_distance_by_cluster(self):
    #     ''' Creates a figure of distance matrix sorted by cluster solution. '''
    #     lab = pd.DataFrame(labels)
    #     lab.columns = ['cluster']
    #     lab['cluster'].sort()
    #     csort = list(lab['cluster'].index)
    #     orderedc = distance_matrix[:,csort]
    #     orderedc = orderedc[csort,:]
    #     plt.imshow(orderedc,aspect='auto',interpolation='nearest')
    #     plt.savefig(figname)

    # def plot_silhouette_scores(self):
    #     pass

    def _create_cluster_images(self, labels, coactivation_maps):
        ''' Creates a Nifti image of reconstructed cluster labels. 
        Args:
            labels: A vector of cluster labels
        Outputs:
            Cluster_k.nii.gz: Will output a nifti image with cluster labels
        '''
        # Reconstruct grid into original space
        # TODO: replace with masker.unmask()
        if hasattr(self, 'grid'):
            regions = self.masker.mask(self.grid)
            unique_regions = np.unique(regions)
            n_regions = unique_regions.size
            m = np.zeros(regions.size)
            for i in range(n_regions):
                m[regions == unique_regions[i]] = labels[i] + 1

            labels = m

        clusters = np.unique(labels)
        n_clusters = len(clusters)

        prefix = '' if self.prefix is None else self.prefix + '_'
        output_dir = join(self.output_dir, prefix + self.algorithm + '_k' + str(n_clusters))

        if not isdir(output_dir):
            os.makedirs(output_dir)

        outfile = join(output_dir,prefix + self.algorithm + '_k' + str(n_clusters) + 'cluster_labels.nii.gz')
        imageutils.save_img(labels, outfile, self.masker)

        # Generate a coactivation map for each cluster
        if coactivation_maps:
            coact_dir = join(output_dir, 'coactivation')
            if not isdir(coact_dir):
                os.makedirs(coact_dir)
            for c in clusters:
                img = np.zeros_like(labels)
                img[labels==c] = 1
                img = self.masker.unmask(img)
                ids = self.dataset.get_ids_by_mask(img, 0.25)
                ma = meta.MetaAnalysis(self.dataset, ids)
                ma.save_results(coact_dir, 'cluster_%d' % c)

