""" Classes representing article-like data that can be mapped to images. """

import logging
import json
from neurosynth.base import imageutils
from neurosynth.base import transformations

logger = logging.getLogger('neurosynth.mappable')


class Mappable(object):

    def __init__(self, data, transformer=None):
        try:
            self.data = data.copy().reset_index()
            self.id = data['id'].values[0]
            # If space is not explicitly set, assume the coordinates are
            # already in the target space.
            self.space = data['space'].values[
                0] if 'space' in data.columns else transformer.target
        except Exception as e:
            logger.error("Missing ID and/or space fields. "
                         "Please check database file, caught: %s" % str(e))
            exit()

        peaks = data[['x', 'y', 'z']].values

        # Convert between stereotactic spaces
        if transformer is not None and self.space != transformer.target:
            peaks = transformer.apply(self.space, peaks)

        # Convert from XYZ coordinates to matrix indices, saving both
        self.xyz = peaks
        self.peaks = transformations.xyz_to_mat(peaks)

    def map_peaks(self):
        """Map all Peaks to a new Nifti1Image."""
        return imageutils.map_peaks_to_image(self.peaks)

    def to_json(self, filename=None):
        json_string = json.dumps(
            {
                'id': self.id,
                'space': self.space,
                'peaks': self.xyz.tolist()
            })
        if filename is not None:
            open(filename, 'w').write(json_string)
        else:
            return json_string

    def to_s(self):
        s = "Mappable ID: %s\n" % self.id
        s += "Nominal space: %s\n" % self.space
        s += "Num. of peaks: %s\n\n" % str(self.peaks.shape[0])
        s += "Peaks:\n\n"
        for p in self.xyz.tolist():
            s += "\t%s\n" % str(p)
        return s


class Article(Mappable):

    def __init__(self, data, transformer=None):
        super(Article, self).__init__(data, transformer)


class Table(Mappable):

    def __init__(self, data, transformer=None, article=None):
        self.article = article
        super(Table, self).__init__(data, transformer)
