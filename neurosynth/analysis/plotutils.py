# # emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# # ex: set sts=4 ts=4 sw=4 et:
# """Miscellaneous plotting functions"""

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.projections.polar import PolarAxes
# from matplotlib.projections import register_projection

# # BORROWED FROM PYPLOT EXAMPLES


# def radar_factory(num_vars, frame='circle'):
#     """Create a radar chart with `num_vars` axes."""
#     # calculate evenly-spaced axis angles
#     theta = 2 * np.pi * np.linspace(0, 1 - 1. / num_vars, num_vars)
#     # rotate theta such that the first axis is at the top
#     theta += np.pi / 2

#     def draw_poly_frame(self, x0, y0, r):
#         # TODO: use transforms to convert (x, y) to (r, theta)
#         verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
#         return plt.Polygon(verts, closed=True, edgecolor='k')

#     def draw_circle_frame(self, x0, y0, r):
#         return plt.Circle((x0, y0), r)

#     frame_dict = {'polygon': draw_poly_frame, 'circle': draw_circle_frame}
#     if frame not in frame_dict:
#         raise ValueError('unknown value for `frame`: %s' % frame)

#     class RadarAxes(PolarAxes):

#         """Class for creating a radar chart (a.k.a. a spider or star chart)

#         http://en.wikipedia.org/wiki/Radar_chart
#         """
#         name = 'radar'
#         # use 1 line segment to connect specified points
#         RESOLUTION = 1
#         # define draw_frame method
#         draw_frame = frame_dict[frame]

#         def fill(self, *args, **kwargs):
#             """Override fill so that line is closed by default"""
#             closed = kwargs.pop('closed', True)
#             return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

#         def plot(self, *args, **kwargs):
#             """Override plot so that line is closed by default"""
#             lines = super(RadarAxes, self).plot(*args, **kwargs)
#             for line in lines:
#                 self._close_line(line)

#         def _close_line(self, line):
#             x, y = line.get_data()
#             # FIXME: markers at x[0], y[0] get doubled-up
#             if x[0] != x[-1]:
#                 x = np.concatenate((x, [x[0]]))
#                 y = np.concatenate((y, [y[0]]))
#                 line.set_data(x, y)

#         def set_varlabels(self, labels):
#             self.set_thetagrids(theta * 180 / np.pi, labels)

#         def _gen_axes_patch(self):
#             x0, y0 = (0.5, 0.5)
#             r = 0.5
#             return self.draw_frame(x0, y0, r)

#     register_projection(RadarAxes)
#     return theta
