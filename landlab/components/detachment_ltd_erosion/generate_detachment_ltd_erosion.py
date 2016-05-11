"""

Landlab component that simulates detachment limited sediment transport is more
general than the stream power component. Doesn't require the upstream node
order, links to flow receiver and flow receiver fields. Instead, takes in
the discharge values on NODES calculated by the OverlandFlow class and
erodes the landscape in response to the output discharge.

.. codeauthor:: Jordan Adams

Examples
--------
>>> import numpy as np
>>> from landlab import RasterModelGrid
>>> from landlab.components.detachment_ltd_erosion.generate_detachment_ltd_erosion import DetachmentLtdErosion

Create a grid on which to calculate detachment ltd sediment transport.

>>> grid = RasterModelGrid((4, 5))

The grid will need some data to provide the detachment limited sediment
transport component. To check the names of the fields that provide input to
the detachment ltd transport component, use the *input_var_names* class
property.

>>> DetachmentLtdErosion.input_var_names
('topographic__elevation', 'topographic__slope', 'water__discharge')

Create fields of data for each of these input variables.

>>> grid.at_node['topographic__elevation'] = np.array([
...     0., 0., 0., 0., 0.,
...     1., 1., 1., 1., 1.,
...     2., 2., 2., 2., 2.,
...     3., 3., 3., 3., 3.])

Using the set topography, now we will calculate slopes on all nodes.


>>> grid.at_node['topographic__slope'] = np.array([
...     -0.        , -0.        , -0.        , -0.        , -0,
...      0.70710678,  1.        ,  1.        ,  1.        ,  0.70710678,
...      0.70710678,  1.        ,  1.        ,  1.        ,  0.70710678,
...     0.70710678,  1.        ,  1.        ,  1.        ,  0.70710678])


Now we will arbitrarily add water discharge to each node for simplicity.
>>> grid.at_node['water__discharge'] = np.array([
...     30., 30., 30., 30., 30.,
...     20., 20., 20., 20., 20.,
...     10., 10., 10., 10., 10.,
...     5., 5., 5., 5., 5.])

Instantiate the `DetachmentLtdErosion` component to work on this grid, and
run it. In this simple case, we need to pass it a time step ('dt')

>>> dt = 10.0
>>> dle = DetachmentLtdErosion(grid)
>>> dle.erode(dt = dt)

After calculating the erosion rate, the elevation field is updated in the
grid. Use the *output_var_names* property to see the names of the fields that
have been changed.

>>> dle.output_var_names
('topographic__elevation',)

The `topographic__elevation` field is defined at nodes.

>>> dle.var_loc('topographic__elevation')
'node'


Now we test to see how the topography changed as a function of the erosion
rate.

>>> grid.at_node['topographic__elevation'] # doctest: +NORMALIZE_WHITESPACE
array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.99993675,  0.99991056,  0.99991056,  0.99991056,  0.99993675,
        1.99995528,  1.99993675,  1.99993675,  1.99993675,  1.99995528,
        2.99996838,  2.99995528,  2.99995528,  2.99995528,  2.99996838])

"""

from landlab import Component
import pylab
import numpy as np
from matplotlib import pyplot as plt
from landlab.field.scalar_data_fields import FieldError

class DetachmentLtdErosion(Component):
    '''  Landlab component that simulates detachment-limited river erosion.

This component calculates changes in elevation in response to vertical incision.


    '''
    _name = 'DetachmentLtdErosion'

    _input_var_names = set([
        'topographic__elevation',
        'topographic__slope',
        'water__discharge',
    ])
    _output_var_names = set([
        'topographic__elevation',
    ])

    _var_units = {
        'topographic__elevation': 'm',
        'topographic__slope': '-',
        'water__discharge': 'm**3/s',
    }

    _var_mapping = {
        'topographic__elevation': 'node',
        'topographic__slope': 'node',
        'water__discharge': 'node',
    }

    _var_doc = {
        'topographic__elevation': 'Land surface topographic elevation',
        'topographic__slope': 'Slope of ',
        'water__discharge': 'node',
    }

    def __init__(self, grid, K_sp = 0.00002, m_sp = 0.5, n_sp = 1.0,
                     uplift_rate = 0.0, entraiment_threshold = 0.0, **kwds):


        """
        Calculate detachment limited erosion rate on nodes.

        Landlab component that generalizes the detachment limited erosion
        equation, primarily to be coupled to the the Landlab OverlandFlow
        component.

        This component adjusts topographic elevation and is contained in the
        landlab.components.detachment_ltd_sed_trp folder.

        Parameters
        ----------
        grid : RasterModelGrid
            A landlab grid.
        K_sp : float, optional
            K in the stream power equation (units vary with other parameters -
            if used with the de Almeida equation it is paramount to make sure
            the time component is set to SECONDS, not YEARS!)
        m_sp : float, optional
            Stream power exponent, power on discharge
        n_sp : float, optional
            Stream power exponent, power on slope
        uplift_rate : float, optional
            changes in topographic elevation due to tectonic uplift
        entrainment_threshold : float, optional
            threshold for sediment movement

        """

        self._grid = grid

        self.K = K_sp
        self.m = m_sp
        self.n = n_sp

        self.I = self._grid.zeros(centering='node')
        self.uplift_rate = uplift_rate
        self.entraiment_threshold = entraiment_threshold

        self.dzdt = self._grid.zeros(centering='node')


    def erode(self, dt, discharge_cms = 'water__discharge',
              slope='topographic__slope',):

        """
        For one time step, this erodes into the grid topography using
        the water discharge and topographic slope.

        The grid field 'topographic__elevation' is altered each time step.

        Inputs
        ------
        dt : time step

        discharge_cms : discharge on the nodes, if from the de Almeida solution
                        have units of cubic meters per second

        slope : topographic slope on each node.

        """
        try:
            S = self._grid.at_node[slope]
        except FieldError:
            print('Slope field was incorrectly passed to the component! Aborting')
        if type(discharge_cms) is str:
            Q = self._grid.at_node[discharge_cms]
        else:
            Q = discharge_cms

        Q_to_m = np.power(Q, self.m)

        S_to_n = np.power(S, self.n)

        self.I = (self.K * (Q_to_m * S_to_n - self.entraiment_threshold))

        self.dzdt = (self.uplift_rate - self.I)

        self._grid['node']['topographic__elevation'] += self.dzdt
