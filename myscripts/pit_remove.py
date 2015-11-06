"""
pit_remove.py
Implementation of algorithm described by Planchon(2001)

Created by JL, Oct 2015
"""

import numpy

import landlab
from landlab import Component, FieldError


class PitRemove(Component):

	def __init__(self, input_grid):

		self._grid = input_grid

		(self._nrows, self._ncols) = self._grid.shape
		self._R0 = numpy.array([0, self._nrows-1, 0, self._nrows-1, 0, self._nrows-1, 0, self._nrows-1])
		self._C0 = numpy.array([0, self._ncols-1, self._ncols-1, 0, self._ncols-1, 0, 0, self._ncols-1])
		self._dR = numpy.array([0, 0, 1, -1, 0, 0, 1, -1])
		self._dC = numpy.array([1, -1, 0, 0, -1, 1, 0, 0])
		self._fR = numpy.array([1, -1, -self._nrows+1, self._nrows-1, 1, -1, -self._nrows+1, self._nrows-1])
		self._fC = numpy.array([-self._ncols+1, self._ncols-1, -1, 1, self._ncols-1, -self._ncols+1, 1, -1])

		self._neighbor_dR = numpy.array([0, 0, 1, -1, 1, 1, -1, -1])
		self._neighbor_dC = numpy.array([1, -1, 0, 0, 1, -1, 1, -1])
		#self._visited = numpy.array([False]*self._grid.number_of_nodes)
		self._huge_number = 32767


	def _get_node_id(self, r, c):

		return r*self._ncols+c


	def _get_node_coord(self, nodeid):

		r = self._grid.node_y[nodeid]/self._grid.dx
		c = self._grid.node_x[nodeid]/self._grid.dx

		return r, c


	def _next_node(self, node, i):
		#For the given node and scan direction i, this function calculates the ID of the next node to consider

		(r, c) = self._get_node_coord(node)

		r = r+self._dR[i]
		c = c+self._dC[i]

		if r<0 or c<0 or r>=self._nrows or c>=self._ncols:
			r = r+self._fR[i]
			c = c+self._fC[i]
			if r<0 or c<0 or r>=self._nrows or c>=self._ncols:
				return -1

		return self._get_node_id(r, c)


	def _do_upstream_node(self, node, depth=1):
		#go upstream

		if depth>5000:
			return 

		(r, c) = self._get_node_coord(node)
		for i in range(8):
			neighbor_r = r+self._neighbor_dR[i]
			neighbor_c = c+self._neighbor_dC[i]
			if neighbor_r<0 or neighbor_c<0 or neighbor_r>=self._nrows or neighbor_c>=self._ncols:
				continue
			neighbor_node = self._get_node_id(neighbor_r, neighbor_c)
			if self._w[neighbor_node]!=self._huge_number:
				continue
			if self._z[neighbor_node]>=self._w[node]:
				self._w[neighbor_node]=self._z[neighbor_node]
				self._do_upstream_node(neighbor_node, depth+1)

		return 


	def pit_fill(self):
		#fill pit

		#initialisation
		self._z = self._grid['node']['topographic__elevation']
		self._w = numpy.array([0.0]*self._grid.number_of_nodes)
		n = self._grid.number_of_nodes
		(border, ) = numpy.where(numpy.logical_or(self._grid.status_at_node==1, self._grid.status_at_node==2))
		for i in range(n):
			if i in border:
				self._w[i] = self._z[i]
			else:
				self._w[i] = self._huge_number

		#explore from the border
		for i in border:
			self._do_upstream_node(i)

		#iteratively scan the DEM
		victory = False
		while not(victory):
			for scan in range(8):
				r = self._R0[scan]
				c = self._C0[scan]
				node = self._get_node_id(r, c)
				something_done = False
				while node!=-1:
					if self._w[node]>self._z[node]:
						for i in range(8):
							neighbor_r = r+self._neighbor_dR[i]
							neighbor_c = c+self._neighbor_dC[i]
							if neighbor_r<0 or neighbor_c<0 or neighbor_r>=self._nrows or neighbor_c>=self._ncols:
								continue
							neighbor_node = self._get_node_id(neighbor_r, neighbor_c)
							if self._z[node]>=self._w[neighbor_node]:
								self._w[node] = self._z[node]
								something_done = True
								self._do_upstream_node(node)
								break
							if self._w[node]>self._w[neighbor_node]:
								self._w[node] = self._w[neighbor_node]
								something_done = True
					node = self._next_node(node, scan)
					(r, c) = self._get_node_coord(node)
				if something_done == False:
					victory = True
					break

		self._grid['node']['topographic__elevation_modified'] = self._w

		return self._grid

