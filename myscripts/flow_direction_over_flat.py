"""
flow_direction_over_flat.py
Implementation of Barnes et al.(2001)

Created by JL, Oct 2015
"""

import numpy as np

import Queue

import landlab
from landlab import Component, FieldError

import pdb

class FlowRouterOverFlat(Component):

	def __init__(self, input_grid):

		self._grid = input_grid
		
		self._n = self._grid.number_of_nodes
		self._nrows, self._ncols = self._grid.shape
		(self._boundary, ) = np.where(self._grid.status_at_node!=0)
		(self._open_boundary, ) = np.where(np.logical_or(self._grid.status_at_node==1, self._grid.status_at_node==2))
		(self._close_boundary, ) = np.where(self._grid.status_at_node==4)

		self._neighbor_dR = np.array([0, 0, 1, -1, 1, 1, -1, -1])
		self._neighbor_dC = np.array([1, -1, 0, 0, 1, -1, 1, -1])


	def _get_node_id(self, r, c):

		return r*self._ncols+c


	def _get_node_coord(self, nodeid):

		r = self._grid.node_y[nodeid]/self._grid.dx
		c = self._grid.node_x[nodeid]/self._grid.dx

		return r, c


	def route_flow(self, dem='topographic__elevation_modified'):
		#main

		self._dem = self._grid['node'][dem]
		self._flow_receiver = self._flow_dirs_d8(self._dem)
		#pdb.set_trace()
		flat_mask, labels = self.resolve_flats()

		self._flow_dirs_over_flat_d8(flat_mask, labels)

		self._grid['node']['flow_receiver'] = self._flow_receiver

		return self._grid


	def resolve_flats(self):

		flat_mask = np.zeros(self._n, dtype='float')
		labels = np.zeros(self._n, dtype='int')
		
		high_edges, low_edges = self._flat_edges()
		
		if low_edges.empty():
			if not(high_edges.empty()):
				print 'There are undrainable flats!'
			else:
				print 'There are no flats!'
			return

		k = 1
		for c in low_edges.queue:
			if labels[c]==0:
				labels = self._label_flats(labels, c, k)
				k = k+1

		"""
		#It's hard ...find another way...
		temp_queue = high_edges
		while not(temp_queue.empty()):
			c = temp_queue.get()
			if labels[c]==0 remove c from high_edges
		"""
		flat_height = np.zeros(k, dtype='float')

		flat_mask, flat_height = self._away_from_higher(flat_mask, labels, flat_height, high_edges)
		flat_mask, flat_height = self._towards_lower(flat_mask, labels, flat_height, low_edges)

		return flat_mask, labels


	def _flat_edges(self):

		low_edges = Queue.Queue()
		high_edges = Queue.Queue()
		
		for node in range(self._n):
			if node in self._boundary:
				continue
			r, c = self._get_node_coord(node)
			for i in range(8):
				neighbor_r = r+self._neighbor_dR[i]
				neighbor_c = c+self._neighbor_dC[i]
				if neighbor_r<0 or neighbor_c<0 or neighbor_r>=self._nrows or neighbor_c>=self._ncols:
					continue
				neighbor_node = self._get_node_id(neighbor_r, neighbor_c)
				if neighbor_node in self._boundary:
					if self._flow_receiver[node]==node and (neighbor_node in self._close_boundary):
						high_edges.put(node)
						break
					continue	
				if self._flow_receiver[node]!=node and self._flow_receiver[neighbor_node]==neighbor_node and self._dem[node]==self._dem[neighbor_node]:
					low_edges.put(node)
					break
				elif self._flow_receiver[node]==node and self._dem[node]<self._dem[neighbor_node]:
					high_edges.put(node)
					break

		return high_edges, low_edges


	def _label_flats(self, labels, node, labelid):

		to_fill = Queue.Queue()

		to_fill.put(node)
		elev = self._dem[node]
		while not(to_fill.empty()):
			node = to_fill.get()
			if self._dem[node]!=elev:
				continue
			if labels[node]!=0:
				continue
			labels[node] = labelid

			r, c = self._get_node_coord(node)
			for i in range(8):
				neighbor_r = r+self._neighbor_dR[i]
				neighbor_c = c+self._neighbor_dC[i]
				if neighbor_r<0 or neighbor_c<0 or neighbor_r>=self._nrows or neighbor_c>=self._ncols:
					continue
				neighbor_node = self._get_node_id(neighbor_r, neighbor_c)
				if neighbor_node in self._boundary:
					continue
				to_fill.put(neighbor_node)

		return labels


	def _away_from_higher(self, flat_mask, labels, flat_height, high_edges):

		k = 1
		MARKER = -100
		high_edges.put(MARKER)

		while high_edges.qsize()>1:
			node = high_edges.get()

			if node==MARKER:
				k += 1
				high_edges.put(MARKER)
				continue
			if flat_mask[node]>0:
				continue
			
			flat_mask[node] = k
			flat_height[labels[node]] = k

			r, c = self._get_node_coord(node)
			for i in range(8):
				neighbor_r = r+self._neighbor_dR[i]
				neighbor_c = c+self._neighbor_dC[i]
				if neighbor_r<0 or neighbor_c<0 or neighbor_r>=self._nrows or neighbor_c>=self._ncols:
					continue
				neighbor_node = self._get_node_id(neighbor_r, neighbor_c)
				if neighbor_node in self._boundary:
					continue
				if labels[neighbor_node]==labels[node] and self._flow_receiver[neighbor_node]==neighbor_node:
					high_edges.put(neighbor_node)

		return flat_mask, flat_height


	def _towards_lower(self, flat_mask, labels, flat_height, low_edges):

		#pdb.set_trace()
		flat_mask = 0-flat_mask
		k = 1
		MARKER = -100
		low_edges.put(MARKER)

		while low_edges.qsize()>1:
			node = low_edges.get()

			if node==MARKER:
				k += 1
				low_edges.put(MARKER)
				continue
			if flat_mask[node]>0:
				continue

			if flat_mask[node]<0:
				flat_mask[node] = flat_height[labels[node]]+flat_mask[node]+2*k
			else:
				flat_mask[node] = 2*k

			r, c = self._get_node_coord(node)
			for i in range(8):
				neighbor_r = r+self._neighbor_dR[i]
				neighbor_c = c+self._neighbor_dC[i]
				if neighbor_r<0 or neighbor_c<0 or neighbor_r>=self._nrows or neighbor_c>=self._ncols:
					continue
				neighbor_node = self._get_node_id(neighbor_r, neighbor_c)
				if neighbor_node in self._boundary:
					continue
				if labels[neighbor_node]==labels[node] and self._flow_receiver[neighbor_node]==neighbor_node:
					low_edges.put(neighbor_node)

		return flat_mask, flat_height


	def _flow_dirs_d8(self, dem):

		flow_receiver = np.arange(self._n)

		for node in range(self._n):
			if node in self._boundary:
				continue
			min_elev = dem[node]
			receiver = node
			r, c = self._get_node_coord(node)
			for i in range(8):
				neighbor_r = r+self._neighbor_dR[i]
				neighbor_c = c+self._neighbor_dC[i]
				if neighbor_r<0 or neighbor_c<0 or neighbor_r>=self._nrows or neighbor_c>=self._ncols:
					continue
				neighbor_node = self._get_node_id(neighbor_r, neighbor_c)
				if neighbor_node in self._open_boundary:
					min_elev = dem[neighbor_node]
					receiver = neighbor_node
					break
				if dem[neighbor_node]<min_elev:
					min_elev = dem[neighbor_node]
					receiver = neighbor_node
			flow_receiver[node] = receiver

		return flow_receiver


	def _flow_dirs_over_flat_d8(self, flat_mask, labels):

		for node in range(self._n):
			if self._flow_receiver[node]!=node:
				continue
			min_elev = flat_mask[node]
			receiver = node
			r, c = self._get_node_coord(node)
			for i in range(8):
				neighbor_r = r+self._neighbor_dR[i]
				neighbor_c = c+self._neighbor_dC[i]
				if neighbor_r<0 or neighbor_c<0 or neighbor_r>=self._nrows or neighbor_c>=self._ncols:
					continue
				neighbor_node = self._get_node_id(neighbor_r, neighbor_c)
				if labels[neighbor_node]!=labels[node]:
					continue
				if flat_mask[neighbor_node]<min_elev:
					min_elev = flat_mask[neighbor_node]
					receiver = neighbor_node
			self._flow_receiver[node] = receiver


