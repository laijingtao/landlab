"""
flow_direction_over_flat.py
Implementation of Barnes et al.(2014)

Created by JL, Oct 2015
"""

import numpy as np

import Queue

import landlab
from landlab import Component, FieldError
from landlab.grid.base import BAD_INDEX_VALUE

from landlab.components.flow_routing.flow_direction_DN import grid_flow_directions

import pdb

class FlowRouterOverFlat(Component):

	def __init__(self, input_grid):

		self._grid = input_grid

		self._n = self._grid.number_of_nodes
		(self._boundary, ) = np.where(self._grid.status_at_node!=0)
		(self._open_boundary, ) = np.where(np.logical_or(self._grid.status_at_node==1, self._grid.status_at_node==2))
		(self._close_boundary, ) = np.where(self._grid.status_at_node==4)

		self._neighbors = np.concatenate((self._grid.neighbors_at_node, self._grid.diagonals_at_node), axis=1)
		self._neighbors[self._neighbors == BAD_INDEX_VALUE] = -1


	def route_flow(self, receiver, dem='topographic__elevation'):
		#main
		self._dem = self._grid['node'][dem]
		"""
		if receiver==None:
			self._flow_receiver = self._flow_dirs_d8(self._dem)
		else:
			self._flow_receiver = receiver
		"""
		self._flow_receiver = receiver
		#(self._flow_receiver, ss) = grid_flow_directions(self._grid, self._dem)

		flat_mask, labels = self._resolve_flats()
		#pdb.set_trace()
		self._flow_dirs_over_flat_d8(flat_mask, labels)


		#a, q, s = flow_accum_bw.flow_accumulation(self._flow_receiver, self._open_boundary, node_cell_area=self._grid.forced_cell_areas)

		#self._grid['node']['flow_receiver'] = self._flow_receiver

		return self._flow_receiver


	def _resolve_flats(self):

		flat_mask = np.zeros(self._n, dtype='float')
		labels = np.zeros(self._n, dtype='int')

		high_edges, low_edges = self._flat_edges()

		"""
		if low_edges.empty():
			if not(high_edges.empty()):
				print 'There are undrainable flats!'
			else:
				print 'There are no flats!'
			return
		"""

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

		#this part is bottleneck
		flat_mask, flat_height = self._away_from_higher(flat_mask, labels, flat_height, high_edges)
		flat_mask, flat_height = self._towards_lower(flat_mask, labels, flat_height, low_edges)

		return flat_mask, labels


	def _flat_edges(self):

		low_edges = Queue.Queue(maxsize=self._n*2)
		high_edges = Queue.Queue(maxsize=self._n*2)
		high_put = high_edges.put
		low_put = low_edges.put
		for node in range(self._n):
			if node in self._boundary:
				continue
			for neighbor_node in self._neighbors[node]:
				if neighbor_node==-1:
					continue
				if neighbor_node in self._boundary:
					if self._flow_receiver[node]==node and (neighbor_node in self._close_boundary):
						high_put(node)
						break
					continue
				if self._flow_receiver[node]!=node and self._flow_receiver[neighbor_node]==neighbor_node and self._dem[node]==self._dem[neighbor_node]:
					low_put(node)
					break
				elif self._flow_receiver[node]==node and self._dem[node]<self._dem[neighbor_node]:
					high_put(node)
					break

		return high_edges, low_edges


	def _label_flats(self, labels, node, labelid):

		to_fill = Queue.Queue(maxsize=self._n*2)
		to_fill_put = to_fill.put
		to_fill_put(node)
		elev = self._dem[node]
		while not(to_fill.empty()):
			node = to_fill.get()
			if labels[node]!=0:
				continue
			labels[node] = labelid
			for neighbor_node in self._neighbors[node]:
				if neighbor_node==-1:
					continue
				if neighbor_node in self._boundary:
					continue
				if labels[neighbor_node]!=0:
					continue
				if self._dem[neighbor_node]!=elev:
					continue
				to_fill_put(neighbor_node)

		return labels


	def _away_from_higher(self, flat_mask, labels, flat_height, high_edges):

		k = 1
		MARKER = -100
		high_put = high_edges.put
		high_get = high_edges.get
		high_qsize = high_edges.qsize
		high_put(MARKER)

		while high_qsize()>1:
			node = high_get()

			if node==MARKER:
				k += 1
				high_put(MARKER)
				continue
			if flat_mask[node]>0:
				continue

			flat_mask[node] = k
			flat_height[labels[node]] = k

			for neighbor_node in self._neighbors[node]:
				if neighbor_node==-1:
					continue
				if neighbor_node in self._boundary:
					continue
				if flat_mask[neighbor_node]>0:
					continue
				if neighbor_node in high_edges.queue:
					continue
				if labels[neighbor_node]==labels[node] and self._flow_receiver[neighbor_node]==neighbor_node:
					high_put(neighbor_node)

		return flat_mask, flat_height


	def _towards_lower(self, flat_mask, labels, flat_height, low_edges):

		flat_mask = 0-flat_mask
		k = 1
		MARKER = -100
		low_put = low_edges.put
		low_get = low_edges.get
		low_qsize = low_edges.qsize
		low_put(MARKER)

		while low_qsize()>1:
			node = low_get()

			if node==MARKER:
				k += 1
				low_put(MARKER)
				continue
			if flat_mask[node]>0:
				continue

			if flat_mask[node]<0:
				flat_mask[node] = flat_height[labels[node]]+flat_mask[node]+2*k
			else:
				flat_mask[node] = 2*k

			for neighbor_node in self._neighbors[node]:
				if neighbor_node==-1:
					continue
				if neighbor_node in self._boundary:
					continue
				if flat_mask[neighbor_node]>0:
					continue
				if neighbor_node in low_edges.queue:
					continue
				if labels[neighbor_node]==labels[node] and self._flow_receiver[neighbor_node]==neighbor_node:
					low_put(neighbor_node)

		return flat_mask, flat_height


	def _flow_dirs_d8(self, dem):

		flow_receiver = np.arange(self._n)

		for node in range(self._n):
			if node in self._boundary:
				continue
			min_elev = dem[node]
			receiver = node
			for neighbor_node in self._neighbors[node]:
				if neighbor_node==-1:
					continue
				if neighbor_node in self._open_boundary:
					receiver = neighbor_node
					break
				if neighbor_node in self._close_boundary:
					continue
				if dem[neighbor_node]<min_elev:
					min_elev = dem[neighbor_node]
					receiver = neighbor_node
			flow_receiver[node] = receiver

		return flow_receiver


	def _flow_dirs_over_flat_d8(self, flat_mask, labels):

		for node in range(self._n):
			if node in self._boundary:
				continue
			if self._flow_receiver[node]!=node:
				continue
			"""
			min_elev = flat_mask[node]
			receiver = node
			for i in range(8):
				neighbor_node = self._neighbors[node][i]
				if neighbor_node==-1:
					continue
				if labels[neighbor_node]!=labels[node]:
					continue
				if flat_mask[neighbor_node]<min_elev:
					min_elev = flat_mask[neighbor_node]
					receiver = neighbor_node
			"""
			potential_receiver = self._neighbors[node]
			potential_receiver = potential_receiver[np.where(potential_receiver!=-1)]
			potential_receiver = potential_receiver[np.where(labels[potential_receiver]==labels[node])]
			receiver = potential_receiver[np.argmin(flat_mask[potential_receiver])]

			self._flow_receiver[node] = receiver
