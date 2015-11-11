#This is a model that creates an idealized glacial upland
#The upland has a moraine along one boundary, a flat upland
#and an incising channel along another boundary
#the landscape erodes by stream power with water routed over 
#topography, but not routed over flats/pits. 


from landlab.components.flow_routing.route_flow_dn import FlowRouter
from landlab.components.stream_power.fastscape_stream_power import SPEroder
from landlab.components.sink_fill.pit_remove import PitRemove
from landlab.components.sink_fill.fill_sinks import SinkFiller
from landlab.components.diffusion.diffusion import LinearDiffuser
from landlab import ModelParameterDictionary
#from landlab.plot import channel_profile as prf
from landlab.plot.imshow import imshow_node_grid
from landlab.io.esri_ascii import write_esri_ascii
from landlab import RasterModelGrid

import numpy as np
import pylab
import os

import pdb

#get the needed properties to build the grid:
input_file = './coupled_params.txt'
inputs = ModelParameterDictionary(input_file)
nrows = inputs.read_int('nrows')
ncols = inputs.read_int('ncols')
dx = inputs.read_float('dx')
#initial_slope = inputs.read_float('initial_slope')
rightmost_elevation = inputs.read_float('rightmost_elevation')
uplift_rate = inputs.read_float('uplift_rate')
runtime = inputs.read_float('total_time')
dt = inputs.read_float('dt')
nt = int(runtime//dt)
k_sp = inputs.read_float('K_sp')
uplift_per_step = uplift_rate * dt
moraine_height = inputs.read_float('moraine_height')
moraine_width = inputs.read_float('moraine_width')
#valley_width = inputs.read_float('valley_width')
#valley_depth = inputs.read_float('valley_depth')
num_outs = inputs.read_int('number_of_outputs')
plot_interval = int(nt//num_outs)

#instantiate the grid object
mg = RasterModelGrid(nrows, ncols, dx)

##create the elevation field in the grid:
#create the field
#specifically, this field has a triangular ramp
#moraine at the north edge of the domain.
mg.create_node_array_zeros('topographic__elevation')
z = mg.create_node_array_zeros()
moraine_start_y = np.max(mg.node_y)-moraine_width
moraine_ys = np.where(mg.node_y>moraine_start_y)
z[moraine_ys]+=(mg.node_y[moraine_ys]-np.min(mg.node_y[moraine_ys]))*(moraine_height/moraine_width)

#set valley
#valley_start_x = np.min(mg.node_x)+valley_width
#valley_ys = np.where((mg.node_x<valley_start_x)&(mg.node_y<moraine_start_y-valley_width))
#z[valley_ys] -= (np.max(mg.node_x[valley_ys])-mg.node_x[valley_ys])*(valley_depth/valley_width)

#set ramp (towards valley)
z[np.where(mg.node_y<moraine_start_y)] -= \
        (np.max(mg.node_x[np.where(mg.node_y<moraine_start_y)])-\
        mg.node_x[np.where(mg.node_y<moraine_start_y)])*(rightmost_elevation/(ncols*dx))
z += rightmost_elevation

#put these values plus roughness into that field
z += np.random.rand(len(z))/1

mg.at_node['topographic__elevation'] = z


#set up grid's boundary conditions (bottom, left, top, right) is inactive
mg.set_closed_boundaries_at_grid_edges(True, False, True, True)
bdy_moraine_ids = np.where((mg.node_y > moraine_start_y) & (mg.node_x == 0))
mg.status_at_node[bdy_moraine_ids]=4
mg.update_links_nodes_cells_to_new_BCs()

# Display a message
print 'Running ...' 

#instantiate the components:
fr = FlowRouter(mg)
sf = SinkFiller(mg)
pr = PitRemove(mg)
sp = SPEroder(mg, input_file)
#diffuse = PerronNLDiffuse(mg, input_file)
lin_diffuse = LinearDiffuser(grid=mg, input_stream=input_file)


#instantiate plot setting
pylab.close('all')
plot_time = plot_interval
plot_num = 0

#folder name
savepath = 'Not all dry_dt=' + str(dt) + '_total_time=' + str(runtime) + '_k_sp=' + \
            str(k_sp) + '_uplift_rate=' + str(uplift_rate) + '(exist_ramp_rightmost=' + str(rightmost_elevation) + ')'
if not os.path.isdir(savepath):
    os.makedirs(savepath)

#perform the loops:
for i in xrange(nt):
    #note the input arguments here are not totally standardized between modules
    #mg = diffuse.diffuse(mg, i*dt)
    #pdb.set_trace()
    mg = lin_diffuse.diffuse(dt)
    #mg = sf.fill_pits()
    #mg = pr.pit_fill()
    mg = fr.route_flow(routing_flat=False)    
    mg = sp.erode(mg, dt)
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step

    if i+1 == plot_time:
        print 'Plotting...'

        plot_num += 1
        pylab.figure(plot_num)
        im = imshow_node_grid(mg, 'topographic__elevation', cmap = 'gist_earth')    
        pylab.savefig(savepath + '/Topography_dt=' + str(dt) + '_t=' + str((i+1)*dt) + 
                '_k_sp=' + str(k_sp) + '_uplift_rate=' + str(uplift_rate) + '.jpg')
        write_esri_ascii(savepath + '/Topography_dt=' + str(dt) + '_t=' + str((i+1)*dt) + 
                '_k_sp=' + str(k_sp) + '_uplift_rate=' + str(uplift_rate) + '.txt', mg, 'topographic__elevation')

        plot_num += 1
        pylab.figure(plot_num)
        im = imshow_node_grid(mg, 'flow_sinks', cmap = 'Blues')
        pylab.savefig(savepath + '/sink_dt=' + str(dt) + '_t=' + str((i+1)*dt) + 
                '_k_sp=' + str(k_sp) + '_uplift_rate=' + str(uplift_rate) + '.jpg')

        """
        plot_num += 1
        pylab.figure(plot_num)
        im = imshow_node_grid(mg, 'drainage_area', cmap = 'Blues')
        pylab.savefig(savepath + '/Drainage_Area_dt=' + str(dt) + '_t=' + str((i+1)*dt) + 
                '_k_sp=' + str(k_sp) + '_uplift_rate=' + str(uplift_rate) + '.jpg')
        
        plot_num += 1
        pylab.figure(plot_num)
        im = imshow_node_grid(mg, 'topographic__steepest_slope', cmap = 'Greens')
        pylab.savefig(savepath + '/Slope_dt=' + str(dt) + '_t=' + str((i+1)*dt) + 
                '_k_sp=' + str(k_sp) + '_uplift_rate=' + str(uplift_rate) + '.jpg')
        """
        
        
        plot_time += plot_interval
        pylab.close('all')

    print 'Completed loop ', i+1
 
#print 'Completed the simulation. Plotting...'

print('Done.')

pylab.close('all')
