import graphviz

# create the directed graph
dot = graphviz.Digraph()

# add nodes to the graph
dot.node('A', 'Generate 3D mesh of terrain with Blender')
dot.node('B', 'Render front-view RGB images and height-map data')
dot.node('C', 'Train pix2pix model with rendered data')
dot.node('D', 'Validate model and optimize generation procedure')
dot.node('E', 'Generate more complicated data with Blender using optimized algorithm')
dot.node('F', 'Train more powerful model with new data')
dot.node('G', 'Repeat process until best model is achieved')

# add edges to the graph
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')

# render the graph
dot.render('terrain_generation_graph', view=True)

