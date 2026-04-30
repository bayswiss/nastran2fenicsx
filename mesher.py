import gmsh
import os
import math

gmsh.initialize()
# Set dimension units
gmsh.option.setString("Geometry.OCCTargetUnit", "M")

# Setting working directory
path = os.path.dirname(os.path.abspath(__file__))
step_file = os.path.join(path, "beam.step")

gmsh.model.occ.importShapes(step_file)
gmsh.model.occ.synchronize()


# Set the target mesh size globally
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.01)

gmsh.model.addPhysicalGroup(3, [1], 1, "structure")

gmsh.model.addPhysicalGroup(2, [5], 2, "clamp")
# gmsh.option.setNumber("Mesh.Algorithm", 5)
# gmsh.option.setNumber("Mesh.Algorithm3D", 8)
# Generate and view
gmsh.model.mesh.generate(3)
gmsh.fltk.run()
gmsh.write("beam.msh")
gmsh.write("beam.bdf")

gmsh.finalize()