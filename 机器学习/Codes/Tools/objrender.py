import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt

fuze_trimesh = trimesh.load('0.obj')
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()
scene.add(mesh)

# pl = pyrender.PointLight(color=[1.0, 1.0, 0.0], intensity=2.0, show_world_axis=true)
# scene.add(pl)
# ic = pyrender.IntrinsicsCamera(fx=0, fy=0, cx=1, cy=1, znear=0.05, zfar=100.0, name=None)
# oc = pyrender.OrthographicCamera(xmag=2, ymag=2, znear = 0.08, zfar = 100.0)
pc = pyrender.PerspectiveCamera(yfov=np.pi / 3, znear = 0.15, zfar = 1700.0, aspectRatio=1.414)

scene.add(pc)
print(pc.get_projection_matrix())
#
# scene.add(oc)
# pyrender.Viewer(scene, use_raymond_lighting=True, show_world_axis=True)
r = pyrender.OffscreenRenderer(200, 200)
color, depth = r.render(scene)
plt.imshow(color)
plt.show()
