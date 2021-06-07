import bmesh

def face_split(me, face_index, edge_indices=[1,3], fac = 0.5):
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    f = bm.faces[face_index]
    e1 = f.edges[edge_indices[0]]
    e2 = f.edges[edge_indices[1]]
    ne1, nv1 = bmesh.utils.edge_split(e1,e1.verts[0],fac)
    ne2, nv2 = bmesh.utils.edge_split(e2,e2.verts[0],fac)
    nf = bmesh.utils.face_split(f,nv1,nv2)
    bm.faces.index_update()
    bm.edges.index_update()
    bm.verts.index_update()
    bm.to_mesh(me)
    bm.free()