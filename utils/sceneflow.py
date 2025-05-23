import torch

def get_4D_inversion(AtA, group_size, batch_size, npoint, flow_dim, EPS):
    assert(group_size + 1 == 4)
    AtA = AtA.reshape(batch_size, npoint, flow_dim // group_size, -1)
    AtA0, AtA1, AtA2, AtA3, \
    AtA4, AtA5, AtA6, AtA7, \
    AtA8, AtA9, AtA10, AtA11, \
    AtA12, AtA13, AtA14, AtA15  = torch.unbind(AtA, dim=3)

    inv0 = AtA5 * AtA10 * AtA15 - \
           AtA5 * AtA11 * AtA14 - \
           AtA9 * AtA6 * AtA15 + \
           AtA9 * AtA7 * AtA14 + \
           AtA13 * AtA6 * AtA11 - \
           AtA13 * AtA7 * AtA10
    inv4 = -AtA4 * AtA10 * AtA15 + \
           AtA4 * AtA11 * AtA14 + \
           AtA8 * AtA6 * AtA15 - \
           AtA8 * AtA7 * AtA14 - \
           AtA12 * AtA6 * AtA11 + \
           AtA12 * AtA7 * AtA10
    inv8 = AtA4 * AtA9 * AtA15 - \
           AtA4 * AtA11 * AtA13 - \
           AtA8 * AtA5 * AtA15 + \
           AtA8 * AtA7 * AtA13 + \
           AtA12 * AtA5 * AtA11 - \
           AtA12 * AtA7 * AtA9
    inv12 = -AtA4 * AtA9 * AtA14 + \
            AtA4 * AtA10 * AtA13 + \
            AtA8 * AtA5 * AtA14 - \
            AtA8 * AtA6 * AtA13 - \
            AtA12 * AtA5 * AtA10 + \
            AtA12 * AtA6 * AtA9
    inv1 = -AtA1 * AtA10 * AtA15 + \
           AtA1 * AtA11 * AtA14 + \
           AtA9 * AtA2 * AtA15 - \
           AtA9 * AtA3 * AtA14 - \
           AtA13 * AtA2 * AtA11 + \
           AtA13 * AtA3 * AtA10
    inv5 = AtA0 * AtA10 * AtA15 - \
           AtA0 * AtA11 * AtA14 - \
           AtA8 * AtA2 * AtA15 + \
           AtA8 * AtA3 * AtA14 + \
           AtA12 * AtA2 * AtA11 - \
           AtA12 * AtA3 * AtA10
    inv9 = -AtA0 * AtA9 * AtA15 + \
           AtA0 * AtA11 * AtA13 + \
           AtA8 * AtA1 * AtA15 - \
           AtA8 * AtA3 * AtA13 - \
           AtA12 * AtA1 * AtA11 + \
           AtA12 * AtA3 * AtA9
    inv13 = AtA0 * AtA9 * AtA14 - \
            AtA0 * AtA10 * AtA13 - \
            AtA8 * AtA1 * AtA14 + \
            AtA8 * AtA2 * AtA13 + \
            AtA12 * AtA1 * AtA10 - \
            AtA12 * AtA2 * AtA9
    inv2 = AtA1 * AtA6 * AtA15 - \
           AtA1 * AtA7 * AtA14 - \
           AtA5 * AtA2 * AtA15 + \
           AtA5 * AtA3 * AtA14 + \
           AtA13 * AtA2 * AtA7 - \
           AtA13 * AtA3 * AtA6
    inv6 = -AtA0 * AtA6 * AtA15 + \
           AtA0 * AtA7 * AtA14 + \
           AtA4 * AtA2 * AtA15 - \
           AtA4 * AtA3 * AtA14 - \
           AtA12 * AtA2 * AtA7 + \
           AtA12 * AtA3 * AtA6
    inv10 = AtA0 * AtA5 * AtA15 - \
            AtA0 * AtA7 * AtA13 - \
            AtA4 * AtA1 * AtA15 + \
            AtA4 * AtA3 * AtA13 + \
            AtA12 * AtA1 * AtA7 - \
            AtA12 * AtA3 * AtA5
    inv14 = -AtA0 * AtA5 * AtA14 + \
            AtA0 * AtA6 * AtA13 + \
            AtA4 * AtA1 * AtA14 - \
            AtA4 * AtA2 * AtA13 - \
            AtA12 * AtA1 * AtA6 + \
            AtA12 * AtA2 * AtA5
    inv3 = -AtA1 * AtA6 * AtA11 + \
           AtA1 * AtA7 * AtA10 + \
           AtA5 * AtA2 * AtA11 - \
           AtA5 * AtA3 * AtA10 - \
           AtA9 * AtA2 * AtA7 + \
           AtA9 * AtA3 * AtA6
    inv7 = AtA0 * AtA6 * AtA11 - \
           AtA0 * AtA7 * AtA10 - \
           AtA4 * AtA2 * AtA11 + \
           AtA4 * AtA3 * AtA10 + \
           AtA8 * AtA2 * AtA7 - \
           AtA8 * AtA3 * AtA6
    inv11 = -AtA0 * AtA5 * AtA11 + \
            AtA0 * AtA7 * AtA9 + \
            AtA4 * AtA1 * AtA11 - \
            AtA4 * AtA3 * AtA9 - \
            AtA8 * AtA1 * AtA7 + \
            AtA8 * AtA3 * AtA5
    inv15 = AtA0 * AtA5 * AtA10 - \
            AtA0 * AtA6 * AtA9 - \
            AtA4 * AtA1 * AtA10 + \
            AtA4 * AtA2 * AtA9 + \
            AtA8 * AtA1 * AtA6 - \
            AtA8 * AtA2 * AtA5
    D = AtA0 * inv0 + AtA1 * inv4 \
        + AtA2 * inv8 + AtA3 * inv12
    D = torch.where(torch.abs(D) < EPS, torch.full_like(D, EPS), D)
    D = torch.unsqueeze(D, -1)
    inv0 = torch.unsqueeze(inv0, -1)
    inv1 = torch.unsqueeze(inv1, -1)
    inv2 = torch.unsqueeze(inv2, -1)
    inv3 = torch.unsqueeze(inv3, -1)
    inv4 = torch.unsqueeze(inv4, -1)
    inv5 = torch.unsqueeze(inv5, -1)
    inv6 = torch.unsqueeze(inv6, -1)
    inv7 = torch.unsqueeze(inv7, -1)
    inv8 = torch.unsqueeze(inv8, -1)
    inv9 = torch.unsqueeze(inv9, -1)
    inv10 = torch.unsqueeze(inv10, -1)
    inv11 = torch.unsqueeze(inv11, -1)
    inv12 = torch.unsqueeze(inv12, -1)
    inv13 = torch.unsqueeze(inv13, -1)
    inv14 = torch.unsqueeze(inv14, -1)
    inv15 = torch.unsqueeze(inv15, -1)
    inv_AtA = torch.cat([inv0, inv1, inv2, inv3,
                         inv4, inv5, inv6, inv7,
                         inv8, inv9, inv10, inv11,
                         inv12, inv13, inv14, inv15], dim=-1)
    inv_AtA = inv_AtA / D
    return inv_AtA.reshape(batch_size, npoint, flow_dim // group_size, group_size + 1, group_size + 1)

def get_st_surface(GROUP_SIZE, batch_size, duplicate_grouped_time, flow_dim, new_points_, npoint, nsample, W=None):
    EPS = 0.000001

    new_points_ = torch.cat([
        new_points_, 
        torch.ones(batch_size, npoint, flow_dim // GROUP_SIZE, nsample, 1, dtype=new_points_.dtype, device=new_points_.device)
        ], dim=-1)

    if W is None:
        AtA = torch.matmul(new_points_.transpose(1, 2), new_points_)
        inv_AtA = get_4D_inversion(AtA, GROUP_SIZE, batch_size, npoint, flow_dim, EPS)
        n_new_points_ = torch.matmul(new_points_.transpose(1, 2), duplicate_grouped_time)
        n_new_points_ = torch.matmul(inv_AtA, n_new_points_)
    else:
        AtA = W * new_points_
        AtA = torch.matmul(new_points_.transpose(1, 2), AtA)
        inv_AtA = get_4D_inversion(AtA, GROUP_SIZE, batch_size, npoint, flow_dim, EPS)
        n_new_points_ = W * duplicate_grouped_time
        n_new_points_ = torch.matmul(new_points_.transpose(1, 2), n_new_points_)
        n_new_points_ = torch.matmul(inv_AtA, n_new_points_)
    return n_new_points_