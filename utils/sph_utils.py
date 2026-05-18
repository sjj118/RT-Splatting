import numpy as np
import torch

epsilon = 1e-10


def _convert_sph_conventions(pts_r_angle1_angle2):
    pts_r_theta_phi = torch.zeros(pts_r_angle1_angle2.shape, device=pts_r_angle1_angle2.device)

    # Radius is the same
    pts_r_theta_phi[:, 0] = pts_r_angle1_angle2[:, 0]
    # Angle 1
    pts_r_theta_phi[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]

    # Angle 2
    ind = pts_r_angle1_angle2[:, 2] < 0
    pts_r_theta_phi[ind, 2] = 2 * np.pi + pts_r_angle1_angle2[ind, 2]
    pts_r_theta_phi[torch.logical_not(ind), 2] = pts_r_angle1_angle2[torch.logical_not(ind), 2]

    return pts_r_theta_phi


def cart2sph(pts_cart):
    # Compute r
    r = torch.ones((len(pts_cart))).cuda()

    # Compute latitude
    z = pts_cart[:, 2]
    lat = torch.arcsin(torch.clamp(z, -1 + 1e-7, 1 - 1e-7))

    # Compute longitude
    x = pts_cart[:, 0]
    y = pts_cart[:, 1]

    near_zeros = torch.abs(x) < epsilon
    sign = torch.sign(x)
    x = x * (near_zeros.logical_not())
    x = x + (near_zeros * epsilon) * sign
    lng = torch.arctan2(y, x)

    # Assemble
    pts_r_lat_lng = torch.cat([r[:, None], lat[:, None], lng[:, None]], dim=-1)

    pts_sph = _convert_sph_conventions(pts_r_lat_lng)
    return pts_sph


def _warn_degree(angles):
    if (np.abs(angles) > 2 * np.pi).any():
        print('Some input value falls outside [-2pi, 2pi]. You sure inputs are in radians')


def sph2cart(pts_sph, convention='lat-lng'):
    """Inverse of :func:`cart2sph`.

    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError('Shape of input must be either (3,) or (n, 3)')

    # Degrees?
    _warn_degree(pts_sph[:, 1:])

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    elif convention == 'theta-phi':
        pts_r_lat_lng = _convert_sph_conventions(pts_sph, 'theta-phi_to_lat-lng')
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart


def _convert_sph_conventions2(pts_r_angle1_angle2, what2what):
    """Internal function converting between different conventions for
    spherical coordinates. See :func:`cart2sph` for conventions.
    """
    # print(pts_r_angle1_angle2[:, 2])

    if what2what == 'lat-lng_to_theta-phi':
        pts_r_theta_phi = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_theta_phi[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_theta_phi[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] < 0
        pts_r_theta_phi[ind, 2] = 2 * np.pi + pts_r_angle1_angle2[ind, 2]
        pts_r_theta_phi[np.logical_not(ind), 2] = pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_theta_phi

    if what2what == 'theta-phi_to_lat-lng':
        pts_r_lat_lng = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_lat_lng[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_lat_lng[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] > np.pi
        # ind = pts_r_angle1_angle2[:, 2] > 0
        pts_r_lat_lng[ind, 2] = pts_r_angle1_angle2[ind, 2] - 2 * np.pi
        pts_r_lat_lng[np.logical_not(ind), 2] = pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_lat_lng

    raise NotImplementedError(what2what)


def uniform_sample_sph(n, r=1, convention='lat-lng'):
    r"""Uniformly samples points on the sphere
    [`source <https://mathworld.wolfram.com/SpherePointPicking.html>`_].

    Args:
        n (int): Total number of points to sample. Must be a square number.
        r (float, optional): Radius of the sphere. Defaults to :math:`1`.
        convention (str, optional): Convention for spherical coordinates.
            See :func:`cart2sph` for conventions.

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians. The points are ordered such that all azimuths are looped
        through first at each elevation.
    """
    pts_r_theta_phi = []
    for u in np.linspace(0, 1, n):
        for v in np.linspace(0, 1, n * 2):
            # theta = np.arccos(2 * u - 1) # [0, pi]
            theta = np.pi * u  # [0,  pi]
            phi = 2 * np.pi * v  # [0, 2pi]
            pts_r_theta_phi.append((r, theta, phi))

    pts_r_theta_phi = np.vstack(pts_r_theta_phi)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = _convert_sph_conventions2(pts_r_theta_phi, 'theta-phi_to_lat-lng')

    elif convention == 'theta-phi':
        pts_sph = pts_r_theta_phi
    else:
        raise NotImplementedError(convention)

    return pts_sph


def cart2sph2(pts_cart, convention='lat-lng'):

    pts_cart = np.array(pts_cart)

    # Validate inputs
    is_one_point = False
    if pts_cart.shape == (3,):
        is_one_point = True
        pts_cart = pts_cart.reshape(1, 3)
    elif pts_cart.ndim != 2 or pts_cart.shape[1] != 3:
        raise ValueError('Shape of input must be either (3,) or (n, 3)')

    # Compute r
    r = np.sqrt(np.sum(np.square(pts_cart), axis=1))

    # Compute latitude
    z = pts_cart[:, 2]
    lat = np.arcsin(z / r)

    # Compute longitude
    x = pts_cart[:, 0]
    y = pts_cart[:, 1]
    lng = np.arctan2(y, x)  # choosing the quadrant correctly

    # Assemble
    pts_r_lat_lng = np.stack((r, lat, lng), axis=-1)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = pts_r_lat_lng
    elif convention == 'theta-phi':
        pts_sph = _convert_sph_conventions2(pts_r_lat_lng, 'lat-lng_to_theta-phi')
    else:
        raise NotImplementedError(convention)

    if is_one_point:
        pts_sph = pts_sph.reshape(3)

    return pts_sph
