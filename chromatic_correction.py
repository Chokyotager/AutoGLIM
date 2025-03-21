import numpy as np

polynomial_degree = 1

def make_polynomial (degree, x, y):
    
    A = list()
    for i in range(degree + 1):
        for j in range(degree - i + 1):
            A.append((x**i) * (y**j))
    return np.column_stack(A)

def calculate_shift (x_r, y_r, x_g, y_g, x_b, y_b):

    # Empirically observed shift
    delta_x_g_r = x_r - x_g
    delta_y_g_r = y_r - y_g

    A = make_polynomial(polynomial_degree, x_g, y_g)

    # QR decomposition
    Q, R = np.linalg.qr(A, mode="reduced")

    dgrx_coeff = np.linalg.solve(R, Q.T @ delta_x_g_r)
    dgry_coeff = np.linalg.solve(R, Q.T @ delta_y_g_r)

    # Repeat for b
    delta_x_b_r = x_r - x_b
    delta_y_b_r = y_r - y_b

    A = make_polynomial(1, x_b, y_b)
    Q, R = np.linalg.qr(A, mode="reduced")

    # dbrx_coeff calculation
    dbrx_coeff = np.linalg.solve(R, Q.T @ delta_x_b_r)
    dbry_coeff = np.linalg.solve(R, Q.T @ delta_y_b_r)

    return dgrx_coeff, dgry_coeff, dbrx_coeff, dbry_coeff

def correct_shift (x_g, y_g, x_b, y_b, dgrx_coeff, dgry_coeff, dbrx_coeff, dbry_coeff):

    A = make_polynomial(polynomial_degree, x_g, y_g)
    
    delta_x_g_r_res = A @ dgrx_coeff
    delta_y_g_r_res = A @ dgry_coeff
    
    x_g2r_res = delta_x_g_r_res + x_g
    y_g2r_res = delta_y_g_r_res + y_g

    A = make_polynomial(polynomial_degree, x_b, y_b)

    delta_x_b_r_res = A @ dbrx_coeff
    delta_y_b_r_res = A @ dbry_coeff

    x_b2r_res = delta_x_b_r_res + x_b
    y_b2r_res = delta_y_b_r_res + y_b

    return x_g2r_res, y_g2r_res, x_b2r_res, y_b2r_res

def calculate_error (x_r, y_r, x_g, y_g, x_b, y_b, dgrx_coeff, dgry_coeff, dbrx_coeff, dbry_coeff):

    A = make_polynomial(polynomial_degree, x_g, y_g)
    
    delta_x_g_r_res = A @ dgrx_coeff
    delta_y_g_r_res = A @ dgry_coeff
    
    x_g2r_res = delta_x_g_r_res + x_g
    y_g2r_res = delta_y_g_r_res + y_g

    A = make_polynomial(polynomial_degree, x_b, y_b)

    delta_x_b_r_res = A @ dbrx_coeff
    delta_y_b_r_res = A @ dbry_coeff

    x_b2r_res = delta_x_b_r_res + x_b
    y_b2r_res = delta_y_b_r_res + y_b

    # Calculate errors
    x_error_g2r = x_r - x_g2r_res
    y_error_g2r = y_r - y_g2r_res

    x_error_b2r = x_r - x_b2r_res
    y_error_b2r = y_r - y_b2r_res

    return x_error_g2r, y_error_g2r, x_error_b2r, y_error_b2r