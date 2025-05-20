using Polynomials
using FastGaussQuadrature
using LinearAlgebra
using Plots

# Define parameters
degree = 30  # degree of polynomial approximation
χ = 10        # bond dimension for STT
ϵ = 1e-10     # SVD truncation threshold

# Define 2D Gaussian PDF
function gaussian_pdf(x, y, Σ)
    det_Σ = det(Σ)
    inv_Σ = inv(Σ)
    
    exponent = -0.5 * [x, y]' * inv_Σ * [x, y]
    return (1 / (2π * sqrt(det_Σ))) * exp(exponent)
end

# Define domain as [-1,1] × [-1,1]
domain = (-1.0, 1.0)

# Generate Gaussian quadrature points and weights
# We use Gauss-Chebyshev quadrature since we're using Chebyshev polynomials
n_quad = degree + 1
x, wx = gausschebyshev(n_quad)
y, wy = gausschebyshev(n_quad)

# Define covariance matrix with correlation
ρ = 0.9 # correlation coefficient
σx = 1.0  # standard deviation in x
σy = 1.0  # standard deviation in y
Σ = [σx^2 ρ*σx*σy; ρ*σx*σy σy^2]

# Evaluate PDF at quadrature points to form matrix Q
Q = zeros(n_quad, n_quad)
for i in 1:n_quad
    for j in 1:n_quad
        Q[i, j] = gaussian_pdf(x[i], y[j], Σ)
    end
end

# Perform SVD decomposition of Q
U, S, V = svd(Q)

# Truncate SVD to bond dimension χ
if χ < length(S)
    S = S[1:χ]
    U = U[:, 1:χ]
    V = V[:, 1:χ]
end

# Construct L and R matrices
L = U * Diagonal(sqrt.(S))
R = Diagonal(sqrt.(S)) * V'

# Generate Chebyshev polynomial basis
# We use ChebyshevT basis (first kind)
chebyshevs = [Polynomials.ChebyshevT([zeros(k)..., 1.0]) for k in 0:degree]

# Calculate normalization constants for Chebyshev polynomials
# For ChebyshevT, N_0 = π and N_k = π/2 for k≥1
norm_consts = [k == 0 ? π : π/2 for k in 0:degree]

# Construct STT cores P1 and P2
P1 = zeros(1, χ, degree + 1)
P2 = zeros(χ, 1, degree + 1)

for n in 1:χ
    for i in 0:degree
        # Compute P1 coefficients with normalization
        P1[1, n, i+1] = sum(wx[k] * chebyshevs[i+1](x[k]) * L[k, n] / norm_consts[i+1] for k in 1:n_quad)
        
        # Compute P2 coefficients with normalization
        P2[n, 1, i+1] = sum(wy[k] * chebyshevs[i+1](y[k]) * R[n, k] / norm_consts[i+1] for k in 1:n_quad)
    end
end

# Function to evaluate the STT approximation
function evaluate_stt(x_val, y_val)
    # Evaluate polynomials at point
    p1_vec = zeros(1, χ)
    p2_vec = zeros(χ, 1)
    
    for n in 1:χ
        for i in 0:degree
            p1_vec[1, n] += P1[1, n, i+1] * chebyshevs[i+1](x_val)
            p2_vec[n, 1] += P2[n, 1, i+1] * chebyshevs[i+1](y_val)
        end
    end
    
    # Multiply P1 and P2
    return sum(p1_vec .* p2_vec')  # Element-wise multiplication and then sum
end

# Verify approximation at grid points
grid_size = 50
x_grid = range(domain[1], domain[2], length=grid_size)
y_grid = range(domain[1], domain[2], length=grid_size)

exact_vals = zeros(grid_size, grid_size)
approx_vals = zeros(grid_size, grid_size)

for i in 1:grid_size
    for j in 1:grid_size
        exact_vals[i, j] = gaussian_pdf(x_grid[i], y_grid[j], Σ)
        approx_vals[i, j] = evaluate_stt(x_grid[i], y_grid[j])
    end
end

# Compute error
max_error = maximum(abs.(exact_vals - approx_vals))
rmse = sqrt(sum((exact_vals - approx_vals).^2) / (grid_size^2))

println("Maximum absolute error: $max_error")
println("Root Mean Square Error: $rmse")
println("SVD singular values: $(S[1:min(5,length(S))])...")

# Plot results
p1 = heatmap(x_grid, y_grid, exact_vals', 
             title="Exact Gaussian PDF", 
             xlabel="x", ylabel="y")

p2 = heatmap(x_grid, y_grid, approx_vals', 
             title="STT Approximation", 
             xlabel="x", ylabel="y")

p3 = heatmap(x_grid, y_grid, abs.(exact_vals - approx_vals)', 
             title="Absolute Error", 
             xlabel="x", ylabel="y")

plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
savefig("gaussian_stt_comparison.png")
