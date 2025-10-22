### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ f2f3493b-4105-4927-ad32-7e5d442d4b17
begin
    import Pkg
    Pkg.activate(".")
end

# ╔═╡ 1e4d8fc4-9c9a-4685-8549-61bede61b9c6
begin
    using Plots, Polynomials, PlutoUI, LaTeXStrings
	using LinearAlgebra
    # import PlotlyBase, PlotlyKaleido
end

# ╔═╡ e5b2fa05-c958-4b7f-b1cc-bee364219c46
begin
    using DataFrames
    import StatsBase as SB
end

# ╔═╡ 539f8b0b-b274-43fc-baa6-bc7d49561f2a
let
    # divergence: Monti Hall problem
    using Random
    Random.seed!(1)

    function montyHall(switchPolicy)
        prize, choice = rand(1:3), rand(1:3)
        if prize == choice
            revealed = rand(setdiff(1:3, choice))
        else
            revealed = rand(setdiff(1:3, [prize, choice]))
        end

        if switchPolicy
            choice = setdiff(1:3, [revealed, choice])[1]
        end
        return choice == prize
    end
    N = 10^6
    println("Success probability with policy I (stay): ",
        sum([montyHall(false) for _ in 1:N]) / N)
    println("Success probability with policy II (switch): ",
        sum([montyHall(true) for _ in 1:N]) / N)
end

# ╔═╡ e0884827-602a-4cd6-87ca-04b0999800de
md"""
# Sin function interpolation (Polynomial fitting example)
*Motivation: Deep learning Foundations and Concepts Chapter 1*

So the motivating example from this book introduces a polynomial regression problem of fitting polynomial of degree $M$ to the dataset consisting of $N$ observations, such that $\mathcal{D} = {x_1, ... , x_N}$.
"""

# ╔═╡ a5eacacf-e0d7-4a64-a22b-a58351cbe640
# Library imports below are hidden

# ╔═╡ a16ec8dc-a1ec-4306-aacd-cbf459c3cbbd
gr();

# ╔═╡ bcc21fa0-7a52-4f95-9b11-44bd1eb57852
x = 0:0.01:1; # 100 data points

# ╔═╡ 3b99d7a3-b0f7-43e5-b295-ea664f2f2098
# Underlying distribution samples
y = sin.(2π * x); 

# ╔═╡ 2ff4e294-65eb-4ee0-aaf3-951d0293857c
# maybe plot this one:
plot(x, y, title=L"$sin(x)$ function")

# ╔═╡ 5b67c64f-15cc-412a-a5e4-6444d3844807
md"How many points to take $N$ = $(@bind N Slider(10:2:100, show_value=true, default=10))"

# ╔═╡ 668cbf97-2ee4-4536-ba3f-ddb17c5a9ffb
x_train = x[1]:1/(N-1):x[end] # Uniform distance away

# ╔═╡ d90d9aa3-eb2d-4d9a-8d51-9b943ca0595d
length(x_train)

# ╔═╡ 90ed8f96-41e2-405a-9ac7-b5adb0753544
begin
    Random.seed!(1234)
    t = sin.(2π .* x_train) .+ randn(Float64, N) ./ 5
end

# ╔═╡ 466f7980-7842-4abd-bf2e-50d74e19796c
md"**Total data points $N$ = $(length(t))**"

# ╔═╡ 36aacf7a-7a4d-44df-a8ed-986b635af64f
begin
    p = plot(x, y, xlims=(0, 1.1), ylims=(-1.2, 1.2))
    p = scatter(p, x_train, t, title=L"$sin(x)$ and some Gaussian noise")
end

# ╔═╡ ca40ae75-664e-48f6-8c67-6b63895730ed
md"
Test for $M$ = $N$ = $N points
"

# ╔═╡ de1c70c6-96c8-442a-9e89-65abef0e6583
# Using Polynomials.jl package to interpolate through points
f = fit(Polynomial, x_train, t)
# f = fit(ArnoldiFit, x_train, t, n=N - 1; var=:x);

# ╔═╡ 77a221c8-19ca-4177-b1df-58b50cfb5711
# begin
# 	ENV["JULIA_EDITOR"] = "code"
# 	@edit fit(Polynomial, x_train, t)
# end

# ╔═╡ 1c4d52ad-f2ac-45c6-93d4-cd2b451513a6
M_range = 0:18

# ╔═╡ 2ba56c2a-aeef-46c3-a296-31100951d316
"""The RMS error between y(w, x) and t target variable"""
function error_rms(w::Vector, x::AbstractVector, t::Vector)
    # @assert length(w) < M_range[end] "length of w vector($(length(w))) exceeds the polynomial amount $(M_range[end])"
    @assert length(x) == length(t)
    y = Polynomial(w)
    diff = y.(x) .- t
    return sqrt(1 / N * (diff' * diff))
end;

# ╔═╡ d71f4b77-cbc8-4b42-80a8-3f00690893dd
md"
Degree of freedom for polynomial $M$: $(@bind M Slider(M_range, show_value=true, default=(N > 3) ? 9 : 0))
"

# ╔═╡ 557518ab-5104-416b-986d-27c68e1ff043
begin
    f_poly = fit(Polynomial, x_train, t, M)
end

# ╔═╡ d3c0b19f-e5f2-4862-a53b-b59f61c43637
w_poly = f_poly.coeffs

# ╔═╡ 8fd2fc65-5c49-4c70-8278-55882979dc2a
begin
    p_1 = plot(x, y, xlims=(0, 1.1))
    base_plot = scatter(p_1, x_train, t, title="sinx and some Gaussian noise")
    base_plot = plot(base_plot, f_poly, extrema(x_train)..., label="$M degree fit")
end

# ╔═╡ 7e4766a7-b6db-4632-b584-992cd43b9d59
# degrees = [i for i in 1:length(t)]
# md"Degrees of polynomial denoted by $M$ = $(@bind M Slider())"

# ╔═╡ cd50c6dd-e508-4429-883a-abace4040604
md"""
Error function:

$E(\textbf{w}) = \frac{1}{2} \sum_{n=1}^{N}\{y(x_n,\textbf{w}) - t_n\}^2$


Root mean square (RMS) error

$E_{\text{RMS}} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} \{ y(x_n, \mathbf{w}) - t_n \}^2}$

"""

# ╔═╡ d058f1e2-c781-4430-9b97-552827f1409a
(f_poly.(x_train) - t)' * (f_poly.(x_train) - t)

# ╔═╡ d6899aff-0c1b-4a0e-9039-22c2076c37b1
begin
    # Generate test data from sinx function
    # Number of test points
    N_test = ceil(N * 2.5) |> Integer
    x_test = x[1]:1/(N_test-1):x[end]
    # x_test = rand(N_test)
    t_test = sin.(2π .* x_test) .+ randn(Float64, N_test) ./ 5
    # Plot sin graph
    plot(x, y, xlims=(0, 1.1), ylims=(-1.2, 1.2))
    # plot the training set
    scatter!(x_train, t, label="training set")
    scatter!(x_test, t_test, label="test set", title="sinx and and test data")
end

# ╔═╡ 948cd614-e9e6-4dcf-ac2d-5444ab646022
"""calculate RMS error between two vectors"""
function error_rms(y::Vector, t::Vector)
	@assert length(y - t) == N || length(y - t) == N_test
	return √(1 / (length(y - t)) * (y - t)' * (y - t))
end;

# ╔═╡ c1fcb1f2-3b3c-4527-9f61-4b9f82ff030c
# New error_rms can live alongside another error_rms thanks to multiple dispatch.
error_rms(w_poly, x_train, t)

# ╔═╡ bb7d59cd-5cb4-4b8c-8eef-a43dd103920f
md"Associated error for following plot = $(error_rms(f_poly.(x_train), t))"

# ╔═╡ 2f101741-63db-47cc-815e-0de4169842c1
error_rms(f_poly.(x_train), t)

# ╔═╡ 657cea90-c746-46ea-9afd-ea5bc8feceb5
# training_errors = [error_rms(fit(x[1:r:end], t, i), x[1:r:end], t) for i in degrees]
begin
    polynomials = [fit(Polynomial, x_train, t, M) for M in M_range]
    # print(polynomials)
    training_errors = broadcast(polynomials) do poly
        error_rms(poly.(x_train), t)
    end
end

# ╔═╡ b54bb9fd-d423-446a-8474-230919d81e52
training_errors

# ╔═╡ 9abf8245-f5c1-419f-af57-a81ceef816d2
test_errors = broadcast(polynomials) do poly
    error_rms(poly.(x_test), t_test)
end

# ╔═╡ 90daedc0-eee7-4df0-8e4b-68163ad39d33
# ╠═╡ disabled = true
#=╠═╡
using LaTeXStrings
  ╠═╡ =#

# ╔═╡ 05deca4a-c891-435c-a1f1-5cf8e4a71438
begin
    # Plot
    plot(M_range, training_errors,
        lw=2, label="Training",
        marker=:o, color=:red, xlims=(0,N+2), ylims=(0., 1.))

    plot!(M_range, test_errors,
        lw=2, label="Test",
        xticks=0:length(t),
        marker=:o, color=:blue)

    # Axis labels and title
    xlabel!(L"$M$ - model complexity")
    ylabel!(L"E_{RMS}")
    title!("Training vs Test Error")

    # Customize the legend
    plot!(legend=:topleft, framestyle=:box, legendtitle="Legend")
end

# ╔═╡ 1cfff5d7-9bc5-4195-97d7-79dd67bd6733
md"""
**Quote (Bishop 2024):**
This may seem paradoxical because a polynomial of a given order contains all
lower-order polynomials as special cases. The $M = 9$ polynomial is therefore capable of generating results at least as good as the $M = 3$ polynomial. Furthermore,
we might suppose that the best predictor of new data would be the function $sin(2πx)$
from which the data was generated (and we will see later that this is indeed the case).
We know that a power series expansion of the function $sin(2πx)$ contains terms of all
orders, so we might expect that results should improve monotonically as we increase
$M$.
"""

# ╔═╡ fe324b38-0f97-44ee-b621-fa92f0f734bd
let
    df = DataFrame(W=[M_range;])
    for M in M_range
        # Fit the polynomial of degree M
        local f_degree = fit(x_train, t, M)

        # Extract coefficients and fill with zeros for lower-order terms
        coeffs = f_degree.coeffs
        padded_coeffs = vcat(coeffs, zeros(M_range[end] + 1 - length(coeffs)))
        # Add to DataFrame
        #println(size(padded_coeffs))
        #println(df)
        df[!, "M = $M"] = padded_coeffs
    end
    df
end

# ╔═╡ c4d82cc7-a7bb-4e32-a8cc-c63818dfb0a6
md"""
As you can see on the above table, our polynomial coefficients are becoming more and more varied as M increases.
"""

# ╔═╡ 99b67edf-4470-4161-b216-23b44f89fcb1
md"""
## 1.2.5 Regularization
"""

# ╔═╡ d87194b3-664a-47ca-ab72-ecd029296fe0
md"""
## What I don't understand!
I don't understand that when increasing value of N( number of data points), the error things are becoming less??? Check bishopbook's chapter 1, pg. 11 end of page. I suspect some error calculation is becoming horribly wrong in this notebook.

!!! info "Key takeaways"
	2024.12.9 Monday. I kind of did fix this notebook for demonstration purposes. **`w`** values are also seem to enlarge as M approaches big (Model becoming complex). As for how I fixed it, I didn't know the test and training datasets were drawn uniformly from $[0, 1]$ range.
"""

# ╔═╡ 2159e520-9eac-4e76-8a1b-75cc26f86ea8
md"""
## Regularization
The error function is modified such that:

$E(\textbf{w}) = \frac{1}{2} \sum_{n=1}^{N}\{y(x_n,\textbf{w}) - t_n\}^2 + \frac{\lambda}{2}||\textbf{w}||^2$

This approach is also known as *weight decay* because the parameters i a neural network are called weights and this regularizer encourages them to decay towards zero.

This means that the Root mean error function is modified to:

$E_{\text{RMS}} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} \{ y(x_n, \mathbf{w}) - t_n \}^2 + \frac{\lambda}{N}||\textbf{w}||^2}$
"""

# ╔═╡ 593a8bd7-2886-460f-8867-2ac6467a5cc6
function error_rms_regularized(y::Vector, t::Vector, w::Vector, λ)
    @assert length(y - t) == N || length(y - t) == N_test
    N_local = length(y - t)
    return √(1 / N_local * ((y - t)' * (y - t) + λ / N_local * w'w))
end

# ╔═╡ 674f24f9-fe36-432b-ba7d-3c84beb0587a
md"""
So far, we've used `Polynomials.jl`'s `fit` method, which assumes us the least squares criterion. But since we modified this criterion with a regularizer, we need to implement polynomial regression.
"""

# ╔═╡ 00bda59e-151f-4da3-9e09-6a99921751b9
let # for testing
	V = vander(Polynomial, x_train, M)
	@show ws = V\t
	isequal(ws, w_poly)
end

# ╔═╡ 467c98d8-cf94-4e44-8944-c687bae9d1a4
begin
	# Ok it was promising. Now let's define
	λ = exp(-10) # from bishop book
	V = vander(Polynomial, x_train, M)
	Λ = fill!(similar(V), 0)
	Λ[diagind(Λ)] .= λ
	regularized_ws = (V .+ Λ)\t  
	# Turns out regularization wasn't that hard after all!
end

# ╔═╡ e54120fa-ed6b-484d-86d1-a4f09256266a
begin
	# Let's construct a polynomial from this regularized_ws
	regularized_poly = Polynomial(regularized_ws)
	plot(base_plot, regularized_poly, extrema(x_train)..., label="$M deg. reg. fit")
end

# ╔═╡ c0a28e40-40ce-4b73-b332-4e620ba5eece
# ╠═╡ disabled = true
#=╠═╡
df = DataFrame(w_poly = w_poly, reg_w = regularized_ws)
  ╠═╡ =#

# ╔═╡ 33e6e31f-6692-410c-8609-2b109a7e06af
# ╠═╡ disabled = true
#=╠═╡
begin
    # Access internal fields of the DataFrame `df`
    columns = getfield(df, :columns)
    colindex = getfield(df, :colindex)
    metadata = getfield(df, :metadata)
    colmetadata = getfield(df, :colmetadata)
    allnotemetadata = getfield(df, :allnotemetadata)

    # Displaying the internal fields
    # println("Columns: $columns")
    # println("Column Index: $colindex")
    # println("Metadata: $metadata")
    # println("Column Metadata: $colmetadata")
    # println("All Note Metadata: $allnotemetadata")

end;
  ╠═╡ =#

# ╔═╡ ccaacce9-8277-43f1-8d00-ec6d4b7fa0c8
md"""
# Chapter2: Probabilities [NOT COMPLETE]
"""

# ╔═╡ 687a76a1-6ab5-42f7-967c-606bf008e1f4
md"""
We can use Plotly for interactivity here.
```
using Plotly??
```
"""

# ╔═╡ 5454f7bd-a0f5-4342-b452-554090aee17e
yₘ(x1, x2) = sin(2π * x1) * sin(2π * x2)

# ╔═╡ 1d3e80c1-883f-4f37-9c78-c6b067d4cee3
surface(x, x, yₘ; xlabel="x1", ylabel="x2", zlabel="y")

# ╔═╡ 8be93355-36ff-42c5-ae9f-b78a89adefd1
@which surface


# ╔═╡ f181798a-bfaa-4942-a7e0-9a159cb3c054
begin #ChatGPT
    # Number of data points
    n = 100

    # Generate data for (b) - where x2 is unobserved (random)
    x1_vals_b = rand(x, n)  # Random values for x1
    x2_vals_b = rand(x, n)  # Random values for x2 (unobserved)
    y_vals_b = [yₘ(x1_vals_b[i], x2_vals_b[i]) + 0.1 * randn() for i in 1:n]  # Add Gaussian noise

    # Plot (b) - Scatter plot of x1 vs y, with high noise
    scatter(x1_vals_b, y_vals_b, label="(b) Unobserved x2", xlabel="x1", ylabel="y", color=:red, legend=false)
    #savefig("plot_b.png")
end


# ╔═╡ 575bf6b7-a711-42c5-8059-46ea14371f94
begin
    # Generate data for (c) - where x2 is fixed
    x1_vals_c = rand(x, n)  # Random values for x1
    x2_fixed = π / 2     # Fixed value for x2
    y_vals_c = [yₘ(x1_vals_c[i], x2_fixed) + 0.1 * randn() for i in 1:n]  # Add Gaussian noise

    # Plot (c) - Scatter plot of x1 vs y, with lower noise (x2 fixed)
    scatter(x1_vals_c, y_vals_c, label="(c) Fixed x2", xlabel="x1", ylabel="y", color=:red, legend=false)
    #savefig("plot_c.png")
end

# ╔═╡ cf667e1c-68eb-455d-a376-596da58e82cd
md"""
### 2.1.2 The sum and product rules

Example:
```
julia> coin_toss() = (rand(1:2) == 1) ? "too" : "suld" 
coin_toss (generic function with 1 method)

julia> coin_toss()
"too"

julia> dice_roll() = rand(1:6)
dice_roll (generic function with 1 method)

julia> N = 10
10

julia> [(dice_roll(), coin_toss()) for i in 1:N]
10-element Vector{Tuple{Int64, String}}:
 (5, "too")
 (4, "suld")
 (4, "too")
 (3, "too")
 (2, "too")
 (1, "suld")
 (1, "too")
 (2, "suld")
 (6, "suld")
 (6, "too")
```
"""

# ╔═╡ 54d1d73f-c096-40ff-8859-0222940c92f0
md"""
### 2.2 Probability densities
"""

# ╔═╡ f4a7978a-45af-4c7f-85e5-16819752a712
function gaussian(x::Float64, μ::Float64, σ::Float64)
    return 1 / (2π)
end

# ╔═╡ Cell order:
# ╟─e0884827-602a-4cd6-87ca-04b0999800de
# ╟─a5eacacf-e0d7-4a64-a22b-a58351cbe640
# ╟─f2f3493b-4105-4927-ad32-7e5d442d4b17
# ╠═1e4d8fc4-9c9a-4685-8549-61bede61b9c6
# ╠═e5b2fa05-c958-4b7f-b1cc-bee364219c46
# ╟─a16ec8dc-a1ec-4306-aacd-cbf459c3cbbd
# ╠═bcc21fa0-7a52-4f95-9b11-44bd1eb57852
# ╠═3b99d7a3-b0f7-43e5-b295-ea664f2f2098
# ╟─2ff4e294-65eb-4ee0-aaf3-951d0293857c
# ╟─466f7980-7842-4abd-bf2e-50d74e19796c
# ╠═668cbf97-2ee4-4536-ba3f-ddb17c5a9ffb
# ╠═d90d9aa3-eb2d-4d9a-8d51-9b943ca0595d
# ╠═90ed8f96-41e2-405a-9ac7-b5adb0753544
# ╟─5b67c64f-15cc-412a-a5e4-6444d3844807
# ╠═36aacf7a-7a4d-44df-a8ed-986b635af64f
# ╟─ca40ae75-664e-48f6-8c67-6b63895730ed
# ╠═de1c70c6-96c8-442a-9e89-65abef0e6583
# ╠═77a221c8-19ca-4177-b1df-58b50cfb5711
# ╠═1c4d52ad-f2ac-45c6-93d4-cd2b451513a6
# ╠═557518ab-5104-416b-986d-27c68e1ff043
# ╠═d3c0b19f-e5f2-4862-a53b-b59f61c43637
# ╠═2ba56c2a-aeef-46c3-a296-31100951d316
# ╟─d71f4b77-cbc8-4b42-80a8-3f00690893dd
# ╠═c1fcb1f2-3b3c-4527-9f61-4b9f82ff030c
# ╟─bb7d59cd-5cb4-4b8c-8eef-a43dd103920f
# ╠═8fd2fc65-5c49-4c70-8278-55882979dc2a
# ╠═7e4766a7-b6db-4632-b584-992cd43b9d59
# ╟─cd50c6dd-e508-4429-883a-abace4040604
# ╠═948cd614-e9e6-4dcf-ac2d-5444ab646022
# ╠═2f101741-63db-47cc-815e-0de4169842c1
# ╠═d058f1e2-c781-4430-9b97-552827f1409a
# ╠═657cea90-c746-46ea-9afd-ea5bc8feceb5
# ╠═b54bb9fd-d423-446a-8474-230919d81e52
# ╠═d6899aff-0c1b-4a0e-9039-22c2076c37b1
# ╠═9abf8245-f5c1-419f-af57-a81ceef816d2
# ╠═90daedc0-eee7-4df0-8e4b-68163ad39d33
# ╠═05deca4a-c891-435c-a1f1-5cf8e4a71438
# ╟─1cfff5d7-9bc5-4195-97d7-79dd67bd6733
# ╟─fe324b38-0f97-44ee-b621-fa92f0f734bd
# ╟─c4d82cc7-a7bb-4e32-a8cc-c63818dfb0a6
# ╠═33e6e31f-6692-410c-8609-2b109a7e06af
# ╟─99b67edf-4470-4161-b216-23b44f89fcb1
# ╟─d87194b3-664a-47ca-ab72-ecd029296fe0
# ╟─2159e520-9eac-4e76-8a1b-75cc26f86ea8
# ╠═593a8bd7-2886-460f-8867-2ac6467a5cc6
# ╟─674f24f9-fe36-432b-ba7d-3c84beb0587a
# ╠═00bda59e-151f-4da3-9e09-6a99921751b9
# ╠═467c98d8-cf94-4e44-8944-c687bae9d1a4
# ╠═e54120fa-ed6b-484d-86d1-a4f09256266a
# ╠═c0a28e40-40ce-4b73-b332-4e620ba5eece
# ╟─ccaacce9-8277-43f1-8d00-ec6d4b7fa0c8
# ╟─687a76a1-6ab5-42f7-967c-606bf008e1f4
# ╠═5454f7bd-a0f5-4342-b452-554090aee17e
# ╠═1d3e80c1-883f-4f37-9c78-c6b067d4cee3
# ╠═8be93355-36ff-42c5-ae9f-b78a89adefd1
# ╠═f181798a-bfaa-4942-a7e0-9a159cb3c054
# ╠═575bf6b7-a711-42c5-8059-46ea14371f94
# ╟─cf667e1c-68eb-455d-a376-596da58e82cd
# ╟─54d1d73f-c096-40ff-8859-0222940c92f0
# ╠═539f8b0b-b274-43fc-baa6-bc7d49561f2a
# ╠═f4a7978a-45af-4c7f-85e5-16819752a712
