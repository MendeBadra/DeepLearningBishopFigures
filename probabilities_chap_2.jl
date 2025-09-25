### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 79d8148b-bdee-4ca3-9e58-b65a1f4402d6
import Pkg; Pkg.activate(".")

# ╔═╡ f4a5051c-48e3-4883-8dc1-245773be58bc
Pkg.add("Distributions")

# ╔═╡ e810de01-29c8-4519-9e64-6046077eed00
using Plots

# ╔═╡ b8af57bb-c461-49af-a9f0-7a9da99b0cb1
using Random,Distributions, PlutoUI,StatsBase

# ╔═╡ 28af8069-05b2-453c-a8cf-977acf23f254
md"n: $(@bind n Slider(2:50, show_value=true, default=2))"

# ╔═╡ e3b53ff3-80a9-4179-ad56-af3a057e1619
# Function to plot Gaussian distributions
function plot_gaussians(n::Int)
    # Generate n random normal distributed values
    data = randn(n)
    println(data)
    # Calculate sample mean (μ_ml) and variance (σ²_ml)
    plot_gaussians(data)
end


# ╔═╡ 63e81789-a7c9-462d-ae79-93c5d71f4e3c
md"""
For gaussian in the diagram
"""

# ╔═╡ f6d92e95-5443-421d-a41c-c20e151da56e
md"`x₀`: $(@bind x0 Slider(-4:.01:4, show_value=true, default=-4.0))"

# ╔═╡ ced9ce62-25ce-4802-992b-70e7ed46bb9e
x1 = x0 + 1.0

# ╔═╡ a4024976-abfc-40d4-85da-1edde29e3969
# Function to plot Gaussian distributions
function plot_gaussians(data::AbstractVector)
    # Generate n random normal distributed values
    # data = randn(n)
	#n = length(data)
    println(data)
    # Calculate sample mean (μ_ml) and variance (σ²_ml)
    μ_ml = mean(data)
    σ²_ml = var(data)

    # True mean and variance for a standard normal distribution
    μ_true = 0
    σ²_true = 1

    # Create the true Gaussian distribution (red)
    x_values = range(-4, 4, length=200)
    true_dist = pdf.(Normal(μ_true, sqrt(σ²_true)), x_values)

    # Create the estimated Gaussian distribution (blue)
    est_dist = pdf.(Normal(μ_ml, sqrt(σ²_ml)), x_values)

    # Plot both distributions
    plot(x_values, true_dist, label="True Gaussian", lw=2, color=:red)
    plot!(x_values, est_dist, label="Estimated Gaussian", lw=2, color=:blue)

    # Add data points to the plot (for clarity)
   scatter!(data, zeros(n), label="Data points", color=:green, legend=:topright)
end


# ╔═╡ e67bfc45-3cae-44f6-a917-5563aa3d4b9c
plot_gaussians(n)

# ╔═╡ ef3b2af4-03a6-4a74-9a2c-465df6a6634b
plot_gaussians([x0, x1])

# ╔═╡ fc1db9c4-e144-4de7-8e4e-8173731aff9b
x3 = randn()

# ╔═╡ 85ee6586-c0bb-40b2-beba-6b2e15cd1699
function plot_gaussians_unbiased(data::AbstractVector)
	n = length(data)
	println(data)
    # Calculate sample mean (μ_ml) and variance (σ²_ml)
    μ_ml = mean(data)
    σ²_ml = n*var(data)/(n-1)

    # True mean and variance for a standard normal distribution
    μ_true = 0
    σ²_true = 1

    # Create the true Gaussian distribution (red)
    x_values = range(-4, 4, length=500)
    true_dist = pdf.(Normal(μ_true, sqrt(σ²_true)), x_values)

    # Create the estimated Gaussian distribution (blue)
    est_dist = pdf.(Normal(μ_ml, sqrt(σ²_ml)), x_values)

    # Plot both distributions
    plot(x_values, true_dist, label="True Gaussian", lw=2, color=:red)
    plot!(x_values, est_dist, label="Estimated Gaussian", lw=2, color=:blue)

    # Add data points to the plot (for clarity)
   scatter!(data, zeros(n), label="Data points", color=:green, legend=:topright)
end

# ╔═╡ 1649679e-26b2-4df9-b599-28df5f261fad
plot_gaussians_unbiased([x0, x1])

# ╔═╡ c05396d2-681b-4d66-948f-3a4cbbdd493b
md"""
### Entropy
Entropy is essentially a measure of disorder with a system.
"""

# ╔═╡ 585fc703-f04f-4b83-a90a-e81d221baa76
md"""
Entropy is a measure of uncertainty or randomness in a distribution. In information theory, entropy `` H `` for a discrete probability distribution ``P = \{p_1, p_2, \dots, p_n\} `` can be defined as:


$H = - \sum_{i=1}^n p_i \log(p_i)$


### Steps to calculate entropy and plot histograms in Julia:

1. **Define the entropy function:**
   - The function will take a probability distribution (histogram) and calculate the entropy.
   
2. **Generate random data:**
   - Create two different distributions: one narrow and one broad to show different entropy values.

3. **Plot the histograms:**
   - Use `Plots.jl` to plot the histograms as shown in the image with probabilities on the y-axis.

### Example Code in Julia

```julia
using Random, StatsBase, Plots

# Function to calculate entropy
function entropy(p)
    # Avoid log(0) issues
    p = p[p .> 0]
    return -sum(p .* log.(p))
end

# Generate two sets of random data (one narrow, one broad)
narrow_data = randn(1000)  # Normal distribution, narrow
broad_data = rand(1000) * 6 - 3  # Uniform distribution, broad

# Calculate histogram and normalized probabilities
function normalized_histogram(data, nbins)
    h = fit(Histogram, data, nbins)
    # Normalize to get probabilities
    p = h.weights / sum(h.weights)
    return h, p
end

# Plot histogram and entropy
function plot_histogram(data, nbins, title_label)
    h, p = normalized_histogram(data, nbins)
    h_ent = entropy(p)
    
    # Plot histogram
    bar(h.edges[1], h.weights / sum(h.weights); linewidth=2, legend=false, alpha=0.5, label="probabilities", fillcolor=:blue, linecolor=:black)
    annotate!(0.1, 0.45, Plots.text("H = $(round(h_ent, digits=2))", :black, :center, 14))
    xlabel!("")
    ylabel!("probabilities")
    title!(title_label)
end

# Create two subplots
plot(layout = (1, 2), size=(1000, 400))

# Plot for narrow distribution
plot_histogram(narrow_data, 30, "Narrow Distribution")

# Plot for broad distribution
plot_histogram(broad_data, 30, "Broad Distribution")
```

### Explanation:
1. **Entropy Calculation:** 
   - The `entropy` function computes the entropy by summing \( -p_i \log(p_i) \) for all non-zero probabilities.
   
2. **Random Data Generation:**
   - `narrow_data`: Gaussian data for a narrow distribution.
   - `broad_data`: Uniformly distributed data for a broad distribution.

3. **Plotting:**
   - Histograms are created using `fit(Histogram, data, nbins)` and then normalized to get probabilities.
   - Histograms are displayed using the `bar` function, and entropy values are displayed on the plot.

This approach should give you plots similar to those in the image, with entropy values calculated for both distributions. Let me know if you need further adjustments or help!
"""

# ╔═╡ d4e19108-263c-4fdc-babd-5577fcd56b79
# Function to calculate entropy
function entropy(p)
    # Avoid log(0) issues
    p = p[p .> 0]
    return -sum(p .* log.(p))
end

# Generate two sets of random data (one narrow, one broad)

# ╔═╡ 5f6cb344-507c-4528-917b-359e2f6e4889
narrow_data = randn(1000)  # Normal distribution, narrow

# ╔═╡ 7f7bda17-03a8-4ec1-9eda-27c0e086fa51
broad_data = rand(1000) .* 6 .- 3  # Uniform distribution, broad

# Calculate histogram and normalized probabilities

# ╔═╡ 49f85870-04fb-4f78-ab2b-c91571e6d3c2
function normalized_histogram(data, nbins)
    h = fit(Histogram, data, nbins=nbins)
    # Normalize to get probabilities
    p = h.weights / sum(h.weights)
    return h, p
end

# Plot histogram and entropy

# ╔═╡ bd718360-619f-4d10-ba81-499a125ef23d
function plot_histogram(data, nbins, title_label)
	    h, p = normalized_histogram(data, nbins)
	    h_ent = entropy(p)
	    
	    # Plot histogram
	    bar(h.edges[1], h.weights / sum(h.weights); linewidth=2, legend=false, alpha=0.5, label="probabilities", fillcolor=:blue, linecolor=:black)
	    annotate!(0.0, maximum(h.weights) / sum(h.weights) * 0.8, Plots.text("H = $(round(h_ent, digits=2))", :black, :center, 14))
	    xlabel!("bins")
	    ylabel!("probabilities")
	    title!(title_label)
	end
	
	# Create two subplots

# ╔═╡ e748daf2-723a-4277-9f6c-65f85e4e152e
plot(layout = (1, 2), size=(1000, 400))
	
	# Plot for narrow distribution

# ╔═╡ 2ee98859-1541-4684-9273-0f54d18159f9
plot_histogram(narrow_data, 30, "Narrow Distribution")
	
	# Plot for broad distribution

# ╔═╡ 97557f62-da1a-4431-8299-f900caf44ee4
plot_histogram(broad_data, 30, "Broad Distribution")

# ╔═╡ 695b0a68-7fca-457f-a332-74aca1064126
let
	
	# Function to calculate entropy
	function entropy(p)
	    # Avoid log(0) issues
	    p = p[p .> 0]
	    return -sum(p .* log.(p))
	end
	
	# Generate two sets of random data (one narrow, one broad)
	narrow_data = randn(1000)  # Normal distribution, narrow
	broad_data = rand(1000) .* 6 .- 3  # Uniform distribution, broad (with broadcasting)
	
	# Function to calculate normalized histogram with keyword argument for nbins
	function normalized_histogram(data; nbins=30)
	    h = fit(Histogram, data; nbins=nbins)  # Using keyword argument `nbins`
	    # Normalize to get probabilities
	    p = h.weights / sum(h.weights)
	    return h, p
	end
	
	# Function to plot histogram and entropy
	function plot_histogram(data; nbins=30, title_label="")
	    h, p = normalized_histogram(data; nbins=nbins)
	    h_ent = entropy(p)
	    
	    # Plot histogram
	    bar(h.edges[1], h.weights / sum(h.weights); linewidth=2, legend=false, alpha=0.5, label="probabilities", fillcolor=:blue, linecolor=:black)
	    
	    # Adjusting annotation placement
	    annotate!(0.0, maximum(h.weights) / sum(h.weights) * 0.8, Plots.text("H = $(round(h_ent, digits=2))", :black, :center, 14))
	    
	    xlabel!("")
	    ylabel!("probabilities")
	    title!(title_label)
	end
	
	# Create two subplots
	plot(layout = (1, 2), size=(1000, 400))
	
	# Plot for narrow distribution
	plot_histogram(narrow_data; nbins=30, title_label="Narrow Distribution")
	
	# Plot for broad distribution
	plot_histogram(broad_data; nbins=30, title_label="Broad Distribution")
end
		

# ╔═╡ Cell order:
# ╠═79d8148b-bdee-4ca3-9e58-b65a1f4402d6
# ╠═f4a5051c-48e3-4883-8dc1-245773be58bc
# ╠═e810de01-29c8-4519-9e64-6046077eed00
# ╠═b8af57bb-c461-49af-a9f0-7a9da99b0cb1
# ╠═28af8069-05b2-453c-a8cf-977acf23f254
# ╠═e67bfc45-3cae-44f6-a917-5563aa3d4b9c
# ╠═e3b53ff3-80a9-4179-ad56-af3a057e1619
# ╟─63e81789-a7c9-462d-ae79-93c5d71f4e3c
# ╠═f6d92e95-5443-421d-a41c-c20e151da56e
# ╠═ced9ce62-25ce-4802-992b-70e7ed46bb9e
# ╠═ef3b2af4-03a6-4a74-9a2c-465df6a6634b
# ╟─a4024976-abfc-40d4-85da-1edde29e3969
# ╠═1649679e-26b2-4df9-b599-28df5f261fad
# ╠═fc1db9c4-e144-4de7-8e4e-8173731aff9b
# ╠═85ee6586-c0bb-40b2-beba-6b2e15cd1699
# ╟─c05396d2-681b-4d66-948f-3a4cbbdd493b
# ╟─585fc703-f04f-4b83-a90a-e81d221baa76
# ╠═d4e19108-263c-4fdc-babd-5577fcd56b79
# ╠═5f6cb344-507c-4528-917b-359e2f6e4889
# ╠═7f7bda17-03a8-4ec1-9eda-27c0e086fa51
# ╠═49f85870-04fb-4f78-ab2b-c91571e6d3c2
# ╠═bd718360-619f-4d10-ba81-499a125ef23d
# ╠═e748daf2-723a-4277-9f6c-65f85e4e152e
# ╠═2ee98859-1541-4684-9273-0f54d18159f9
# ╠═97557f62-da1a-4431-8299-f900caf44ee4
# ╠═695b0a68-7fca-457f-a332-74aca1064126
