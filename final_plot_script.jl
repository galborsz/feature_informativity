using Plots
using StatsPlots
using JSON
using Statistics
using StatsBase
using HypothesisTests: MannWhitneyUTest, SignedRankTest, pvalue
using Statistics
using CSV
using DataFrames
# using CairoMakie

# Define inventories
inventories = ["HC", "SPE", "JFH"]
colors = [:red, :green, :blue]

# Load all data first
all_data = Dict()
for inv in inventories
    filename = "data_all_languages_$(inv)_features.json"
    all_data[inv] = JSON.parsefile(filename)
end

# Load pb_languages_formatted.csv to get actual phoneme inventories
pb_languages_file = joinpath("phonemic_inventories", "pb_languages_formatted.csv")
pb_languages = CSV.read(pb_languages_file, DataFrame)

# Load weighted_avg_mdl from JSON file
weighted_avg_mdl_raw = JSON.parsefile("weighted_avg_mdl_data.json")

# Convert to proper types (JSON loads as Vector{Any}, need Vector{Float64})
weighted_avg_mdl = Dict(inv => Dict("Real" => Float64[], "Random" => Float64[]) for inv in inventories)
for inv in inventories
    weighted_avg_mdl[inv]["Real"] = Float64.(weighted_avg_mdl_raw[inv]["Real"])
    weighted_avg_mdl[inv]["Random"] = Float64.(weighted_avg_mdl_raw[inv]["Random"])
end

# Helper function to compute rank-biserial effect size for paired samples
function rank_biserial_paired(x::Vector{<:Real}, y::Vector{<:Real})
    d = x .- y
    pos = sum(d .> 0)
    neg = sum(d .< 0)
    ties = sum(d .== 0)
    n = pos + neg + ties
    A = (pos + 0.5 * ties) / n
    return 2A - 1    # in [-1,1]
end

# Count frequencies for each inventory system
function count_frequencies(values, bin_edges)
    counts = zeros(Int, length(bin_edges) - 1)
    for val in values
        for i in 1:(length(bin_edges) - 1)
            if bin_edges[i] <= val < bin_edges[i + 1]
                counts[i] += 1
                break
            elseif i == length(bin_edges) - 1 && val == bin_edges[end]
                counts[i] += 1
                break
            end
        end
    end
    return counts
end


# ============================================================
# PLOT 1 VIOLIN VERSION: Average MDL per language (violin plot)
# ============================================================

# Collect average MDL per language for each inventory
avg_mdl_per_language = Dict(inv => Float64[] for inv in inventories)

for inv in inventories
    for (language, lang_data) in all_data[inv]
        min_lengths = lang_data["min_lengths"]
        # Calculate average MDL for this language
        mdl_values_lang = [Float64(mdl) for (feature, mdl) in min_lengths]
        push!(avg_mdl_per_language[inv], mean(mdl_values_lang))
    end
end

# Create bins for language averages with more bins
all_lang_avgs = vcat(avg_mdl_per_language["HC"], avg_mdl_per_language["SPE"], avg_mdl_per_language["JFH"])

println("\n============================================================")
println("Creating violin plot version of Plot 1...")
println("============================================================")

# Create violin plot for language averages
p2_violin = plot(
    xlabel="Feature System",
    ylabel="Average MDL per Language",
    legend=false,
    size=(800, 700),
    margin=10Plots.mm,
    bottom_margin=12Plots.mm,
    left_margin=12Plots.mm,
    top_margin=10Plots.mm,
    guidefontsize=14,
    tickfontsize=12,
    framestyle=:box,
    grid=:y,
    gridalpha=0.3,
    thickness_scaling=1.5,
    xticks=(1:3, inventories)
)

# p2_violin.topspinevisible = false
# p2_violin.rightspinevisible = false

# Add violin plots for each inventory
for (idx, inv) in enumerate(inventories)
    violin!(p2_violin, 
            fill(idx, length(avg_mdl_per_language[inv])),
            avg_mdl_per_language[inv],
            fillcolor=colors[idx],
            fillalpha=0.6,
            linecolor=colors[idx],
            linewidth=2,
            side=:both,
            label="")
    
    # Add median line
    med_val = median(avg_mdl_per_language[inv])
    plot!(p2_violin, [idx - 0.3, idx + 0.3], [med_val, med_val],
          color=colors[idx],
          linewidth=2,
          label="")
end

# Perform Wilcoxon signed rank tests and add significance brackets
println("\nWilcoxon signed rank tests for Plot 2 violin (Real language distributions):")

# Find the min and max y-values across all distributions
min_y = minimum([minimum(avg_mdl_per_language[inv]) for inv in inventories])
max_y = maximum([maximum(avg_mdl_per_language[inv]) for inv in inventories])
data_range = max_y - min_y
bracket_spacing = data_range * 0.08  # Vertical spacing between bracket levels as % of data range
tick_height = data_range * 0.02  # Height of vertical tick marks at edges

# Function to get star notation based on p-value
function get_stars(p_value)
    if p_value < 0.001
        return "***"
    elseif p_value < 0.01
        return "**"
    elseif p_value < 0.05
        return "*"
    else
        return "ns"
    end
end

# Test 1: HC vs SPE (positions 1 and 2) - lowest bracket
test_hc_spe_violin = SignedRankTest(avg_mdl_per_language["HC"], avg_mdl_per_language["SPE"])
p_hc_spe_violin = pvalue(test_hc_spe_violin)
r_hc_spe_violin = rank_biserial_paired(avg_mdl_per_language["HC"], avg_mdl_per_language["SPE"])
n_hc_spe_violin = length(avg_mdl_per_language["HC"])
stars_hc_spe = get_stars(p_hc_spe_violin)

println("  HC vs SPE:")
println("    p-value = $(round(p_hc_spe_violin, sigdigits=4))")
println("    effect size (rank-biserial) = $(round(r_hc_spe_violin, digits=3))")
println("    sample size = $n_hc_spe_violin")
println("    significance: $stars_hc_spe")

if stars_hc_spe != "ns"
    # Add bracket line and stars for HC vs SPE
    y_bracket_1 = max_y + bracket_spacing
    # Horizontal line
    plot!(p2_violin, [1, 2], [y_bracket_1, y_bracket_1], 
        color=:black, linewidth=2, label="")
    # Left vertical tick
    plot!(p2_violin, [1, 1], [y_bracket_1 - tick_height, y_bracket_1 + tick_height], 
        color=:black, linewidth=2, label="")
    # Right vertical tick
    plot!(p2_violin, [2, 2], [y_bracket_1 - tick_height, y_bracket_1 + tick_height], 
        color=:black, linewidth=2, label="")
    # Stars
    annotate!(p2_violin, 1.5, y_bracket_1 + 0.01, text(stars_hc_spe, 14, :center))
end

# Test 2: SPE vs JFH (positions 2 and 3) - middle bracket
test_spe_jfh_violin = SignedRankTest(avg_mdl_per_language["SPE"], avg_mdl_per_language["JFH"])
p_spe_jfh_violin = pvalue(test_spe_jfh_violin)
r_spe_jfh_violin = rank_biserial_paired(avg_mdl_per_language["SPE"], avg_mdl_per_language["JFH"])
n_spe_jfh_violin = length(avg_mdl_per_language["SPE"])
stars_spe_jfh = get_stars(p_spe_jfh_violin)

println("  SPE vs JFH:")
println("    p-value = $(round(p_spe_jfh_violin, sigdigits=4))")
println("    effect size (rank-biserial) = $(round(r_spe_jfh_violin, digits=3))")
println("    sample size = $n_spe_jfh_violin")
println("    significance: $stars_spe_jfh")

if stars_spe_jfh != "ns"
    # Add bracket line and stars for SPE vs JFH
    y_bracket_2 = max_y + 2 * bracket_spacing
    # Horizontal line
    plot!(p2_violin, [2, 3], [y_bracket_2, y_bracket_2], 
        color=:black, linewidth=2, label="")
    # Left vertical tick
    plot!(p2_violin, [2, 2], [y_bracket_2 - tick_height, y_bracket_2 + tick_height], 
        color=:black, linewidth=2, label="")
    # Right vertical tick
    plot!(p2_violin, [3, 3], [y_bracket_2 - tick_height, y_bracket_2 + tick_height], 
        color=:black, linewidth=2, label="")
    # Stars
    annotate!(p2_violin, 2.5, y_bracket_2 + 0.01, text(stars_spe_jfh, 14, :center))
end

# Test 3: HC vs JFH (positions 1 and 3) - highest bracket
test_hc_jfh_violin = SignedRankTest(avg_mdl_per_language["HC"], avg_mdl_per_language["JFH"])
p_hc_jfh_violin = pvalue(test_hc_jfh_violin)
r_hc_jfh_violin = rank_biserial_paired(avg_mdl_per_language["HC"], avg_mdl_per_language["JFH"])
n_hc_jfh_violin = length(avg_mdl_per_language["HC"])
stars_hc_jfh = get_stars(p_hc_jfh_violin)

println("  HC vs JFH:")
println("    p-value = $(round(p_hc_jfh_violin, sigdigits=4))")
println("    effect size (rank-biserial) = $(round(r_hc_jfh_violin, digits=3))")
println("    sample size = $n_hc_jfh_violin")
println("    significance: $stars_hc_jfh")

if stars_hc_jfh != "ns"
    # Add bracket line and stars for HC vs JFH
    y_bracket_3 = max_y + 3 * bracket_spacing
    # Horizontal line
    plot!(p2_violin, [1, 3], [y_bracket_3, y_bracket_3], 
        color=:black, linewidth=2, label="")
    # Left vertical tick
    plot!(p2_violin, [1, 1], [y_bracket_3 - tick_height, y_bracket_3 + tick_height], 
        color=:black, linewidth=2, label="")
    # Right vertical tick
    plot!(p2_violin, [3, 3], [y_bracket_3 - tick_height, y_bracket_3 + tick_height], 
        color=:black, linewidth=2, label="")
    # Stars
    annotate!(p2_violin, 2, y_bracket_3 + 0.01, text(stars_hc_jfh, 14, :center))
end

# Adjust y-axis limits to accommodate brackets
y_lower = min_y - data_range * 0.05  # Add 5% padding below
y_upper = max_y + 4 * bracket_spacing + data_range * 0.08  # Add space for brackets plus padding
ylims!(p2_violin, (y_lower, y_upper))

# Save violin plot
savefig(p2_violin, "language_average_mdl_violin.png")

println("\nLanguage average violin plot saved as language_average_mdl_violin.png")

# ============================================================
# PLOT 1 HISTOGRAM VERSION: Average informativity per language
# ============================================================

# Collect average informativity (1/MDL) per language for each inventory
avg_mdl_per_language = Dict(inv => Float64[] for inv in inventories)

for inv in inventories
    for (language, lang_data) in all_data[inv]
        min_lengths = lang_data["min_lengths"]
        # Calculate average informativity (1/MDL) for this language
        mdl_values_lang = [Float64(mdl) for (feature, mdl) in min_lengths]
        informativity_values = 1.0 ./ mdl_values_lang
        push!(avg_mdl_per_language[inv], mean(informativity_values))
    end
end

# Create bins for language averages with more bins
all_lang_avgs = vcat(avg_mdl_per_language["HC"], avg_mdl_per_language["SPE"], avg_mdl_per_language["JFH"])
min_avg = minimum(all_lang_avgs)
max_avg = maximum(all_lang_avgs)
# Use 40 bins for better resolution
bin_width_lang = (max_avg - min_avg) / 40
avg_bin_edges = min_avg:bin_width_lang:(max_avg + bin_width_lang)

# Count frequencies for language averages
hc_lang_counts = count_frequencies(avg_mdl_per_language["HC"], avg_bin_edges)
spe_lang_counts = count_frequencies(avg_mdl_per_language["SPE"], avg_bin_edges)
jfh_lang_counts = count_frequencies(avg_mdl_per_language["JFH"], avg_bin_edges)

# Create x-values for language average plot
avg_x_values = [avg_bin_edges[i] for i in 1:(length(avg_bin_edges) - 1)]

# Calculate means and medians of language averages
mean_hc_lang = mean(avg_mdl_per_language["HC"])
mean_spe_lang = mean(avg_mdl_per_language["SPE"])
mean_jfh_lang = mean(avg_mdl_per_language["JFH"])

median_hc_lang = median(avg_mdl_per_language["HC"])
median_spe_lang = median(avg_mdl_per_language["SPE"])
median_jfh_lang = median(avg_mdl_per_language["JFH"])

# Calculate y-axis limit for language plot
max_lang_count = maximum([maximum(hc_lang_counts), maximum(spe_lang_counts), maximum(jfh_lang_counts)])
y_limit_lang = max_lang_count * 1.1

# Create overlapping histogram for language averages
p2 = plot(
    avg_x_values,
    hc_lang_counts,
    seriestype=:bar,
    label="HC",
    xlabel="Average Informativity per Language",
    ylabel="Language Count",
    fillcolor=colors[1],
    fillalpha=0.5,
    linecolor=colors[1],
    linewidth=1.5,
    bar_width=bin_width_lang,
    ylims=(0, y_limit_lang),
    legend=:topright,
    size=(1000, 700),
    margin=10Plots.mm,
    bottom_margin=12Plots.mm,
    left_margin=12Plots.mm,
    top_margin=10Plots.mm,
    guidefontsize=9,
    tickfontsize=9,
    legendfontsize=9,
    framestyle=:box,
    grid=:y,
    gridalpha=0.3,
    thickness_scaling=1.5
)

# Add SPE and JFH as overlapping bars
bar!(p2, avg_x_values, spe_lang_counts,
     label="SPE",
     fillcolor=colors[2],
     fillalpha=0.5,
     linecolor=colors[2],
     linewidth=1.5,
     bar_width=bin_width_lang)

bar!(p2, avg_x_values, jfh_lang_counts,
     label="JFH",
     fillcolor=colors[3],
     fillalpha=0.5,
     linecolor=colors[3],
     linewidth=1.5,
     bar_width=bin_width_lang)

# Add vertical lines for medians of language averages (without legend labels)
vline!(p2, [median_hc_lang], color=colors[1], linestyle=:dash, linewidth=2, label="")
vline!(p2, [median_spe_lang], color=colors[2], linestyle=:dash, linewidth=2, label="")
vline!(p2, [median_jfh_lang], color=colors[3], linestyle=:dash, linewidth=2, label="")

# Create mini plots showing each distribution separately (vertical layout)
p2_hc = plot(
    avg_x_values,
    hc_lang_counts,
    seriestype=:bar,
    label="",
    title="HC",
    fillcolor=colors[1],
    fillalpha=0.6,
    linecolor=colors[1],
    linewidth=1.5,
    bar_width=bin_width_lang,
    ylims=(0, y_limit_lang),
    size=(180, 200),
    titlefontsize=10,
    guidefontsize=9,
    tickfontsize=8,
    legendfontsize=8,
    framestyle=:box,
    grid=:y,
    gridalpha=0.3,
    # margin=3Plots.mm,
    # bottom_margin=5Plots.mm,
    # top_margin=8Plots.mm,
    legend=:topright
)
vline!(p2_hc, [median_hc_lang], color=colors[1], linestyle=:dash, linewidth=1.5, 
       label="median: $(round(median_hc_lang, digits=2))")

p2_spe = plot(
    avg_x_values,
    spe_lang_counts,
    seriestype=:bar,
    label="",
    title="SPE",
    fillcolor=colors[2],
    fillalpha=0.6,
    linecolor=colors[2],
    linewidth=1.5,
    bar_width=bin_width_lang,
    ylims=(0, y_limit_lang),
    size=(180, 200),
    titlefontsize=10,
    guidefontsize=9,
    tickfontsize=8,
    legendfontsize=8,
    framestyle=:box,
    grid=:y,
    gridalpha=0.3,
    # margin=3Plots.mm,
    # bottom_margin=5Plots.mm,
    # top_margin=8Plots.mm,
    legend=:topright
)
vline!(p2_spe, [median_spe_lang], color=colors[2], linestyle=:dash, linewidth=1.5, 
       label="median: $(round(median_spe_lang, digits=2))")

p2_jfh = plot(
    avg_x_values,
    jfh_lang_counts,
    seriestype=:bar,
    label="",
    title="JFH",
    xlabel="Average Informativity per Language",
    fillcolor=colors[3],
    fillalpha=0.6,
    linecolor=colors[3],
    linewidth=1.5,
    bar_width=bin_width_lang,
    ylims=(0, y_limit_lang),
    size=(180, 200),
    titlefontsize=10,
    guidefontsize=9,
    tickfontsize=8,
    legendfontsize=8,
    framestyle=:box,
    grid=:y,
    gridalpha=0.3,
    # margin=3Plots.mm,
    # bottom_margin=5Plots.mm,
    # top_margin=8Plots.mm,
    legend=:topright
)
vline!(p2_jfh, [median_jfh_lang], color=colors[3], linestyle=:dash, linewidth=1.5, 
       label="median: $(round(median_jfh_lang, digits=2))")

# Combine mini plots vertically with shared y-axis label
p2_mini = plot(p2_hc, p2_spe, p2_jfh, 
               layout=(3, 1), 
               size=(300, 650),
               ylabel="Language Count",
               left_margin=12Plots.mm)

# Combine main plot with mini plots horizontally
p2_combined = plot(p2, p2_mini, layout=(1, 2), size=(1300, 700))

# Save language average plot
savefig(p2_combined, "language_average_informativity_histogram.png")

println("\nLanguage average plot saved as language_average_informativity_histogram.png")
println("\nLanguage average statistics:")
for inv in inventories
    println("\n$inv:")
    println("  Total languages: $(length(avg_mdl_per_language[inv]))")
    println("  Mean of language averages: $(round(mean(avg_mdl_per_language[inv]), digits=2))")
    println("  Median of language averages: $(round(median(avg_mdl_per_language[inv]), digits=2))")
end


# ============================================================
# PLOT 1 DENSITY VERSION: Average informativity per language (density plot)
# ============================================================

println("\n============================================================")
println("Creating density plot version of Plot 2...")
println("============================================================")

# Create density plot for language averages
p2_density = plot(
    xlabel="Average Informativity per Language",
    ylabel="Density",
    legend=:topleft,
    size=(1000, 700),
    margin=10Plots.mm,
    bottom_margin=12Plots.mm,
    left_margin=12Plots.mm,
    top_margin=10Plots.mm,
    guidefontsize=14,
    tickfontsize=12,
    legendfontsize=11,
    framestyle=:box,
    grid=:y,
    gridalpha=0.3,
    thickness_scaling=1.5
)

# Add density curves for each inventory
density!(p2_density, avg_mdl_per_language["HC"],
         label="HC",
         color=colors[1],
         linewidth=2.5,
         fillrange=0,
         fillalpha=0.3,
         fillcolor=colors[1])

density!(p2_density, avg_mdl_per_language["SPE"],
         label="SPE",
         color=colors[2],
         linewidth=2.5,
         fillrange=0,
         fillalpha=0.3,
         fillcolor=colors[2])

density!(p2_density, avg_mdl_per_language["JFH"],
         label="JFH",
         color=colors[3],
         linewidth=2.5,
         fillrange=0,
         fillalpha=0.3,
         fillcolor=colors[3])

# Add vertical lines for medians (without legend labels to avoid clutter)
vline!(p2_density, [median_hc_lang], color=colors[1], linestyle=:dash, linewidth=2, label="")
vline!(p2_density, [median_spe_lang], color=colors[2], linestyle=:dash, linewidth=2, label="")
vline!(p2_density, [median_jfh_lang], color=colors[3], linestyle=:dash, linewidth=2, label="")

# Perform Wilcoxon signed rank tests between all pairs
println("\nWilcoxon signed rank tests for Plot 2 (Real language distributions):")

# HC vs SPE
test_hc_spe = SignedRankTest(avg_mdl_per_language["HC"], avg_mdl_per_language["SPE"])
p_hc_spe = pvalue(test_hc_spe)
r_hc_spe = rank_biserial_paired(avg_mdl_per_language["HC"], avg_mdl_per_language["SPE"])
n_hc_spe = length(avg_mdl_per_language["HC"])
println("  HC vs SPE:")
println("    p-value = $(round(p_hc_spe, sigdigits=4))")
println("    effect size (rank-biserial) = $(round(r_hc_spe, digits=3))")
println("    sample size = $n_hc_spe")

# HC vs JFH
test_hc_jfh = SignedRankTest(avg_mdl_per_language["HC"], avg_mdl_per_language["JFH"])
p_hc_jfh = pvalue(test_hc_jfh)
r_hc_jfh = rank_biserial_paired(avg_mdl_per_language["HC"], avg_mdl_per_language["JFH"])
n_hc_jfh = length(avg_mdl_per_language["HC"])
println("  HC vs JFH:")
println("    p-value = $(round(p_hc_jfh, sigdigits=4))")
println("    effect size (rank-biserial) = $(round(r_hc_jfh, digits=3))")
println("    sample size = $n_hc_jfh")

# SPE vs JFH
test_spe_jfh = SignedRankTest(avg_mdl_per_language["SPE"], avg_mdl_per_language["JFH"])
p_spe_jfh = pvalue(test_spe_jfh)
r_spe_jfh = rank_biserial_paired(avg_mdl_per_language["SPE"], avg_mdl_per_language["JFH"])
n_spe_jfh = length(avg_mdl_per_language["SPE"])
println("  SPE vs JFH:")
println("    p-value = $(round(p_spe_jfh, sigdigits=4))")
println("    effect size (rank-biserial) = $(round(r_spe_jfh, digits=3))")
println("    sample size = $n_spe_jfh")

# Create mini density plots showing each distribution separately (vertical layout)
p2_hc_density = plot(
    title="HC",
    size=(180, 200),
    titlefontsize=10,
    guidefontsize=9,
    tickfontsize=8,
    legendfontsize=8,
    framestyle=:box,
    grid=:y,
    gridalpha=0.3,
    legend=:topleft
)
density!(p2_hc_density, avg_mdl_per_language["HC"],
         label="",
         color=colors[1],
         linewidth=2,
         fillrange=0,
         fillalpha=0.5,
         fillcolor=colors[1])
vline!(p2_hc_density, [median_hc_lang], color=colors[1], linestyle=:dash, linewidth=1.5, 
       label="median: $(round(median_hc_lang, digits=2))")

p2_spe_density = plot(
    title="SPE",
    size=(180, 200),
    titlefontsize=10,
    guidefontsize=9,
    tickfontsize=8,
    legendfontsize=8,
    framestyle=:box,
    grid=:y,
    gridalpha=0.3,
    legend=:topleft
)
density!(p2_spe_density, avg_mdl_per_language["SPE"],
         label="",
         color=colors[2],
         linewidth=2,
         fillrange=0,
         fillalpha=0.5,
         fillcolor=colors[2])
vline!(p2_spe_density, [median_spe_lang], color=colors[2], linestyle=:dash, linewidth=1.5, 
       label="median: $(round(median_spe_lang, digits=2))")

p2_jfh_density = plot(
    xlabel="Average Informativity per Language",
    title="JFH",
    size=(180, 200),
    titlefontsize=10,
    guidefontsize=9,
    tickfontsize=8,
    legendfontsize=8,
    framestyle=:box,
    grid=:y,
    gridalpha=0.3,
    legend=:topleft
)
density!(p2_jfh_density, avg_mdl_per_language["JFH"],
         label="",
         color=colors[3],
         linewidth=2,
         fillrange=0,
         fillalpha=0.5,
         fillcolor=colors[3])
vline!(p2_jfh_density, [median_jfh_lang], color=colors[3], linestyle=:dash, linewidth=1.5, 
       label="median: $(round(median_jfh_lang, digits=2))")

# Combine mini density plots vertically with shared y-axis label
p2_mini_density = plot(p2_hc_density, p2_spe_density, p2_jfh_density, 
                       layout=(3, 1), 
                       size=(300, 650),
                       ylabel="Density",
                       left_margin=12Plots.mm)

# Combine main plot with mini plots horizontally
p2_combined_density = plot(p2_density, p2_mini_density, layout=(1, 2), size=(1300, 700))

# Save density plot
savefig(p2_combined_density, "language_average_informativity_density.png")

println("\nLanguage average density plot saved as language_average_informativity_density.png")




# ============================================================
# PLOT 2 DENSITY VERSION: Individual inventory density plots with Real and Random
# ============================================================

println("\n============================================================")
println("Creating density plot versions of Plot 2...")
println("============================================================")

# Create individual density plots for each inventory
for inv in inventories
    if !isempty(weighted_avg_mdl[inv]["Real"]) || !isempty(weighted_avg_mdl[inv]["Random"])
        # Determine color for this inventory
        color_idx = findfirst(==(inv), inventories)
        color_real = colors[color_idx]
        color_random = :gray
        
        # Calculate statistics
        real_median = !isempty(weighted_avg_mdl[inv]["Real"]) ? median(weighted_avg_mdl[inv]["Real"]) : 0.0
        random_median = !isempty(weighted_avg_mdl[inv]["Random"]) ? median(weighted_avg_mdl[inv]["Random"]) : 0.0
        
        # Create density plot
        p_inv_density = plot(
            xlabel="Average Minimal Description Length",
            ylabel="Density",
            title="Feature system: $inv",
            legend=:topleft,
            size=(1000, 600),
            margin=10Plots.mm,
            bottom_margin=12Plots.mm,
            left_margin=12Plots.mm,
            top_margin=12Plots.mm,
            guidefontsize=14,
            tickfontsize=12,
            legendfontsize=11,
            framestyle=:box,
            grid=:y,
            gridalpha=0.3,
            thickness_scaling=1.5
        )
        
        # Add Real data density
        if !isempty(weighted_avg_mdl[inv]["Real"])
            density!(p_inv_density, weighted_avg_mdl[inv]["Real"],
                     label="Real",
                     color=color_real,
                     linewidth=2.5,
                     fillrange=0,
                     fillalpha=0.4,
                     fillcolor=color_real)
        end
        
        # Add Random data density
        if !isempty(weighted_avg_mdl[inv]["Random"])
            density!(p_inv_density, weighted_avg_mdl[inv]["Random"],
                     label="Random",
                     color=color_random,
                     linewidth=2.5,
                     fillrange=0,
                     fillalpha=0.3,
                     fillcolor=color_random)
        end
        
        # Add vertical lines for medians
        if !isempty(weighted_avg_mdl[inv]["Real"])
            vline!(p_inv_density, [real_median], 
                   color=color_real, 
                   linestyle=:dash, 
                   linewidth=2, 
                   label="Real Median: $(round(real_median, digits=2))")
        end
        
        if !isempty(weighted_avg_mdl[inv]["Random"])
            vline!(p_inv_density, [random_median], 
                   color=color_random, 
                   linestyle=:dash, 
                   linewidth=2, 
                   label="Random Median: $(round(random_median, digits=2))")
        end
        
        # Perform MannWhitneyUTest test between Real and Random
        if !isempty(weighted_avg_mdl[inv]["Real"]) && !isempty(weighted_avg_mdl[inv]["Random"])
            # Helper function to compute rank-biserial effect size for unpaired samples
            function rank_biserial_unpaired(x::Vector{<:Real}, y::Vector{<:Real})
                scores = Float64[]
                for xi in x, yj in y
                    push!(scores, xi > yj ? 1.0 : (xi == yj ? 0.5 : 0.0))
                end
                A = mean(scores)
                return 2A - 1    # in [-1,1]
            end
            
            test_real_random = MannWhitneyUTest(weighted_avg_mdl[inv]["Real"], weighted_avg_mdl[inv]["Random"])
            p_real_random = pvalue(test_real_random)
            r_real_random = rank_biserial_unpaired(weighted_avg_mdl[inv]["Real"], weighted_avg_mdl[inv]["Random"])
            n_real = length(weighted_avg_mdl[inv]["Real"])
            n_random = length(weighted_avg_mdl[inv]["Random"])
            
            println("\n  MannWhitneyUTest test for $inv (Real vs Random):")
            println("    p-value = $(round(p_real_random, sigdigits=4))")
            println("    effect size (rank-biserial) = $(round(r_real_random, digits=3))")
            println("    sample size (Real) = $n_real")
            println("    sample size (Random) = $n_random")
        end
        
        # Save density plot
        savefig(p_inv_density, "mdl_distribution_$(inv)_density.png")
        
        println("\nDensity plot saved as mdl_distribution_$(inv)_density.png")
    end
end




# ============================================================




# ============================================================
# VIOLIN PLOT 3: Average MDL per language grouped by language family
# ============================================================

println("\n============================================================")
println("Creating violin plots by language family...")
println("============================================================")

# Collect average informativity (1/MDL) per language with family information
family_informativity_data = Dict(inv => Dict{String, Vector{Float64}}() for inv in inventories)

for inv in inventories
    println("\nProcessing inventory: $inv for family violin plot")
    
    for (language, lang_data) in all_data[inv]
        if haskey(lang_data, "min_lengths")
            min_lengths = lang_data["min_lengths"]
            
            # Get the language family from pb_languages_formatted.csv
            lang_row = pb_languages[pb_languages[!, :language] .== language, :]
            if nrow(lang_row) == 0
                continue  # Skip if language not found in CSV
            end
            
            family = lang_row[1, :family]
            
            # Calculate average informativity (1/MDL) for this language
            mdl_values_lang = [Float64(mdl) for (feature, mdl) in min_lengths]
            if !isempty(mdl_values_lang)
                informativity_values = 1.0 ./ mdl_values_lang
                avg_informativity = mean(informativity_values)
                
                # Add to family data
                if !haskey(family_informativity_data[inv], family)
                    family_informativity_data[inv][family] = Float64[]
                end
                push!(family_informativity_data[inv][family], avg_informativity)
            end
        end
    end
end

# Filter families to only include those with at least 5 languages
# This makes the plot more readable and statistically meaningful
MIN_LANGUAGES_PER_FAMILY = 5

# First, find families that have >= MIN_LANGUAGES_PER_FAMILY in at least one inventory
all_families = Set{String}()
for inv in inventories
    for (family, informativities) in family_informativity_data[inv]
        if length(informativities) >= MIN_LANGUAGES_PER_FAMILY
            push!(all_families, family)
        end
    end
end

if isempty(all_families)
    println("  No families with >= $MIN_LANGUAGES_PER_FAMILY languages, skipping...")
else
    # Calculate median informativity across all inventories for sorting
    family_medians_combined = Dict{String, Float64}()
    for family in all_families
        all_informativities = Float64[]
        for inv in inventories
            if haskey(family_informativity_data[inv], family)
                append!(all_informativities, family_informativity_data[inv][family])
            end
        end
        if !isempty(all_informativities)
            family_medians_combined[family] = median(all_informativities)
        end
    end
    
    # Sort families by combined median
    sorted_families = sort(collect(all_families), by=f -> family_medians_combined[f])
    
    println("\nCreating combined violin plot with all feature systems")
    println("  Families included: $(length(sorted_families))")
    
    # Prepare data for combined violin plot
    # We'll use x-positions to group families and separate inventories within each family
    all_data_array = Float64[]
    all_x_positions = Float64[]
    all_colors = []
    
    # Create x-tick positions and labels
    x_tick_positions = Float64[]
    x_tick_labels = String[]
    
    spacing_between_families = 4.0  # Space between family groups
    spacing_within_family = 1.0     # Space between inventories within a family
    current_x = 0.0
    
    for (family_idx, family) in enumerate(sorted_families)
        # Center position for this family group
        family_center = current_x + 1.5 * spacing_within_family
        push!(x_tick_positions, family_center)
        push!(x_tick_labels, family)
        
        # Add data for each inventory
        for (inv_idx, inv) in enumerate(inventories)
            x_pos = current_x + (inv_idx - 1) * spacing_within_family
            
            if haskey(family_informativity_data[inv], family) && length(family_informativity_data[inv][family]) > 0
                informativities = family_informativity_data[inv][family]
                for informativity in informativities
                    push!(all_data_array, informativity)
                    push!(all_x_positions, x_pos)
                    push!(all_colors, colors[inv_idx])
                end
            end
        end
        
        current_x += spacing_between_families
    end
    
    # Create the combined violin plot
    p_violin_combined = plot(
        xlabel="Language Family",
        ylabel="Average Informativity per Language",
        title="Average Informativity by Language Family (All Feature Systems)",
        legend=:topright,
        size=(max(1400, length(sorted_families) * 120), 800),
        margin=15Plots.mm,
        bottom_margin=25Plots.mm,
        left_margin=15Plots.mm,
        top_margin=12Plots.mm,
        guidefontsize=14,
        tickfontsize=9,
        titlefontsize=14,
        legendfontsize=12,
        framestyle=:box,
        grid=:y,
        gridalpha=0.3,
        thickness_scaling=1.5,
        xticks=(x_tick_positions, x_tick_labels),
        xrotation=45
    )
    
    # Add violins for each inventory separately so we can control colors
    for (inv_idx, inv) in enumerate(inventories)
        # Collect data for this inventory
        inv_data = Float64[]
        inv_x_positions = Float64[]
        
        current_x = 0.0
        for family in sorted_families
            x_pos = current_x + (inv_idx - 1) * spacing_within_family
            
            if haskey(family_informativity_data[inv], family) && length(family_informativity_data[inv][family]) > 0
                informativities = family_informativity_data[inv][family]
                append!(inv_data, informativities)
                append!(inv_x_positions, fill(x_pos, length(informativities)))
            end
            
            current_x += spacing_between_families
        end
        
        # Add violin for this inventory
        if !isempty(inv_data)
            violin!(p_violin_combined,
                    inv_x_positions,
                    inv_data,
                    fillcolor=colors[inv_idx],
                    fillalpha=0.5,
                    linecolor=colors[inv_idx],
                    linewidth=2,
                    label=inv,
                    side=:both)
        end
    end
    
    # Save the combined plot
    savefig(p_violin_combined, "informativity_by_family_combined.png")
    println("  Combined violin plot saved as informativity_by_family_combined.png")
    
    # Print statistics for each family and inventory
    println("\n  Family statistics:")
    for family in sorted_families
        println("\n    $family:")
        for inv in inventories
            if haskey(family_informativity_data[inv], family) && !isempty(family_informativity_data[inv][family])
                informativities = family_informativity_data[inv][family]
                println("      $inv: n=$(length(informativities)), median=$(round(median(informativities), digits=2)), mean=$(round(mean(informativities), digits=2))")
            end
        end
    end
end
