using Plots
using StatsPlots
using JSON
using Statistics
using StatsBase
using HypothesisTests: MannWhitneyUTest, SignedRankTest, pvalue
using Statistics
# using CairoMakie
using CSV
using DataFrames
using GLM

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

function rank_biserial_paired(x::Vector{<:Real}, y::Vector{<:Real})
    d = x .- y
    pos = sum(d .> 0)
    neg = sum(d .< 0)
    ties = sum(d .== 0)
    n = pos + neg + ties
    A = (pos + 0.5 * ties) / n
    return 2A - 1
end



# ============================================================
# FIRST PLOT: Histogram of all feature informativity values
# ============================================================

# Collect all MDL values for each inventory
mdl_values = Dict(inv => Float64[] for inv in inventories)

for inv in inventories
    for (language, lang_data) in all_data[inv]
        if haskey(lang_data, "min_lengths")
            min_lengths = lang_data["min_lengths"]
            # Collect all MDL values from this language
            for (feature, mdl) in min_lengths
                push!(mdl_values[inv], Float64(mdl))
            end
        end
    end
end

# Create bins that work for all distributions
all_values = vcat(mdl_values["HC"], mdl_values["SPE"], mdl_values["JFH"])
min_val = floor(Int, minimum(all_values))
max_val = ceil(Int, maximum(all_values))
bin_edges = min_val:1:max_val

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

hc_counts = count_frequencies(mdl_values["HC"], bin_edges)
spe_counts = count_frequencies(mdl_values["SPE"], bin_edges)
jfh_counts = count_frequencies(mdl_values["JFH"], bin_edges)

# Use integer x-values (the bin centers are the integer MDL values)
x_values = collect(min_val:(max_val-1))

# Calculate means for each distribution
mean_hc = median(mdl_values["HC"])
mean_spe = median(mdl_values["SPE"])
mean_jfh = median(mdl_values["JFH"])

# Calculate y-axis limit with extra space at top
max_count = maximum([maximum(hc_counts), maximum(spe_counts), maximum(jfh_counts)])
y_limit = max_count * 1.1  # Add 10% extra space at top

# Create grouped bar plot with proper spacing
p = groupedbar(
    x_values,
    hcat(hc_counts, spe_counts, jfh_counts),
    label=["HC" "SPE" "JFH"],
    xlabel="Informativity (Minimal Description Length)",
    ylabel="Feature Count",
    fillcolor=[colors[1] colors[2] colors[3]],
    fillalpha=0.6,  # Make bars semi-transparent
    linecolor=[colors[1] colors[2] colors[3]],  # Bar edges same color
    linewidth=1.5,  # Bar edge thickness
    bar_width=0.5,  # Width of each individual bar
    xticks=x_values,  # Put ticks at integer values
    ylims=(0, y_limit),  # Set y-axis limits with extra space
    legend=:topright,
    size=(1000, 700),
    margin=10Plots.mm,  # Add 10mm margin on all sides
    bottom_margin=12Plots.mm,  # Extra margin at bottom for x-label
    left_margin=12Plots.mm,  # Extra margin at left for y-label
    top_margin=10Plots.mm,  # Extra margin at top
    guidefontsize=14,  # Axis label font size
    tickfontsize=12,  # Tick label font size
    legendfontsize=11,  # Legend font size
    framestyle=:box,  # Draw box around plot
    grid=:y,  # Only horizontal grid lines
    gridalpha=0.3,  # Light grid
    thickness_scaling=1.5  # Make axis lines thicker
)

# Add vertical lines for means
vline!(p, [mean_hc], color=colors[1], linestyle=:dash, linewidth=2, 
       label="HC mean ($(round(mean_hc, digits=2)))")
vline!(p, [mean_spe], color=colors[2], linestyle=:dash, linewidth=2, 
       label="SPE mean ($(round(mean_spe, digits=2)))")
vline!(p, [mean_jfh], color=colors[3], linestyle=:dash, linewidth=2, 
       label="JFH mean ($(round(mean_jfh, digits=2)))")

# Save plot
savefig(p, "feature_informativity_histogram.png")

println("Plot saved as feature_informativity_histogram.png")
println("\nSummary statistics:")
for inv in inventories
    println("\n$inv:")
    println("  Total features: $(length(mdl_values[inv]))")
    println("  Mean MDL: $(round(mean(mdl_values[inv]), digits=2))")
    println("  Median MDL: $(round(median(mdl_values[inv]), digits=2))")
end





# ============================================================
# SECOND PLOT: Average informativity per language
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
min_avg = minimum(all_lang_avgs)
max_avg = maximum(all_lang_avgs)
# Use 100 bins for better resolution
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
    xlabel="Average MDL per Language",
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
    xlabel="Average MDL per Language",
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
# THIRD PLOT SET: Individual inventory plots (like plot_script_random.py)
# With random sampling
# ============================================================

# Configuration
NUM_SAMPLES = 10  # Number of random samples per language (matching Python)

println("\n============================================================")
println("Generating random samples and computing informativity...")
println("============================================================")

# Step 1: Collect all unique phonemes from pb_languages_formatted.csv to create phoneme pool
all_phonemes = Set{String}()
for row in eachrow(pb_languages)
    inventory_str = row[Symbol("core inventory")]
    # Parse the inventory string which is in Python list format: ['a', 'b', ...]
    # Remove brackets and split by comma
    inventory_str = strip(inventory_str, ['[', ']'])
    phonemes = [strip(p, [' ', '\'', '"']) for p in split(inventory_str, ',')]
    for phoneme in phonemes
        if !isempty(phoneme) && phoneme != ""
            push!(all_phonemes, phoneme)
        end
    end
end
phoneme_pool = collect(all_phonemes)
println("Total unique phonemes in pool: $(length(phoneme_pool))")

# Function to read feature inventory (matching Python's readinventory)
function readinventory(filename)
    featdict = Dict()
    allsegments = Set{String}()
    
    filepath = joinpath("feature_sets", "$(filename)_features.txt")
    lines = [strip(line) for line in readlines(filepath)]
    fields = split(lines[1])
    
    for f in fields
        featdict[f] = Dict(
            "name" => f,
            "+" => Set{String}(),
            "-" => Set{String}()
        )
    end
    
    for i in 2:length(lines)
        thisline = lines[i]
        if length(thisline) == 0
            continue
        end
        linefields = split(thisline)
        if length(linefields) != length(fields) + 1
            println("Field length mismatch on line $i")
            exit()
        end
        phoneme = linefields[1]
        push!(allsegments, phoneme)
        for j in 2:length(linefields)
            if linefields[j] == "+" || linefields[j] == "-"
                push!(featdict[fields[j-1]][linefields[j]], phoneme)
            end
        end
    end
    
    return featdict, allsegments
end

# Function to store features for one solution (thread-safe version)
function store_feats(solutions_dict, maxlen_ref, fd, feats, modes)
    length_feats = length(feats)
    if !haskey(solutions_dict, length_feats)
        solutions_dict[length_feats] = []
    end
    thissol = []
    for (idx, feat) in enumerate(feats)
        push!(thissol, modes[idx] * fd[feat]["name"])
    end
    push!(solutions_dict[length_feats], "[" * join(thissol, ",") * "]")
end

# Recursive function to check natural classes (thread-safe version)
function reccheck(solutions_dict, maxlen_ref, fd, basefeats, basemodes, feats, modes, correct, baseindex, current_base)
    if length(feats) > maxlen_ref[]
        return
    end
    
    # Check if current combination is a solution
    if current_base == correct
        store_feats(solutions_dict, maxlen_ref, fd, feats, modes)
        if length(feats) < maxlen_ref[]
            maxlen_ref[] = length(feats)
        end
    end
    
    numelem = length(basefeats)
    for i in baseindex:numelem
        if basefeats[i] ∉ feats
            new_base = intersect(current_base, fd[basefeats[i]][basemodes[i]])
            if !isempty(new_base)
                reccheck(solutions_dict, maxlen_ref, fd, basefeats, basemodes, vcat(feats, [basefeats[i]]), vcat(modes, [basemodes[i]]), correct, i + 1, new_base)
            end
        end
    end
end

# Process phoneme inventory and return natural classes per phoneme (thread-safe version)
function process_phoneme_inventory(allsegments, fd, features)
    natural_classes_perphoneme = Dict()
    
    for phoneme in allsegments
        testset = Set([phoneme])
        base = copy(allsegments)
        feats = String[]
        modes = String[]
        
        # Find all features that describe this phoneme
        for feat in features
            if testset ⊆ fd[feat]["+"]
                base = intersect(base, fd[feat]["+"])
                push!(feats, feat)
                push!(modes, "+")
            elseif testset ⊆ fd[feat]["-"]
                base = intersect(base, fd[feat]["-"])
                push!(feats, feat)
                push!(modes, "-")
            end
        end
        
        # Use local variables instead of global
        local_solutions = Dict()
        local_maxlen = Ref(length(feats))
        
        if base == testset
            reccheck(local_solutions, local_maxlen, fd, feats, modes, String[], String[], base, 1, allsegments)
            
            if !haskey(natural_classes_perphoneme, phoneme)
                natural_classes_perphoneme[phoneme] = []
            end
            
            for s in values(local_solutions)
                append!(natural_classes_perphoneme[phoneme], s)
            end
        end
    end
    
    return natural_classes_perphoneme
end

# Get descriptive information for natural classes
function get_general_info_natural_classes(natural_classes, keys)
    min_lengths = Dict()
    min_lengths_phonemes = Dict()
    avg_lengths = Dict(key => [0, 0] for key in keys)
    min_descriptions = Dict()
    count_phoneme = Dict()
    count_lengths = Dict()
    
    for (phoneme, sublists) in natural_classes
        parsed_sublists = []
        for sublist in sublists
            parsed = split(strip(sublist, ['[', ']']), ',')
            push!(parsed_sublists, parsed)
            
            # Process features in this sublist
            for value in parsed
                value = strip(value, ['+', '-'])
                
                # Update min_lengths
                if haskey(min_lengths, value)
                    min_lengths[value] = min(min_lengths[value], length(parsed))
                else
                    min_lengths[value] = length(parsed)
                end
                
                # Update avg_lengths
                if haskey(avg_lengths, value)
                    avg_lengths[value][1] += length(parsed)
                    avg_lengths[value][2] += 1
                end
            end
            
            # Update min_lengths_phonemes
            if haskey(min_lengths_phonemes, phoneme)
                min_lengths_phonemes[phoneme] = min(min_lengths_phonemes[phoneme], length(parsed))
            else
                min_lengths_phonemes[phoneme] = length(parsed)
            end
        end
        
        # Get minimal descriptions for this phoneme
        min_len = min_lengths_phonemes[phoneme]
        min_descriptions[phoneme] = [parsed for parsed in parsed_sublists if length(parsed) == min_len]
        
        # Count features in minimal descriptions
        for sublist in min_descriptions[phoneme]
            for value in sublist
                value = strip(value, ['+', '-'])
                count_phoneme[value] = get(count_phoneme, value, 0) + 1
            end
            
            sublist_len = length(sublist)
            count_lengths[sublist_len] = get(count_lengths, sublist_len, 0) + 1
        end
    end
    
    avg_lengths = Dict(k => v[1] / v[2] for (k, v) in avg_lengths if v[2] != 0)
    
    return min_lengths, min_descriptions, count_phoneme, avg_lengths, count_lengths
end

# Function to compute weighted average MDL (matching Python's compute_avg_mdl)
function compute_weighted_avg_mdl(allsegments, min_lengths, min_descriptions)
    total_avg_length = 0.0
    feature_count = 0
    
    for phoneme in allsegments
        if haskey(min_descriptions, phoneme)
            feature_descriptions = min_descriptions[phoneme]
            
            # Get unique features: flatten all sublists and strip +/-
            unique_features = Set{String}()
            for sublist in feature_descriptions
                for item in sublist
                    feature_name = strip(item, ['+', '-'])
                    push!(unique_features, feature_name)
                end
            end
            
            # Add MDL for each unique feature
            for feature in unique_features
                if haskey(min_lengths, feature)
                    total_avg_length += min_lengths[feature]
                    feature_count += 1
                end
            end
        end
    end
    
    if feature_count > 0
        return total_avg_length / feature_count
    end
    return nothing
end

# Function to compute simple average MDL (same as Plot 2 approach)
# Takes all features in min_lengths and computes their mean
function compute_simple_avg_mdl(min_lengths)
    mdl_values = [Float64(mdl) for (feature, mdl) in min_lengths]
    if !isempty(mdl_values)
        return mean(mdl_values)
    end
    return nothing
end

# Function to process phoneme inventory and get natural classes
# (Simplified version - we'll use the pre-computed data from JSON)
function get_weighted_mdl_for_sample(sampled_phonemes, inv, all_data_inv)
    # For random samples, we need to compute from scratch
    # But since we don't have the feature dictionary loaded, 
    # we'll use a proxy: sample from existing languages and use their MDL
    # This matches the Python approach where random samples are processed
    
    # Find a language with similar inventory size to use as proxy
    target_size = length(sampled_phonemes)
    
    for (language, lang_data) in all_data_inv
        if haskey(lang_data, "min_lengths") && haskey(lang_data, "min_descriptions")
            inventory_size = length(keys(lang_data["min_descriptions"]))
            if inventory_size == target_size
                # Use this language's data structure
                min_lengths = lang_data["min_lengths"]
                min_descriptions = lang_data["min_descriptions"]
                
                return compute_weighted_avg_mdl(sampled_phonemes, min_lengths, min_descriptions)
            end
        end
    end
    
    return nothing
end

# Function to compute random sample MDL (for parallel execution)
function compute_random_sample_mdl(inventory_size, phoneme_pool, featdict, features)
    # Randomly sample phonemes from the pool
    sampled_phonemes = Set(StatsBase.sample(phoneme_pool, inventory_size, replace=false))
    
    # Compute natural classes for the sampled phonemes
    natural_classes = process_phoneme_inventory(sampled_phonemes, featdict, features)
    
    # Get informativity information
    sample_min_lengths, _, _, _, _ = 
        get_general_info_natural_classes(natural_classes, features)
    
    # Compute simple mean of MDL values for this random sample
    if !isempty(sample_min_lengths)
        mdl_values_sample = [Float64(mdl) for (feature, mdl) in sample_min_lengths]
        return mean(mdl_values_sample)
    end
    return nothing
end

# Collect average MDL per language for each inventory (using simple mean like Plot 2)
weighted_avg_mdl = Dict(inv => Dict("Real" => Float64[], "Random" => Float64[]) for inv in inventories)

println("\nUsing $(Threads.nthreads()) threads for parallel processing")

# Process each inventory
for inv in inventories
    println("\nProcessing inventory: $inv")
    
    # Read the feature dictionary for this inventory
    featdict, all_segments_inv = readinventory(inv)
    features = collect(keys(featdict))
    
    lang_count = 0
    
    for (language, lang_data) in all_data[inv]
        min_lengths = lang_data["min_lengths"]
        min_descriptions = lang_data["min_descriptions"]
        
        # Get the actual phoneme inventory from pb_languages_formatted.csv
        lang_row = pb_languages[pb_languages[!, :language] .== language, :]
        if nrow(lang_row) == 0
            continue  # Skip if language not found in CSV
        end
        
        inventory_str = lang_row[1, Symbol("core inventory")]
        # Parse the inventory string which is in Python list format: ['a', 'b', ...]
        inventory_str = strip(inventory_str, ['[', ']'])
        allsegments_list = [strip(p, [' ', '\'', '"']) for p in split(inventory_str, ',')]
        allsegments = Set([p for p in allsegments_list if !isempty(p) && p != ""])
        inventory_size = length(allsegments)
        
        # Compute real average using simple mean (same as Plot 2)
        mdl_values_lang = [Float64(mdl) for (feature, mdl) in min_lengths]
        if !isempty(mdl_values_lang)
            real_avg_mdl = mean(mdl_values_lang)
            push!(weighted_avg_mdl[inv]["Real"], real_avg_mdl)
            lang_count += 1
            
            # Generate NUM_SAMPLES random inventories and compute their average MDL in parallel
            sample_avg_mdls = Vector{Union{Float64, Nothing}}(undef, NUM_SAMPLES)
            
            Threads.@threads for sample_num in 1:NUM_SAMPLES
                sample_avg_mdls[sample_num] = compute_random_sample_mdl(
                    inventory_size, phoneme_pool, featdict, features
                )
            end
            
            # Filter out nothing values and compute mean
            valid_samples = [x for x in sample_avg_mdls if x !== nothing]
            if !isempty(valid_samples)
                mean_random_mdl = mean(valid_samples)
                push!(weighted_avg_mdl[inv]["Random"], mean_random_mdl)
            end
            
            if lang_count % 10 == 0
                println("  Processed $lang_count languages...")
            end
        end
    end
    
    println("  Processed $lang_count languages")
    println("  Real samples: $(length(weighted_avg_mdl[inv]["Real"]))")
    println("  Random samples: $(length(weighted_avg_mdl[inv]["Random"]))")
end


weighted_avg_mdl
# Save weighted_avg_mdl to JSON file
output_filename = "weighted_avg_mdl_data.json"
open(output_filename, "w") do io
    JSON.print(io, weighted_avg_mdl, 4)
end
println("\nweighted_avg_mdl data saved to $output_filename")


# Load weighted_avg_mdl from JSON file
weighted_avg_mdl_raw = JSON.parsefile("weighted_avg_mdl_data.json")

# Convert to proper types (JSON loads as Vector{Any}, need Vector{Float64})
weighted_avg_mdl = Dict(inv => Dict("Real" => Float64[], "Random" => Float64[]) for inv in inventories)
for inv in inventories
    weighted_avg_mdl[inv]["Real"] = Float64.(weighted_avg_mdl_raw[inv]["Real"])
    weighted_avg_mdl[inv]["Random"] = Float64.(weighted_avg_mdl_raw[inv]["Random"])
end




for inv in inventories
    if !isempty(weighted_avg_mdl[inv]["Real"]) || !isempty(weighted_avg_mdl[inv]["Random"])
        # Combine all values to determine bin range
        all_values_inv = vcat(weighted_avg_mdl[inv]["Real"], weighted_avg_mdl[inv]["Random"])
        
        if !isempty(all_values_inv)
            min_val = minimum(all_values_inv)
            max_val = maximum(all_values_inv)
            bin_width = (max_val - min_val) / 50
            bin_edges = min_val:bin_width:(max_val + bin_width)
            
            # Count frequencies for both Real and Random
            real_counts = count_frequencies(weighted_avg_mdl[inv]["Real"], bin_edges)
            random_counts = count_frequencies(weighted_avg_mdl[inv]["Random"], bin_edges)
            
            x_vals = [bin_edges[i] for i in 1:(length(bin_edges) - 1)]
            
            # Calculate statistics
            real_median = !isempty(weighted_avg_mdl[inv]["Real"]) ? median(weighted_avg_mdl[inv]["Real"]) : 0.0
            random_median = !isempty(weighted_avg_mdl[inv]["Random"]) ? median(weighted_avg_mdl[inv]["Random"]) : 0.0
            
            # Determine color for this inventory
            color_idx = findfirst(==(inv), inventories)
            color_real = colors[color_idx]
            color_random = :gray  # Use gray for random samples
            
            # Calculate y-axis limit
            max_count = maximum([maximum(real_counts), maximum(random_counts)])
            y_lim = max_count * 1.1
            
            # Create plot with Real data
            p_inv = plot(
                x_vals,
                real_counts,
                seriestype=:bar,
                label="Real",
                xlabel="Average Minimal Description Length",
                ylabel="Language Count",
                title="Feature system: $inv",
                fillcolor=color_real,
                fillalpha=0.6,
                linecolor=color_real,
                linewidth=1.5,
                bar_width=bin_width,
                ylims=(0, y_lim),
                legend=:topright,
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
            
            # Add Random data as overlapping bars
            if !isempty(weighted_avg_mdl[inv]["Random"])
                bar!(p_inv, x_vals, random_counts,
                     label="Random",
                     fillcolor=color_random,
                     fillalpha=0.5,
                     linecolor=color_random,
                     linewidth=1.5,
                     bar_width=bin_width)
            end
            
            # Add vertical lines for medians
            if !isempty(weighted_avg_mdl[inv]["Real"])
                vline!(p_inv, [real_median], 
                       color=color_real, 
                       linestyle=:dash, 
                       linewidth=2, 
                       label="Real Median: $(round(real_median, digits=2))")
            end
            
            if !isempty(weighted_avg_mdl[inv]["Random"])
                vline!(p_inv, [random_median], 
                       color=color_random, 
                       linestyle=:dash, 
                       linewidth=2, 
                       label="Random Median: $(round(random_median, digits=2))")
            end
            
            # Save plot
            savefig(p_inv, "mdl_distribution_$(inv).png")
            
            println("\nPlot saved as mdl_distribution_$(inv).png")
            println("Statistics for $inv:")
            println("  Real - Total languages: $(length(weighted_avg_mdl[inv]["Real"]))")
            if !isempty(weighted_avg_mdl[inv]["Real"])
                println("  Real - Mean: $(round(mean(weighted_avg_mdl[inv]["Real"]), digits=2))")
                println("  Real - Median: $(round(real_median, digits=2))")
            end
            if !isempty(weighted_avg_mdl[inv]["Random"])
                println("  Random - Total samples: $(length(weighted_avg_mdl[inv]["Random"]))")
                println("  Random - Mean: $(round(mean(weighted_avg_mdl[inv]["Random"]), digits=2))")
                println("  Random - Median: $(round(random_median, digits=2))")
            end
        end
    end
end

println("\n✓ All plots created successfully!")

# ============================================================
# PLOT 2 DENSITY VERSION: Average informativity per language (density plot)
# ============================================================

println("\n============================================================")
println("Creating density plot version of Plot 2...")
println("============================================================")

# Create density plot for language averages
p2_density = plot(
    xlabel="Average MDL per Language",
    ylabel="Density",
    legend=:topright,
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
    legend=:topright
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
    legend=:topright
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
    xlabel="Average MDL per Language",
    title="JFH",
    size=(180, 200),
    titlefontsize=10,
    guidefontsize=9,
    tickfontsize=8,
    legendfontsize=8,
    framestyle=:box,
    grid=:y,
    gridalpha=0.3,
    legend=:topright
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
# PLOT 2 VIOLIN VERSION: Average informativity per language (violin plot)
# ============================================================

println("\n============================================================")
println("Creating violin plot version of Plot 2...")
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

p2_violin.topspinevisible = false
p2_violin.rightspinevisible = false

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

# Find the maximum y-value across all distributions to position brackets above
max_y = maximum([maximum(avg_mdl_per_language[inv]) for inv in inventories]) - 0.2
bracket_spacing = 0.15  # Vertical spacing between bracket levels
tick_height = 0.05  # Height of vertical tick marks at edges

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
annotate!(p2_violin, 1.5, y_bracket_1 + 0.06, text(stars_hc_spe, 14, :center))

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
annotate!(p2_violin, 2.5, y_bracket_2 + 0.06, text(stars_spe_jfh, 14, :center))

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
annotate!(p2_violin, 2, y_bracket_3 + 0.06, text(stars_hc_jfh, 14, :center))

# Adjust y-axis limits to accommodate brackets
ylims!(p2_violin, (1.5, max_y + 4 * bracket_spacing + 0.1))

# Save violin plot
savefig(p2_violin, "language_average_informativity_violin.png")

println("\nLanguage average violin plot saved as language_average_informativity_violin.png")



# ============================================================
# PLOT 3 DENSITY VERSION: Individual inventory density plots with Real and Random
# ============================================================

println("\n============================================================")
println("Creating density plot versions of Plot 3...")
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
            legend=:topright,
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

println("\n✓ All density plots created successfully!")





# ============================================================
# VIOLIN PLOT: Average MDL per language grouped by language family
# ============================================================

println("\n============================================================")
println("Creating violin plots by language family...")
println("============================================================")

# Collect average MDL per language with family information
family_mdl_data = Dict(inv => Dict{String, Vector{Float64}}() for inv in inventories)

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
            
            # Calculate average MDL for this language (simple mean)
            mdl_values_lang = [Float64(mdl) for (feature, mdl) in min_lengths]
            if !isempty(mdl_values_lang)
                avg_mdl = mean(mdl_values_lang)
                
                # Add to family data
                if !haskey(family_mdl_data[inv], family)
                    family_mdl_data[inv][family] = Float64[]
                end
                push!(family_mdl_data[inv][family], avg_mdl)
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
    for (family, mdls) in family_mdl_data[inv]
        if length(mdls) >= MIN_LANGUAGES_PER_FAMILY
            push!(all_families, family)
        end
    end
end

if isempty(all_families)
    println("  No families with >= $MIN_LANGUAGES_PER_FAMILY languages, skipping...")
else
    # Calculate median MDL across all inventories for sorting
    family_medians_combined = Dict{String, Float64}()
    for family in all_families
        all_mdls = Float64[]
        for inv in inventories
            if haskey(family_mdl_data[inv], family)
                append!(all_mdls, family_mdl_data[inv][family])
            end
        end
        if !isempty(all_mdls)
            family_medians_combined[family] = median(all_mdls)
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
            
            if haskey(family_mdl_data[inv], family) && length(family_mdl_data[inv][family]) > 0
                mdls = family_mdl_data[inv][family]
                for mdl in mdls
                    push!(all_data_array, mdl)
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
        ylabel="Average MDL per Language",
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
            
            if haskey(family_mdl_data[inv], family) && length(family_mdl_data[inv][family]) > 0
                mdls = family_mdl_data[inv][family]
                append!(inv_data, mdls)
                append!(inv_x_positions, fill(x_pos, length(mdls)))
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
    savefig(p_violin_combined, "mdl_by_family_combined.png")
    println("  Combined violin plot saved as mdl_by_family_combined.png")
    
    # Print statistics for each family and inventory
    println("\n  Family statistics:")
    for family in sorted_families
        println("\n    $family:")
        for inv in inventories
            if haskey(family_mdl_data[inv], family) && !isempty(family_mdl_data[inv][family])
                mdls = family_mdl_data[inv][family]
                println("      $inv: n=$(length(mdls)), median=$(round(median(mdls), digits=2)), mean=$(round(mean(mdls), digits=2))")
            end
        end
    end
end

println("\n✓ All violin plots created successfully!")

# ============================================================
# INDIVIDUAL LANGUAGE MDL HISTOGRAM: Feature MDL distribution per language
# ============================================================

println("\n============================================================")
println("Creating individual language MDL histograms...")
println("============================================================")

# Function to create histogram for a specific language
function create_language_mdl_histogram(language_name, all_data, inventories, colors)
    # Collect MDL values for this language from all three feature systems
    language_mdl_data = Dict(inv => Float64[] for inv in inventories)
    
    found = false
    for inv in inventories
        if haskey(all_data[inv], language_name)
            lang_data = all_data[inv][language_name]
            if haskey(lang_data, "min_lengths")
                min_lengths = lang_data["min_lengths"]
                for (feature, mdl) in min_lengths
                    push!(language_mdl_data[inv], Float64(mdl))
                end
                found = true
            end
        end
    end
    
    if !found
        println("  Language '$language_name' not found in data")
        return nothing
    end
    
    # Find the range of MDL values across all systems
    all_mdl_values = vcat([language_mdl_data[inv] for inv in inventories]...)
    if isempty(all_mdl_values)
        println("  No MDL data found for '$language_name'")
        return nothing
    end
    
    # Always use MDL values 1-4 for consistency across all languages
    mdl_bins = 1:4
    
    # Count frequencies and convert to percentages for each inventory
    mdl_percentages = Dict(inv => zeros(Float64, length(mdl_bins)) for inv in inventories)
    
    for inv in inventories
        total_features = length(language_mdl_data[inv])
        if total_features > 0
            for mdl_val in language_mdl_data[inv]
                bin_idx = findfirst(x -> x == floor(Int, mdl_val), mdl_bins)
                if bin_idx !== nothing
                    mdl_percentages[inv][bin_idx] += 1
                end
            end
            # Convert counts to percentages
            mdl_percentages[inv] .= (mdl_percentages[inv] ./ total_features) .* 100
        end
    end
    
    # Create grouped bar plot
    x_positions = collect(mdl_bins)
    percentages_matrix = hcat([mdl_percentages[inv] for inv in inventories]...)
    
    p_lang = groupedbar(
        x_positions,
        percentages_matrix,
        label=permutedims(inventories),  # Transpose for correct legend labels
        xlabel="Minimal Description Length (MDL)",
        ylabel="Percentage of Features (%)",
        title="Language: $language_name",
        fillcolor=permutedims(colors),
        fillalpha=0.7,
        linecolor=permutedims(colors),
        linewidth=1.5,
        bar_width=0.6,
        xticks=x_positions,
        ylims=(0, 100),  # Percentage scale from 0 to 100
        legend=:topright,
        size=(1000, 600),
        margin=10Plots.mm,
        bottom_margin=12Plots.mm,
        left_margin=12Plots.mm,
        top_margin=12Plots.mm,
        guidefontsize=10,
        tickfontsize=10,
        legendfontsize=10,
        framestyle=:box,
        grid=:y,
        gridalpha=0.3,
        thickness_scaling=1.5
    )
    
    # Save plot
    safe_name = replace(language_name, " " => "_", "/" => "-", "(" => "", ")" => "")
    savefig(p_lang, "plot3/language_mdl_$(safe_name).png")
    
    println("  Plot saved as language_mdl_$(safe_name).png")
    
    # Print statistics
    println("  Statistics for $language_name:")
    for inv in inventories
        if !isempty(language_mdl_data[inv])
            println("    $inv: $(length(language_mdl_data[inv])) features, " *
                   "median=$(round(median(language_mdl_data[inv]), digits=2)), " *
                   "mean=$(round(mean(language_mdl_data[inv]), digits=2))")
        end
    end
    
    return p_lang
end


println("\nCreating histograms for example languages...")
for lang in pb_languages[:, "language"]
    create_language_mdl_histogram(lang, all_data, inventories, colors)
end

println("\n✓ Language-specific MDL histograms created successfully!")
println("\nTo create a histogram for a specific language, use:")
println("  create_language_mdl_histogram(\"LanguageName\", all_data, inventories, colors)")

































# --- Diagnostics: build per-language maps and check pairing between HC and SPE ---
println("\nRunning pairing diagnostics for HC vs SPE...")

# Build maps language -> mean MDL for each inventory (safer pairing)
real_by_lang = Dict(inv => Dict{String, Float64}() for inv in inventories)
for inv in inventories
    for (language, lang_data) in all_data[inv]
        if haskey(lang_data, "min_lengths")
            vals = [Float64(mdl) for (f, mdl) in lang_data["min_lengths"]]
            if !isempty(vals)
                real_by_lang[inv][language] = mean(vals)
            end
        end
    end
end


# Compute intersection of languages for HC and SPE
common_hc_spe = sort(collect(intersect(keys(real_by_lang["HC"]), keys(real_by_lang["SPE"]))))
println("  Common HC<->SPE languages: ", length(common_hc_spe))

if !isempty(common_hc_spe)
    hc_vals = [real_by_lang["HC"][l] for l in common_hc_spe]
    spe_vals = [real_by_lang["SPE"][l] for l in common_hc_spe]
    diffs_hc_minus_spe = hc_vals .- spe_vals

    println("  median(HC) = ", median(hc_vals), "  median(SPE) = ", median(spe_vals))
    println("  median(diff HC-SPE) = ", median(diffs_hc_minus_spe), "  mean(diff) = ", mean(diffs_hc_minus_spe))
    println("  counts: pos=", sum(diffs_hc_minus_spe .> 0), ", neg=", sum(diffs_hc_minus_spe .< 0), ", ties=", sum(diffs_hc_minus_spe .== 0))

    # Paired Signed Rank test and paired rank-biserial
    sr_test = SignedRankTest(hc_vals, spe_vals)
    p_sr = pvalue(sr_test)

    r_pb = rank_biserial_paired(hc_vals, spe_vals)
    println("  SignedRankTest p = ", round(p_sr, sigdigits=6))
    println("  paired rank-biserial = ", round(r_pb, digits=4))

    # Inspect signed rank sums (where the statistic comes from)
    mask = diffs_hc_minus_spe .!= 0
    absd = abs.(diffs_hc_minus_spe[mask])
    signs = sign.(diffs_hc_minus_spe[mask])

    # Compute average ranks with tie handling (replacement for StatsBase.ranks)
    function average_ranks(vals::Vector{<:Real})
        n = length(vals)
        if n == 0
            return Float64[]
        end
        idx = sortperm(vals)
        ranks_local = zeros(Float64, n)
        i = 1
        while i <= n
            j = i
            while j + 1 <= n && vals[idx[j+1]] == vals[idx[i]]
                j += 1
            end
            avg = (i + j) / 2.0
            for k in i:j
                ranks_local[idx[k]] = avg
            end
            i = j + 1
        end
        return ranks_local
    end

    ranks = average_ranks(absd)
    Wpos = sum(ranks[signs .> 0])
    Wneg = sum(ranks[signs .< 0])
    println("  W+ = ", round(Wpos, digits=2), "  W- = ", round(Wneg, digits=2))

    # Save histogram of paired differences for quick visual check
    using Plots
    p_diff = histogram(diffs_hc_minus_spe, bins=45, xlabel="HC - SPE", ylabel="Count",
                       title="Paired differences (HC - SPE)", legend=false)
    savefig(p_diff, "diagnostic_diff_HC_minus_SPE.png")
    println("  Saved paired-differences histogram as diagnostic_diff_HC_minus_SPE.png")
else
    println("  No common languages between HC and SPE found for diagnostics.")
end


# ------------------------------------------------------------

# ============================================================
# REGRESSION ANALYSIS: Predict average MDL per language
# Predictors:
#  1) Number of phonemes in that language
#  2) Number of features in the feature system (HC, SPE, JFH)
#  3) Per-feature phoneme coverage: mean, median, std of # phonemes described by each feature
# Models: Linear regression (baseline) + interactions
# ============================================================

println("\n============================================================")
println("Running regression analysis to predict average MDL per language...")
println("============================================================")

# Try to load GLM; if not available, print installation instructions
# Build dataset: one row per (inventory, language)
reg_rows = []

for inv in inventories
    println("\nProcessing inventory $inv for regression...")
    
    # Read feature dictionary
    featdict, _ = readinventory(inv)
    num_features = length(keys(featdict))
    all_feature_names = collect(keys(featdict))
    
    # For each language in this inventory
    for (language, lang_data) in all_data[inv]
        
        # Outcome: average MDL for this language (same calculation as Plot 2)
        mdl_values_lang = [Float64(mdl) for (feature, mdl) in lang_data["min_lengths"]]
        if isempty(mdl_values_lang)
            continue
        end
        avg_mdl = mean(mdl_values_lang)
        
        # Get phonemes in this language
        lang_row = pb_languages[pb_languages[!, :language] .== language, :]
        if nrow(lang_row) == 0
            continue  # Skip if language not found in CSV
        end
        
        inventory_str = String(lang_row[1, Symbol("core inventory")])
        inventory_str = strip(inventory_str, ['[', ']'])
        phonemes = [strip(p, [' ', '\'', '"']) for p in split(inventory_str, ',')]
        phonemes = [p for p in phonemes if !isempty(p) && p != ""]
        phoneme_count = length(phonemes)
        
        # For this language: count how many phonemes each feature describes (with + sign)
        # A feature "describes" a phoneme if that phoneme is in the feature's "+" set
        feature_phoneme_counts = Int[]
        
        for feat_name in all_feature_names
            plus_set = featdict[feat_name]["+"]
            # Count how many phonemes in this language are described by this feature (have +)
            count_described = 0
            for phoneme in phonemes
                if phoneme in plus_set
                    count_described += 1
                end
            end
            push!(feature_phoneme_counts, count_described)
        end
        
        # Summary statistics of feature usage for this language
        feat_ph_mean = isempty(feature_phoneme_counts) ? 0.0 : mean(feature_phoneme_counts)
        feat_ph_median = isempty(feature_phoneme_counts) ? 0.0 : median(feature_phoneme_counts)
        feat_ph_max = isempty(feature_phoneme_counts) ? 0.0 : maximum(feature_phoneme_counts)
        feat_ph_min = isempty(feature_phoneme_counts) ? 0.0 : minimum(feature_phoneme_counts)
        feat_ph_std = (isempty(feature_phoneme_counts) || length(feature_phoneme_counts) < 2) ? 0.0 : std(feature_phoneme_counts)

        # For each phoneme: count how many features have "+" for that phoneme
        # This represents the description length of each phoneme
        phoneme_description_lengths = Int[]
        
        for phoneme in phonemes
            count_plus_features = 0
            for feat_name in all_feature_names
                plus_set = featdict[feat_name]["+"]
                if phoneme in plus_set
                    count_plus_features += 1
                end
            end
            push!(phoneme_description_lengths, count_plus_features)
        end
        
        # Summary statistics of phoneme description lengths
        phoneme_desc_mean = isempty(phoneme_description_lengths) ? 0.0 : mean(phoneme_description_lengths)
        phoneme_desc_median = isempty(phoneme_description_lengths) ? 0.0 : median(phoneme_description_lengths)
        phoneme_desc_max = isempty(phoneme_description_lengths) ? 0.0 : maximum(phoneme_description_lengths)
        phoneme_desc_min = isempty(phoneme_description_lengths) ? 0.0 : minimum(phoneme_description_lengths)
        phoneme_desc_std = (isempty(phoneme_description_lengths) || length(phoneme_description_lengths) < 2) ? 0.0 : std(phoneme_description_lengths)
        
        # Store row
        push!(reg_rows, (
            inventory = inv,
            language = language,
            avg_mdl = avg_mdl,
            phoneme_count = phoneme_count,
            num_features = num_features,
            feat_phoneme_mean = feat_ph_mean,
            feat_phoneme_median = feat_ph_median,
            feat_phoneme_std = feat_ph_std,
            feat_phoneme_max = feat_ph_max,
            phoneme_desc_mean = phoneme_desc_mean,
            phoneme_desc_median = phoneme_desc_median,
            phoneme_desc_std = phoneme_desc_std,
            phoneme_desc_min = phoneme_desc_min,
            phoneme_desc_max = phoneme_desc_max

        ))
    end
end

df_reg = DataFrame(reg_rows)

println("\nCollected $(nrow(df_reg)) observations (language × inventory combinations)")
println("Summary of raw predictors:")
println("  phoneme_count: min=$(minimum(df_reg.phoneme_count)), max=$(maximum(df_reg.phoneme_count)), mean=$(round(mean(df_reg.phoneme_count), digits=1))")
println("  num_features: min=$(minimum(df_reg.num_features)), max=$(maximum(df_reg.num_features)), mean=$(round(mean(df_reg.num_features), digits=1))")
println("  feat_phoneme_mean: min=$(round(minimum(df_reg.feat_phoneme_mean), digits=1)), max=$(round(maximum(df_reg.feat_phoneme_mean), digits=1)), mean=$(round(mean(df_reg.feat_phoneme_mean), digits=1))")

if nrow(df_reg) < 10
    println("\nNot enough observations ($(nrow(df_reg))) to fit reliable regression models.")
else
    # Normalize predictors (z-score standardization)
    println("\nNormalizing predictors (z-score standardization)...")
    
    function zscore_safe(x)
        μ = mean(x)
        σ = std(x)
        if σ == 0.0 || isnan(σ)
            # If no variation, return zeros
            return zeros(Float64, length(x))
        else
            return (x .- μ) ./ σ
        end
    end
    
    df_reg.phoneme_count_z = zscore_safe(df_reg.phoneme_count)
    df_reg.num_features_z = zscore_safe(df_reg.num_features)
    df_reg.feat_phoneme_mean_z = zscore_safe(df_reg.feat_phoneme_mean)
    df_reg.feat_phoneme_median_z = zscore_safe(df_reg.feat_phoneme_median)
    df_reg.feat_phoneme_std_z = zscore_safe(df_reg.feat_phoneme_std)
    df_reg.feat_phoneme_max_z = zscore_safe(df_reg.feat_phoneme_max)
    df_reg.phoneme_desc_mean_z = zscore_safe(df_reg.phoneme_desc_mean)
    df_reg.phoneme_desc_median_z = zscore_safe(df_reg.phoneme_desc_median)
    df_reg.phoneme_desc_std_z = zscore_safe(df_reg.phoneme_desc_std)
    df_reg.phoneme_desc_max_z = zscore_safe(df_reg.phoneme_desc_max)
    df_reg.phoneme_desc_min_z = zscore_safe(df_reg.phoneme_desc_min)

    println("Predictors normalized (mean=0, std=1)")
    
    # Model 1: Linear regression with all predictors (no interactions)
    println("\n" * "="^60)
    println("MODEL 1: Linear regression (normalized predictors, no interactions)")
    println("="^60)

    formula1 = @formula(avg_mdl ~ phoneme_count_z + num_features_z + feat_phoneme_mean_z + feat_phoneme_median_z + feat_phoneme_std_z + feat_phoneme_max_z + phoneme_desc_mean_z + phoneme_desc_median_z + phoneme_desc_std_z + phoneme_desc_max_z + phoneme_desc_min_z)
    lm1 = lm(formula1, df_reg)
    
    println("\nCoefficient table:")
    display(coeftable(lm1))
    
    r2_1 = r2(lm1)
    adj_r2_1 = adjr2(lm1)
    println("\nR² = $(round(r2_1, digits=4))")
    println("Adjusted R² = $(round(adj_r2_1, digits=4))")
    
    println("\nInterpretation: Each coefficient represents the change in avg_mdl")
    println("(original units) for a 1 SD increase in the predictor.")
    
    # Model 2: Add interaction terms
    println("\n" * "="^60)
    println("MODEL 2: Linear regression with interactions")
    println("="^60)
    println("Interactions: phoneme_count × num_features, phoneme_count × feat_phoneme_mean")

    formula2 = @formula(avg_mdl ~ phoneme_count_z * num_features_z + phoneme_count_z * feat_phoneme_mean_z + feat_phoneme_median_z + feat_phoneme_std_z + feat_phoneme_max_z + phoneme_desc_mean_z + phoneme_desc_median_z + phoneme_desc_std_z + phoneme_desc_max_z + phoneme_desc_min_z)
    lm2 = lm(formula2, df_reg)
    
    println("\nCoefficient table:")
    display(coeftable(lm2))
    
    r2_2 = r2(lm2)
    adj_r2_2 = adjr2(lm2)
    println("\nR² = $(round(r2_2, digits=4))")
    println("Adjusted R² = $(round(adj_r2_2, digits=4))")
    
    # Compare models
    println("\n" * "="^60)
    println("MODEL COMPARISON")
    println("="^60)
    println("Model 1 (no interactions):")
    println("  R² = $(round(r2_1, digits=4)), Adjusted R² = $(round(adj_r2_1, digits=4))")
    println("\nModel 2 (with interactions):")
    println("  R² = $(round(r2_2, digits=4)), Adjusted R² = $(round(adj_r2_2, digits=4))")
    
    if adj_r2_2 > adj_r2_1
        println("\nModel 2 has better adjusted R² (interactions improve fit)")
    elseif adj_r2_2 < adj_r2_1
        println("\nModel 1 has better adjusted R² (interactions don't improve fit enough)")
    else
        println("\nModels have similar adjusted R²")
    end
    
    println("\n✓ Regression analysis completed successfully!")
        
end
