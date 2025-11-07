using Plots
using StatsPlots
using JSON
using Statistics
using StatsBase
using HypothesisTests: MannWhitneyUTest, SignedRankTest, pvalue
using Statistics
using CSV
using DataFrames

# Define inventories
inventories = ["HC", "SPE", "JFH"]
colors = [:red, :green, :blue]

# Load all data first
all_data = Dict()
for inv in inventories
    filename = "data_all_languages_$(inv)_features.json"
    all_data[inv] = JSON.parsefile(filename)
end

# Configuration
NUM_SAMPLES = 1  # Number of random samples per language (matching Python)

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

println("\n✓ All plots created successfully!")

