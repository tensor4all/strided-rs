### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ 323269d2-f801-11f0-8d95-4519f700d1da
begin
	@assert Base.Threads.nthreads() == 1
	using Strided
	using BenchmarkTools
	using Statistics
end

# ╔═╡ 29d8b9fc-eca5-4cce-b18b-692cfe76227c
versioninfo()

# ╔═╡ a56c11fc-4a80-4820-9d4d-8510a5da4829
copy_permuted = let
	A = rand(1000, 1000)
	B = similar(A)
	permAlazy = PermutedDimsArray(A, (2, 1))  # lazy permuted view (no copy)
	@benchmark $B .= $permAlazy
end

# ╔═╡ 726d5523-d496-4665-875e-be80feb35754
copy_permuted_strided = let
	A = rand(1000, 1000)
	B = similar(A)
	permAlazy = PermutedDimsArray(A, (2, 1))  # lazy permuted view (no copy)
	@benchmark @strided $B .= $permAlazy
end

# ╔═╡ 91871ece-2b69-462c-859c-421533b3fef2
mean(copy_permuted.times) / mean(copy_permuted_strided.times)

# ╔═╡ e9ee229e-803e-4ab7-aa7e-f79c436b3e97
zip_map_mixed = let
	A = rand(1000, 1000)
	B = rand(1000, 1000)
	permAlazy = PermutedDimsArray(A, (2, 1))
	out = similar(A)
	@benchmark $out .= $permAlazy .+ $B	
end

# ╔═╡ d281a1ed-caf6-4bef-9598-3df4cb4e530f
zip_map_mixed_strided = let
	A = rand(1000, 1000)
	B = rand(1000, 1000)
	permAlazy = PermutedDimsArray(A, (2, 1))
	out = similar(A)
	@benchmark @strided $out .= $permAlazy .+ $B
end

# ╔═╡ c7987f13-ee2b-4ba8-80ac-7cb95affb6f9
mean(zip_map_mixed.times) / mean(zip_map_mixed_strided.times)

# ╔═╡ 9fa850bf-3abb-4daf-9862-73655140f20d
reduce_transposed = let
	A = rand(1000, 1000)
	@benchmark sum($A)
end

# ╔═╡ 1b9b8c6d-aa71-4e1c-aa8f-fcc8c599799c
reduce_transposed_strided = let
	A = rand(1000, 1000)
	@benchmark sum($A')
end

# ╔═╡ 09cbdc49-fbe9-4854-b10c-2913818f38db
mean(reduce_transposed.times) / mean(reduce_transposed_strided.times)

# ╔═╡ d8d1383a-e070-49ef-8c3d-b2e54e686cf5
let
	A = rand(1000, 1000)
	@benchmark @strided sum($A')
end

# ╔═╡ c385609f-b78e-4ad3-b95d-2961064aec9a
symmetrize_aat = let
	A = rand(4000, 4000)
	B = similar(A)
	@benchmark $B .= ($A .+ $A') ./ 2;
end

# ╔═╡ 0d107efb-617a-4598-ac31-1ff6fe6a80d8
symmetrize_aat_strided = let
	A = rand(4000, 4000)
	B = similar(A)
	@benchmark @strided $B .= ($A .+ $A') ./ 2;
end

# ╔═╡ c1259929-5868-4178-a870-05098e65e8ab
mean(symmetrize_aat.times) / mean(symmetrize_aat_strided.times)

# ╔═╡ 8c54539c-3f04-486f-8de2-0131f8368913
scale_transpose = let
	A = rand(1000, 1000)
	B = similar(A)
	@benchmark $B .= 3 .* $A';
end

# ╔═╡ 6e0eb066-d783-444b-aa01-0d9395539d06
scale_transpose_strided = let
	A = rand(1000, 1000)
	B = similar(A)
	@benchmark @strided $B .= 3 .* $A';
end

# ╔═╡ f18130bc-dc8d-41ed-9617-8349a0253b9d
mean(scale_transpose.times) / mean(scale_transpose_strided.times)

# ╔═╡ 04158108-cc50-4850-8d0c-a1068a8ebce9
nonlinear_map = let
	A = rand(1000, 1000)
	B = similar(A)
	@benchmark $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
end

# ╔═╡ d4facb58-37f6-4849-9e62-75eeb45ed391
nonlinear_map_strided = let
	A = rand(1000, 1000)
	B = similar(A)
	@benchmark @strided $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
end

# ╔═╡ 65c3e4a6-be6b-425d-8477-8e0e65660fbe
mean(nonlinear_map.times) / mean(nonlinear_map_strided.times)

# ╔═╡ bd460e68-4f07-4589-a308-ce4051718d7e
permutedims_4d = let
	A = randn(32,32,32,32)
	B = similar(A)
	@benchmark permutedims!($B, $A, (4,3,2,1))
end

# ╔═╡ ef02e39e-9a3e-43be-9c75-9869d21b0167
permutedims_4d_strided = let
	A = randn(32,32,32,32)
	B = similar(A)
	@benchmark @strided permutedims!($B, $A, (4,3,2,1))
end

# ╔═╡ 4c21fe9b-b9dd-4e3f-a643-59d9d17b0ada
mean(permutedims_4d.times) / mean(permutedims_4d_strided.times)

# ╔═╡ 9e6b8173-7c62-4f8a-bbf7-6884a0bc1a7f
multi_permute_sum = let
	A = randn(32,32,32,32)
	B = similar(A)
	@benchmark $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
end

# ╔═╡ f557a7f9-1152-44fd-892d-02d55994d4e2
multi_permute_sum_strided = let
	A = randn(32,32,32,32)
	B = similar(A)
	@benchmark @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
end

# ╔═╡ e37c3e54-7ebf-48e7-94f7-8a536f148408
mean(multi_permute_sum.times) / mean(multi_permute_sum_strided.times)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Strided = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"

[compat]
BenchmarkTools = "~1.6.3"
Strided = "~2.3.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "db67261165a1c136ec24d7eedf5f6c164b5bb1f2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"

    [deps.PackageExtensionCompat.weakdeps]
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Strided]]
deps = ["LinearAlgebra", "StridedViews", "TupleTools"]
git-tree-sha1 = "c2e72c33ac8871d104901db736aecb36b223f10c"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "2.3.2"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "e34a59ea9c7abc8f10bfd77578de9d64bded2859"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.4.3"

    [deps.StridedViews.extensions]
    StridedViewsCUDAExt = "CUDA"
    StridedViewsPtrArraysExt = "PtrArrays"

    [deps.StridedViews.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    PtrArrays = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "9297459be9e338e546f5c4bedb59b3b5674da7f1"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╠═323269d2-f801-11f0-8d95-4519f700d1da
# ╠═29d8b9fc-eca5-4cce-b18b-692cfe76227c
# ╠═a56c11fc-4a80-4820-9d4d-8510a5da4829
# ╠═726d5523-d496-4665-875e-be80feb35754
# ╠═91871ece-2b69-462c-859c-421533b3fef2
# ╠═e9ee229e-803e-4ab7-aa7e-f79c436b3e97
# ╠═d281a1ed-caf6-4bef-9598-3df4cb4e530f
# ╠═c7987f13-ee2b-4ba8-80ac-7cb95affb6f9
# ╠═9fa850bf-3abb-4daf-9862-73655140f20d
# ╠═1b9b8c6d-aa71-4e1c-aa8f-fcc8c599799c
# ╠═09cbdc49-fbe9-4854-b10c-2913818f38db
# ╠═d8d1383a-e070-49ef-8c3d-b2e54e686cf5
# ╠═c385609f-b78e-4ad3-b95d-2961064aec9a
# ╠═0d107efb-617a-4598-ac31-1ff6fe6a80d8
# ╠═c1259929-5868-4178-a870-05098e65e8ab
# ╠═8c54539c-3f04-486f-8de2-0131f8368913
# ╠═6e0eb066-d783-444b-aa01-0d9395539d06
# ╠═f18130bc-dc8d-41ed-9617-8349a0253b9d
# ╠═04158108-cc50-4850-8d0c-a1068a8ebce9
# ╠═d4facb58-37f6-4849-9e62-75eeb45ed391
# ╠═65c3e4a6-be6b-425d-8477-8e0e65660fbe
# ╠═bd460e68-4f07-4589-a308-ce4051718d7e
# ╠═ef02e39e-9a3e-43be-9c75-9869d21b0167
# ╠═4c21fe9b-b9dd-4e3f-a643-59d9d17b0ada
# ╠═9e6b8173-7c62-4f8a-bbf7-6884a0bc1a7f
# ╠═f557a7f9-1152-44fd-892d-02d55994d4e2
# ╠═e37c3e54-7ebf-48e7-94f7-8a536f148408
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
