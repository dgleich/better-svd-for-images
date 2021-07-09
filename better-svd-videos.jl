### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 34ff0254-da58-11eb-3cd1-9de133b02ae1
begin
	import Pkg
	Pkg.add(["Images","TestImages",
			"TiledIteration",
			"PlutoUI", "Plots", "Combinatorics",
			"MosaicViews", "PerceptualColourMaps", "VideoIO"])
	using Images,TestImages,LinearAlgebra,TiledIteration,PlutoUI,Plots
	using MosaicViews, PerceptualColourMaps, FFTW, VideoIO, Combinatorics
end

# ╔═╡ d238625c-7fcd-4d34-8285-6f889240248d
begin 
	vid = openvideo(VideoIO.testvideo("ladybird")) do f 
			collect(f)
		end
end

# ╔═╡ 340f339b-9f00-4228-bd65-0a27f58fc996
size(vid)

# ╔═╡ 55b4bd75-e696-4a3c-997a-22ad4e59aa92
begin
	function crop_to_multiple(im, factor; center::Bool = true)
		newsize = div.(size(im), factor).*factor 
		offsets = div.(mod.(size(im), factor), center ? 2 : 1)
		return im[range.(offsets.+1, newsize.+offsets, step=1)...]
	end
	vidarr = crop_to_multiple(cat(vid...,dims=3),40) # video as array	
end

# ╔═╡ 63c1dc77-4c8f-4afb-a220-bc06238e1d92
begin 
	function array2frames(Z::Array{T,3}) where T
		collect(eachslice(Z;dims=3))
	end
	array2frames(vidarr)
end

# ╔═╡ efd07cac-cddf-474f-9cd2-3589dc62b190
begin 
	tile_sizes = [5,10,20,40]
	frame_sizes = [5,10,20,40]
end 

# ╔═╡ d9a53542-9a5b-4735-bc78-08b86dd35707
@bind tilesz_str Select(string.(tile_sizes) .=> string.(tile_sizes), default="10")

# ╔═╡ 1c843884-070a-41ed-aaac-c46c15bff20d
tilesz = parse(Int, tilesz_str)

# ╔═╡ 5d6be6c2-f011-49cb-9366-64aa48b72a7a
@bind framesz_str Select(string.(frame_sizes) .=> string.(frame_sizes), default="10")

# ╔═╡ 9ee7409a-9bda-4827-9630-56c0d6be3f18
framesz = parse(Int, framesz_str)

# ╔═╡ 22987ff8-c802-41d3-9a70-135fd3daf629
begin 
	function tile_reshape(im,tilesz,framesz) 
	  hcat(vec(map(tile -> vec(im[tile...]),
			TileIterator(axes((im)), (tilesz,tilesz,framesz))))...)
	end
	X = tile_reshape(vidarr, tilesz, framesz)
end

# ╔═╡ 3674247e-224b-485d-b981-90c0161139be
#tU,tS,tV = svd(Float64.(Gray.(X))); # this is too slow, ugh... 

# ╔═╡ 0d8c3e6f-05c8-4ce2-8289-aef937afd8aa
size(X)

# ╔═╡ be92a63e-b329-46d7-a66c-07e7981e683b
Q,R = qr(Float64.(Gray.(X))');

# ╔═╡ ae872b1c-75ea-4c39-a59b-fbf66fc717ce
tVQ, tS, tU = svd(R);

# ╔═╡ 9f140f91-be19-4f03-96fc-4c82cdb14ab2
tV = Q*tVQ;

# ╔═╡ 898ca787-59ba-4775-a57d-7d7c00b3faaf
begin
	# Need to be careful here as we need to normalize things across all images!
	# Here's the code for apply diverging colour map...
#=
	    minv = minimum(img)
    maxv = maximum(img)

    if refval < minv || refval > maxv
        @warn("Reference value is outside the range of image values")
    end

    dr = maximum([maxv - refval, refval - minv])
    rnge = [-dr dr] .+ refval

    return applycolourmap(img, cmap, rnge)
	=# 
	function array2frames_cmap(Z::Array{T,3}) where T
		refval = 0 
		minv,maxv = extrema(Z)
		dr = maximum([maxv - refval, refval - minv])
	    rnge = [-dr dr] .+ refval
		
		frames = array2frames(Z)
		
		map(A -> colorview(RGB, 
			permutedims(applycolourmap(
                        copy(A), cmap("D2"), rnge),
					   (3,1,2))), frames)
	end
	array2frames_cmap(reshape(tU[:,1], (tilesz,tilesz,framesz)))
end


# ╔═╡ 77e39090-d27f-4f3b-a2f1-fb8b8aad21ed
begin
	map(i->
		(array2frames_cmap(reshape(tU[:,i], (tilesz,tilesz,framesz)))),
		1:10)
end

# ╔═╡ 9f631d3e-d0ae-4b6e-81e8-814f604f17e9
begin 
	
	# way more complicated than strictly necessary,
	# but this is a general solution to find the ith index
	# in a multinomial expansion
	
	# https://math.stackexchange.com/questions/2576122/can-stars-and-bars-method-be-used-with-constraints-on-the-number-of-stars
function stars_and_bars_upper_constraints(n::Integer,upper::NTuple{N,Int}) where N
  k = length(upper)
  #@show k
  nall = binomial(n-1,k-1)
  #@show nall
  for v in upper
    if n-1-v <= 0
      continue
    end
    nall -= binomial(n-1-v,k-1)
  end
  return nall
end

""" Find the ith tuple with sum equal to s and no element larger than the upperbounds. """
function tuple_sum_unrank(s::Integer, upper::NTuple{N,Int}, i::Integer) where N
	# TODO, if sum of upper >= s, then there is a trick...
	# https://math.stackexchange.com/questions/1388764/integer-solutions-using-pie/1388847#1388847
	# see drhab answer...
	for ta = multiexponents(N, s-N) # sum to s-N (because )
		t = NTuple{N,Int}(ta)
		t  = t.+1 # adjust to sum to s so we add one to each index
		if all(t .<= upper) # check for upper-boudns
			i -=1
			if i==0
				return t
			end
		end
	end
	return i
end

##

function idct_id(offset,sz,fsz)
  A = zeros(sz,sz,fsz)
  # find the triangular index
	for s=3:sz+sz+fsz # these are the diagonals...
		# size of the current diagonal
		ndiag = stars_and_bars_upper_constraints(s, (sz,sz,fsz))
		if offset > ndiag
			offset -= ndiag
		else
			# ideally we'd like to unrank this... but ugh...
			# instead, we are just going to enumerate and throw out
			# invalid stuff...
			#@show ndiag, offset
			t = tuple_sum_unrank(s, (sz, sz, fsz), offset)
			#@show t, typeof(t)
			@assert typeof(t) <: Tuple
			A[t...] = 1
			break
		end
	end
	return A
end
	
	
	
	map(i->
		array2frames_cmap(idct(idct_id(i,tilesz,framesz))),
		1:10)
end

# ╔═╡ 3182eb0a-b5c8-4292-ab9f-201473769637
map(i->array2frames_cmap(dct(reshape(tU[:,i], (tilesz,tilesz,framesz)))), 1:10)

# ╔═╡ 40cd74a9-791b-44ab-81ee-940d24fe2e4b
begin 
	# Ahh, I stole this from a github issue/pull request for ...
	# https://github.com/JuliaStats/MultivariateStats.jl/pull/130/files
	## Comparison to other implementations
	# The implementation of varimax in R row-normlises the input matrix before
	# application of the algorithm and rescales the rows afterwards.
	## Reference
	# Mikkel B. Stegmann, Karl Sjöstrand, Rasmus Larsen, "Sparse modeling of
	# landmark and texture variability using the orthomax criterion,"
	# Proc. SPIE 6144, Medical Imaging 2006: Image Processing, 61441G
	# (10 March 2006); doi: 10.1117/12.651293
	function orthomax(F::AbstractMatrix, γ, miniter, maxiter, ϵ)
		n, p = size(F)
		if n < 2
			return (F, Matrix{eltype(F)}(I, p, p))
		end

		# Warm up step
		# Do one step. If that first step did not lead away from the identity
		# matrix enough use a random orthogonal matrix as a starting point.
		M = svd(F' * (F .^ 3 - γ / n * F * Diagonal(vec(sum(F .^ 2, dims=1)))))
		R = M.U * M.Vt
		if norm(R - Matrix{eltype(R)}(I, p, p)) < ϵ
			R = qr(randn(p, p)).Q
		end

		# Main iterations
		d = 0
		lastchange = NaN
		converged = false
		for i in 1:maxiter
			dold  = d
			B = F * R
			M = svd(F' * (B .^ 3 - γ / n * B * Diagonal(vec(sum(B .^ 2, dims=1)))))
			R = M.U * M.Vt
			d = sum(M.S)
			lastchange = abs(d - dold) / d
			if lastchange < ϵ && i >= miniter
				converged = true
				break
			end
		end

		converged || throw(ConvergenceException(maxiter, lastchange, ϵ))

		(F * R, R)
	 end
	tU2 = orthomax(tU[:,1:32], 1.0, 20, 1000, 1e-12)[1]
end

# ╔═╡ f3fbac2b-3582-4404-9391-a2099c97e124
begin
	map(i->
		(array2frames_cmap(reshape(tU2[:,i], (tilesz,tilesz,framesz)))),
		1:size(tU2,2))
end

# ╔═╡ 7d617284-7779-470a-821a-296572f00578
begin
	rtile = 3
	function tiled_svd_approx(r::Int, tilesz::Int, framesz::Int; tU, tS, tV)
		Y = tU[:,1:r]*Diagonal(tS[1:r])*tV[:,1:r]'
		B = similar(vidarr, Float64)
		for (i,tile) in enumerate(TileIterator(axes(vidarr), (tilesz,tilesz,framesz)))
		  B[tile...] .= reshape(Y[:,i], (tilesz,tilesz,framesz))
		end
		return B, length(tU[:,1:r])+length(tV[:,1:r])
	end
	tile_svd_approx_gray, nparam_tile_gray = tiled_svd_approx(rtile,tilesz,framesz; tU, tS, tV)
	# save a video
	frames = map(M -> Gray{N0f8}.(clamp01nan.(M)), array2frames((tile_svd_approx_gray)))
end	

# ╔═╡ 6d40196d-9f4a-4baa-b0e8-d8f43244ef7d
VideoIO.save("ladybird-grey-tile-$(tilesz)-frame-$(framesz)-rank-$rtile.mp4", frames)

# ╔═╡ 77cfd1a7-2326-47da-b710-9aa443aa62cc
frames

# ╔═╡ Cell order:
# ╠═34ff0254-da58-11eb-3cd1-9de133b02ae1
# ╠═d238625c-7fcd-4d34-8285-6f889240248d
# ╠═340f339b-9f00-4228-bd65-0a27f58fc996
# ╠═55b4bd75-e696-4a3c-997a-22ad4e59aa92
# ╠═63c1dc77-4c8f-4afb-a220-bc06238e1d92
# ╠═efd07cac-cddf-474f-9cd2-3589dc62b190
# ╠═d9a53542-9a5b-4735-bc78-08b86dd35707
# ╠═1c843884-070a-41ed-aaac-c46c15bff20d
# ╠═5d6be6c2-f011-49cb-9366-64aa48b72a7a
# ╠═9ee7409a-9bda-4827-9630-56c0d6be3f18
# ╠═22987ff8-c802-41d3-9a70-135fd3daf629
# ╠═3674247e-224b-485d-b981-90c0161139be
# ╠═0d8c3e6f-05c8-4ce2-8289-aef937afd8aa
# ╠═be92a63e-b329-46d7-a66c-07e7981e683b
# ╠═ae872b1c-75ea-4c39-a59b-fbf66fc717ce
# ╠═9f140f91-be19-4f03-96fc-4c82cdb14ab2
# ╠═898ca787-59ba-4775-a57d-7d7c00b3faaf
# ╠═77e39090-d27f-4f3b-a2f1-fb8b8aad21ed
# ╠═9f631d3e-d0ae-4b6e-81e8-814f604f17e9
# ╠═3182eb0a-b5c8-4292-ab9f-201473769637
# ╠═40cd74a9-791b-44ab-81ee-940d24fe2e4b
# ╠═f3fbac2b-3582-4404-9391-a2099c97e124
# ╠═7d617284-7779-470a-821a-296572f00578
# ╠═6d40196d-9f4a-4baa-b0e8-d8f43244ef7d
# ╠═77cfd1a7-2326-47da-b710-9aa443aa62cc
