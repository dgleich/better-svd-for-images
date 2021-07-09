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

# ╔═╡ ccbc7c67-e16a-48f1-a927-c002257fcc54
begin
	import Pkg
	Pkg.add(["Images","TestImages","FileIO",
			"TiledIteration",
			"PlutoUI", "Plots", 
			"MosaicViews", "PerceptualColourMaps", "VideoIO"])
	using Images,TestImages,LinearAlgebra,TiledIteration,PlutoUI,Plots
	using MosaicViews, PerceptualColourMaps, FFTW, FileIO, Random
end

# ╔═╡ cbd2ea36-7ef6-4ea8-82ac-34b3e96ed8d5
md"""
### Origin and brief references
> Paul Constantine and I looked at PCA of large collections of images in the context of computing Tall and Skinny QR factorizations where we first saw discrete fourier transform-like structure as we show below for a single image. Paul had originally noticed smoothness in terms of parameterized solutions to linear systems and parameterized computational science problems, which is closely related.
> In a different vein, Lek-Heng Lim often sharply critiqued work that blindly deploys data-as-model frameworks without sufficient thought. In this case, the matrix-of-tiles is at least a vector-space of samples, so it has better properties _as a linear operator_ or _linear space_. (Which is why we are able to find structure inside it.)
> Finally, Vaneet Agrawal mentioned _reshaping_ video data (originally a "3d" matrix) into a 5d array to improve low-rank modeling. This was in the context of work on low-tubal rank matrix completition, which uses a tensor computing framework based on ideas by Misha Kilmer and subsequent work by Kilmer, Bramen, and Hao. (See also our own take on these ideas in a paper with Gleich, Greif, and Varah.) Here we are really doing the same thing. 
> And of course, there are influences from all the neural network patch or tile convolution models in all of the recent deep learning on images setups.
"""

# ╔═╡ 59c37536-fff2-480a-80bc-f97125e117af
md"## Initializing packages
This adds all the packages you need to run these examples.
"

# ╔═╡ 2c8a2738-77b1-4803-91d3-9b5172e4a94a
md"""# We need to start with a picture. 


$(@bind im_choice Select(["webcam","upload","url","coffee","cat","earth","mandrill","house","blobs"] |> x -> x .=> x,default="coffee"))

The box above allows you to choose one! 
- `webcam`: see the webcam image area below. You can take your own picture!
- `upload`: see the upload area below. You can send us a picture from your computer. (I have no idea what happens to it, but we don't intentially do anything nefarious.) 
- `url`: you have the picture on the internet somewhere we can download it from. Great, tell us the URL and we'll grab it for you. 
- or use one of the named images from `TestImages.jl`. 

My favorite is `coffee`. Nothing like a fantastic coffee! 
"""

# ╔═╡ 3112233a-543d-48f5-823f-5b1e1cafcb61
begin 
	function process_raw_camera_data(raw_camera_data)
	# the raw image data is a long byte array, we need to transform it into something
	# more "Julian" - something with more _structure_.
	
	# The encoding of the raw byte stream is:
	# every 4 bytes is a single pixel
	# every pixel has 4 values: Red, Green, Blue, Alpha
	# (we ignore alpha for this notebook)
	
	# So to get the red values for each pixel, we take every 4th value, starting at 
	# the 1st:
	reds_flat = UInt8.(raw_camera_data["data"][1:4:end])
	greens_flat = UInt8.(raw_camera_data["data"][2:4:end])
	blues_flat = UInt8.(raw_camera_data["data"][3:4:end])
	
	# but these are still 1-dimensional arrays, nicknamed 'flat' arrays
	# We will 'reshape' this into 2D arrays:
	
	width = raw_camera_data["width"]
	height = raw_camera_data["height"]
	
	# shuffle and flip to get it in the right shape
	reds = reshape(reds_flat, (width, height))' / 255.0
	greens = reshape(greens_flat, (width, height))' / 255.0
	blues = reshape(blues_flat, (width, height))' / 255.0
	
	# we have our 2D array for each color
	# Let's create a single 2D array, where each value contains the R, G and B value of 
	# that pixel
	
	RGB.(reds, greens, blues)
end
	
	function camera_input(;max_size=200, default_url="https://i.imgur.com/SUmi94P.png")
"""
<span class="pl-image waiting-for-permission">
<style>
	
	.pl-image.popped-out {
		position: fixed;
		top: 0;
		right: 0;
		z-index: 5;
	}

	.pl-image #video-container {
		width: 250px;
	}

	.pl-image video {
		border-radius: 1rem 1rem 0 0;
	}
	.pl-image.waiting-for-permission #video-container {
		display: none;
	}
	.pl-image #prompt {
		display: none;
	}
	.pl-image.waiting-for-permission #prompt {
		width: 250px;
		height: 200px;
		display: grid;
		place-items: center;
		font-family: monospace;
		font-weight: bold;
		text-decoration: underline;
		cursor: pointer;
		border: 5px dashed rgba(0,0,0,.5);
	}

	.pl-image video {
		display: block;
	}
	.pl-image .bar {
		width: inherit;
		display: flex;
		z-index: 6;
	}
	.pl-image .bar#top {
		position: absolute;
		flex-direction: column;
	}
	
	.pl-image .bar#bottom {
		background: black;
		border-radius: 0 0 1rem 1rem;
	}
	.pl-image .bar button {
		flex: 0 0 auto;
		background: rgba(255,255,255,.8);
		border: none;
		width: 2rem;
		height: 2rem;
		border-radius: 100%;
		cursor: pointer;
		z-index: 7;
	}
	.pl-image .bar button#shutter {
		width: 3rem;
		height: 3rem;
		margin: -1.5rem auto .2rem auto;
	}

	.pl-image video.takepicture {
		animation: pictureflash 200ms linear;
	}

	@keyframes pictureflash {
		0% {
			filter: grayscale(1.0) contrast(2.0);
		}

		100% {
			filter: grayscale(0.0) contrast(1.0);
		}
	}
</style>

	<div id="video-container">
		<div id="top" class="bar">
			<button id="stop" title="Stop video">✖</button>
			<button id="pop-out" title="Pop out/pop in">⏏</button>
		</div>
		<video playsinline autoplay></video>
		<div id="bottom" class="bar">
		<button id="shutter" title="Click to take a picture">📷</button>
		</div>
	</div>
		
	<div id="prompt">
		<span>
		Enable webcam
		</span>
	</div>

<script>
	// based on https://github.com/fonsp/printi-static (by the same author)

	const span = currentScript.parentElement
	const video = span.querySelector("video")
	const popout = span.querySelector("button#pop-out")
	const stop = span.querySelector("button#stop")
	const shutter = span.querySelector("button#shutter")
	const prompt = span.querySelector(".pl-image #prompt")

	const maxsize = $(max_size)

	const send_source = (source, src_width, src_height) => {
		const scale = Math.min(1.0, maxsize / src_width, maxsize / src_height)

		const width = Math.floor(src_width * scale)
		const height = Math.floor(src_height * scale)

		const canvas = html`<canvas width=\${width} height=\${height}>`
		const ctx = canvas.getContext("2d")
		ctx.drawImage(source, 0, 0, width, height)

		span.value = {
			width: width,
			height: height,
			data: ctx.getImageData(0, 0, width, height).data,
		}
		span.dispatchEvent(new CustomEvent("input"))
	}
	
	const clear_camera = () => {
		window.stream.getTracks().forEach(s => s.stop());
		video.srcObject = null;

		span.classList.add("waiting-for-permission");
	}

	prompt.onclick = () => {
		navigator.mediaDevices.getUserMedia({
			audio: false,
			video: {
				facingMode: "environment",
			},
		}).then(function(stream) {

			stream.onend = console.log

			window.stream = stream
			video.srcObject = stream
			window.cameraConnected = true
			video.controls = false
			video.play()
			video.controls = false

			span.classList.remove("waiting-for-permission");

		}).catch(function(error) {
			console.log(error)
		});
	}
	stop.onclick = () => {
		clear_camera()
	}
	popout.onclick = () => {
		span.classList.toggle("popped-out")
	}

	shutter.onclick = () => {
		const cl = video.classList
		cl.remove("takepicture")
		void video.offsetHeight
		cl.add("takepicture")
		video.play()
		video.controls = false
		console.log(video)
		send_source(video, video.videoWidth, video.videoHeight)
	}
	
	
	document.addEventListener("visibilitychange", () => {
		if (document.visibilityState != "visible") {
			clear_camera()
		}
	})


	// Set a default image

	const img = html`<img crossOrigin="anonymous">`

	img.onload = () => {
	console.log("helloo")
		send_source(img, img.width, img.height)
	}
	img.src = "$(default_url)"
	console.log(img)
</script>
</span>
""" |> HTML
end

	
	md"""The code to handle the camera input is hidden. It's still a bit raw. It saves the images in `im_webcam`"""
end


# ╔═╡ 2ac8cb5d-ffff-4561-bc08-2ffc368338e2
md""" ### Get your own webcam image. 

From <https://github.com/mitmath/18S191/blob/Fall20/lecture_notebooks/week1/01-images-intro.jl>

$(@bind raw_camera_data camera_input(;max_size=1000))
"""

# ╔═╡ 03604f42-66ce-4a39-b1b1-f5fd49bfccdb
im_webcam = process_raw_camera_data(raw_camera_data)

# ╔═╡ e6e10ee8-2bcb-4743-86be-ffac6b591b96
md""" ### Upload a picture. 
This uses the `PlutoUI.jl` upload code.

$(@bind image_upload_data FilePicker([MIME("image/jpg"),MIME("image/jpeg")]))
"""

# ╔═╡ 2d4125ac-5c10-4150-bc00-e2cfcbfaf5ad
im_upload = load(IOBuffer(image_upload_data["data"]))

# ╔═╡ 52d8a212-0b34-4306-b410-e3d3dce7de3f
md""" ### Give a URL

$(@bind image_url TextField((120,1);default="https://user-images.githubusercontent.com/6933510/110924885-d7f1b200-8322-11eb-9df7-7abf29c8db7d.png"))

This defaults to the the MIT Computational Thinking tree image.
"""

# ╔═╡ fd4b3fef-9df2-4253-81b4-98a9c59a0176
im_url = load(download(image_url))

# ╔═╡ 13b28401-d752-4dae-9ddf-bfd59f3ab995
md""" ## The image we'll use for our SVD analysis.

We crop this slightly to make it a multiple of 64 to make our tile-based analysis more straightforward.

"""

# ╔═╡ 08c9f59e-4408-409a-b324-36dedc005c65
begin
	imagename = im_choice 
	if im_choice == "webcam"
		im = im_webcam
	elseif im_choice == "upload"
		im = im_upload
	elseif im_choice == "url"
		im = im_url
	elseif im_choice == "coffee"
		im = testimage("coffee")	
	elseif im_choice == "cat"
		im = testimage("chelsea")
	elseif im_choice == "earth"
		im = testimage("earth_apollo17") # 3000x3000-ish
		im = imresize(im, ratio=0.25)
	elseif im_choice == "house"
		im = testimage("house")
	elseif im_choice == "mandrill"
		im = testimage("mandrill")
	elseif im_choice == "blobs"
		im = testimage("blobs")
	else
		@error("Please make up your mind!")
	end
	
	function crop_to_multiple(im, factor; center::Bool = true)
		newsize = div.(size(im), factor).*factor 
		offsets = div.(mod.(size(im), factor), center ? 2 : 1)
		return im[range.(offsets.+1, newsize.+offsets, step=1)...]
	end
	im = crop_to_multiple(im, 64)
end

# ╔═╡ 4632bc52-9106-4d05-8f6f-a189e577388d
md"""
But we aren't going to handle color approximations here. (There is another workbook on this that gets deeper into those weeds.) So we are really approximating the following image."""

# ╔═╡ 6a13f71e-983b-4d48-8787-8d58abbebbf5
Gray.(im)

# ╔═╡ 17dae8a4-010a-4816-b5a4-c9448d3cf8b1
md"""
In the standard explaination, we pick a rank for the SVD-based approximation, and then display that approximation. 

**Rank for SVD approximation** (Usually you see something around 15-20.) 
- `rsvd` $(@bind rsvd Slider(1:50,default=15)) 
"""

# ╔═╡ fddb68e0-adc2-4ff5-908d-e0204bf074e3
md""" `rsvd` = $(rsvd)"""

# ╔═╡ c372842c-a529-4ad1-8edb-47d755018dbc
begin 
	#= This is the standard Image-as-Matrix SVD example. =#
  	A = Float64.(Gray.(im))
  	U, S, V = svd(A); 
end; # output hidden because it isn't that interesting

# ╔═╡ dbace9f6-6b02-4e91-88a4-92fe27ec063d
begin
  	function standard_svd_approx(r::Int) 
    	B = U[:,1:r]*Diagonal(S[1:r])*V[:,1:r]'
    	nparam = length(U[:,1:r])+ length(V[:,1:r])
		return B, nparam
  	end 
  	svd_approx_gray, nparam_gray = standard_svd_approx(rsvd)
  	Gray.(svd_approx_gray) # show this
end

# ╔═╡ 037bc839-97e0-44d9-a000-c3cb4a5dcf99
with_terminal() do 
  println("Rank $rsvd approximation")
  println("  Error % = $(100*norm(A-svd_approx_gray)/norm(A))")
  println("  parameters = $nparam_gray")
end

# ╔═╡ 36dccd6c-0024-4f49-88b0-f6b5b2901f5d
md"""
## Number of parameters vs. Error

It's super handy to look at the amount of error compared with the number of parameters. In this case, we look at error as a percent of total image. This is just the 2-norm difference of the image - approximation as a vector (also called Frobenius 2-norm), divided by the norm of the image.
"""

# ╔═╡ 822c2c32-a3e1-40a3-a98f-6fa11a660035
begin 
  function get_nparam_approx_data(reference_im, r, approx_fun)
	approx_im, nparam = approx_fun(r) 
	return nparam, 100*norm(approx_im-reference_im)/norm(reference_im)
  end
  map(r->get_nparam_approx_data(A, r, standard_svd_approx), 1:20)	
end	
	

# ╔═╡ 74cc32bb-dc19-4a1e-9b6a-79fb782e00b3
begin
	scatter(map(r->get_nparam_approx_data(A, r, standard_svd_approx), 1:20), label="Image-as-Matrix SVD")
	ylabel!("Error %")
	xlabel!("Parameters")
end 

# ╔═╡ 11a12eb5-2928-47c1-87c5-2faed7e1ba02
md""" # Image as Matrix-of-Tiles SVD Example

The entire point of this workbook is to convince you there is a better approach based on thinking of the matrix as a collection of image tiles or image patches. This leads to _better_ approximation results but also leads to _deeper_ questions and _more interesting analysis_. In particular, we will see the discrete cosine transform (DCT), which is used in the JPEG format, arise naturally via the SVD! 

The idea is that we divide the image into little tiles, and then represent the images as a matrix where each column is a _vector_ representation of the little tile. 

Here is what this looks like for 16x16 tiles. So each column of the matrix will be `16*16 = 256` sized. 
"""

# ╔═╡ eae7193c-f287-4f62-bb50-cfc998638ba9
begin
""" This is a really crappy implementation to show tiles. This is done just to make it possible. """
function show_tiles(im, tilesz::Integer)
  img = Gray.(im)
  imc = RGB.(img) # convert back to RGB
  for i=tilesz:tilesz:size(imc,1) # show lines
    imc[i,:] .= colorant"yellow"
  end
  for j=tilesz:tilesz:size(imc,2) # show lines
    imc[:,j] .= colorant"yellow"
  end
  imc
end
show_tiles(im, 16)
end


# ╔═╡ c3fd4712-7802-4685-aa5b-4b9d8f81aeab
md""" #### Pick the tile size below 

> We only support a few tile sizes because we set this up to be simple where the image has to evenly divide into tiles.

- Tile size $(@bind tilesz_str Select(string.([4,8,16,32,64]) .=> string.([4,8,16,32,64]), default="8"))
"""

# ╔═╡ cf8ba3f6-3347-49e5-ad5a-92518a4755b7
tilesz = parse(Int, tilesz_str)

# ╔═╡ e953db96-42a3-4506-ad68-3e84cabc47bd
md"""
The following code builds the `pixels-by-tiles` matrix `X`. It uses the `TiledIteration` package to iterate over each tile and then lays it into a column of the matrix by just `hcat`ing all the output from the tile iterator. 

This is done on the color image to show the different color regions. 
"""

# ╔═╡ 606d5856-7051-4633-893a-c33f691dd8ac
begin 
	function tile_reshape(im,tilesz) 
	  hcat(
			map(tile -> vec(im[tile...]),
				TileIterator(axes(im), (tilesz,tilesz))
			)...
		)
			
	end
	X = tile_reshape(im, tilesz)
end

# ╔═╡ 46d589b2-87a1-48a3-86a3-f784a17d6724
md"""We actually compute on the greyscale matrix as before."""

# ╔═╡ 90233a0d-c04b-4216-8b0b-9d7c1912150f
Gray.(X)

# ╔═╡ 02811d5e-7948-4cf1-9330-dd95e7bcd55a
md"""
On important note is that we aren't making any more data when we do this. The values of `X` and `A` are the same, and there are exactly the same number of them. The key difference is that those in `X` are structured in terms of tiles. This allows us to exploit different structure in the image.
"""

# ╔═╡ 3632bf49-d6bf-4de0-9450-5233517db4ef
length(X), length(A)

# ╔═╡ 342cc50a-81e9-4688-829a-e7018cff30d6
md"""
As before, we compute the SVD of the matrix X. The output `tU` means `tiled U` as a short-hand. But we need to remember that this needs to be interpreted differently.
"""

# ╔═╡ 25215211-7c33-41db-a327-7c2f45500b3c
tU,tS,tV = svd(Float64.(Gray.(X))); # output hidden because it's boring

# ╔═╡ 2ae1b465-2f67-483c-bd56-af0b859940f7
md"""
Here, we pick a rank for the Tiled SVD approximation. 

**Rank for Tiled SVD approximation** (Usually you see something around 3-5.) 
- `rtile` $(@bind rtile Slider(1:25,default=3)) 

The reconstruction code given a tile size is very different. We first build an rank `rtile` approximation of the matrix `X` in the matrix `Y`. Then we read each column of `Y` back into a matrix `B` that approximates the matrix `A` by using the TileIterator again.
"""

# ╔═╡ c2233159-59f4-4516-859c-5c25170d53aa
md" `rtile` = $(rtile)"

# ╔═╡ 3e7902ab-4f75-4d14-901b-d2e380996c8f
begin
	function tiled_svd_approx(r::Int, tilesz::Int; tU, tS, tV)
		Y = tU[:,1:r]*Diagonal(tS[1:r])*tV[:,1:r]'
		B = copy(A)
		for (i,tile) in enumerate(TileIterator(axes(A), (tilesz,tilesz)))
		  B[tile...] .= reshape(Y[:,i], (tilesz,tilesz))
		end
		return B, length(tU[:,1:r])+length(tV[:,1:r])
	end
	tile_svd_approx_gray, nparam_tile_gray = tiled_svd_approx(rtile,tilesz; tU, tS, tV)
	Gray.(tile_svd_approx_gray) # show this
end	

# ╔═╡ 49bd0281-151c-4306-bcf8-2eb8c618e04d
begin 
punchline_str = begin
	iam_approx,nparam_iam = standard_svd_approx(rsvd)
	iat_approx,nparam_iat = tiled_svd_approx(rtile,tilesz; tU, tS, tV)
	iam_error = 100*norm(iam_approx-A)/norm(A)
	iat_error =  100*norm(iat_approx-A)/norm(A)
	md"""
| Image-as-Matrix SVD Approximation | Images-as-$tilesz x $tilesz Tiles Approximation |
|:---------:|:-----------:|
| $(Gray.(iam_approx)) | $(Gray.(iat_approx)) | 
| $(nparam_iam) parameters | $(nparam_iat) parameters |
| Rank $(rsvd) approximation | Rank $(rtile) approximation |
| $(round(iam_error;digits=2)) % Error | $(round(iat_error;digits=2)) % Error |
"""
end
md"> Expand to see code to generate punchline figures"
end


# ╔═╡ b9473b23-957b-4d23-8ad5-893c6c4063cf
md"""# Pedagogical image-as-matrix compared with image-as-tiles SVD examples

A common example in linear algebra classes or matrix computations or numerical computing classes is to take the SVD of a matrix that represents an image. Typically, the matrix is just taken _to be_ the image, which we call _image-as-matrix_ here. Then we create a low-rank approximation (from the SVD) of that matrix, and show the result as an image.

This example is curious for a few reasons. 
- First, the _image-as-matrix_ view creates a matrix that is a rather odd linear operator. It's synatically valid, of course. This leads into a long discussion of _what is a matrix_ (but not _what is **the** matrix!) that we'll avoid.
- Second, the goal is to approximate the image! We'll see there is a better strategy to approximate the underlying images (and that makes much more sense as a linear operator). 

The punchline is below. 

$(punchline_str)
"""

# ╔═╡ 8c059291-0564-4ec7-80a0-13432962254a
with_terminal() do 
  println("Tiled rank $rtile approximation")
  println("  Error % = $(100*norm(A-tile_svd_approx_gray)/norm(A))")
  println("  parameters = $nparam_tile_gray")
end

# ╔═╡ b688c31c-5394-40e8-9e6c-2f3fb64fffaf
md""" ## The number of parameters when `rtile = rsvd` are not the same! 
Note that for the `rtile=3` approximation (the default set in this notebook) there are 10,560 parameters. The `rsvd=15` approximation (the default set here) there are 14,400 parameters for the coffee image. (This will vary depending on what image you have.

So what we want to look at is Error % as a function of tile-size, rank, and number of parameters. That's what we show below.

The standard Image-as-Matrix approximation is shown in red. We get better approximations for the tiled examples (on the coffee image) if the curve is below the red curve. This happens for many different tile sizes. But it does not for `64x64` tiles. This is because there are just too few such tiles for a small image. If you want to see those work, try a bigger image like `earth`.
"""

# ╔═╡ 3d9f7a23-b5d3-49ef-ae6f-d3644ba0bcf8
begin
	plot(map(r->get_nparam_approx_data(A, r, standard_svd_approx), 1:20), label="Image-as-Matrix SVD",marker=:dot,linewidth=3,color=:red)
	
	for ptilesz = [4,8,16,32,64]
		pX = tile_reshape(A, ptilesz)
		ptU,ptS,ptV = svd(pX)
		plot!(map(r->get_nparam_approx_data(A, r, r->tiled_svd_approx(r,ptilesz;tU=ptU,tV=ptV,tS=ptS)), 1:min(20,min(size(pX)...))), label="Image-as-$(ptilesz)x$(ptilesz)-Tiles SVD " ,marker=:dot)
	end
	
	ylabel!("Error %")
	xlabel!("Parameters")
	xlims!(1,length(U[:,1:20]) + length(V[:,1:20]))
end 

# ╔═╡ 860b01bd-47e0-41f2-bd05-f5f3957b8370
md""" Of course, it's often useful to look at the picture the other way. For a fixed level of error?, is the tiled approximation better than the Image-as-Matrix approximation?

What we want to see for a given error level (x-axis) which approximation has fewer parameters. So we take their ratio. In this case, let's look at the ratio of so that ratios above 1 show that the tiled approximation is better. 
"""

# ╔═╡ 9fed5b5e-0106-484f-898c-1d664033b339
begin 
	# Here we want to make the opposite plot, and show the number of parameters used for a given approximation level.
	# This uses the fixed-tile size above (tilesz)
	svd_results = map(r->get_nparam_approx_data(A, r, standard_svd_approx), 1:20)
	tiled_svd_results = map(r->get_nparam_approx_data(A, r, 
								r->tiled_svd_approx(r,tilesz; tU,tS,tV)), 
							1:min(20,min(size(X)...)))
	relapprox = Tuple{Float64,Float64}[]
	for (nparam,approx_level) in svd_results
		maxparams = maximum(first, tiled_svd_results)
		for (nparam_tile,tile_approx_level) in tiled_svd_results
			if tile_approx_level < approx_level
				push!(relapprox, (approx_level, nparam/nparam_tile))
				break
			end
		end
	end

	plot(relapprox,marker=:dot, size=(600,500), label="Image as $(tilesz)x$(tilesz) Tiles" )
	xlabel!("Error %")
	hline!([1.0], label="", linewidth=2)
	annotate!(maximum(relapprox[1]), 1.0, 
			("↑ Tiled Approx is Better", :right, :bottom))
	annotate!(maximum(relapprox[1]), 1.0, 
			("↓ Standard Approx is Better", :right, :top))

	ylabel!("Image-as-Matrix Approx Params / Tiled Approx Params ")
end

# ╔═╡ 5b99d2df-fb5a-4fcc-b5f9-f44ca5328814
md"""
# Interpreting the Tiled SVD basis.

One of the best reasons to look at the tiled approximation is that the left singular vectors `tU` are themselves `image patches` that we can look at! 

For this section, you get the best results with the `cat` picture or `earth`, but the others aren't too far off. So I suggest switching to the cat picture.
"""

# ╔═╡ e869b69b-b79f-4bd6-bd2f-6015e22028d9
md""" > The following is a little function to upscale images to make them bigger to see. It's a good exercise in multidimensional array programming in Julia as this will work for vectors, matrices, tensors, etc."""

# ╔═╡ 00c0d888-f192-4ddc-9a94-5cf9624ddd26
begin 
	function upscale(X::AbstractArray, factor::Int)
		sz2 = factor.*size(X) 
		Y = similar(X, sz2...)
		I1 = oneunit(first(CartesianIndices(X)))
		
		for xi = CartesianIndices(X)
			start = factor*(xi-I1)+I1
			last = factor*(xi)
			for yi in start:last 
				Y[yi] = X[xi]
			end
		end
		return Y
	end
	upscale([1 2;3 4],3)
end

# ╔═╡ def9d9d3-2185-4d16-a63f-e415bb91eac3
md""" Below are the first 16 singular vectors of `tU` resized to be little image tiles themselves. (column-wise organization! so 1-4 are in first column...) 

The colors are "green" for negative and "purple" for positive and "white" for near zero. These are scaled so that "white" always indicates 0. 

> Remember that singular vectors have an ambiguous sign. So you may see different colors, but the patterns should be the same.

The first vector should be constant, which should be either pure green or purple. (Up to what your eyeball/monitor combo can show.) 

But the second and third vector for `cat` show a roughly. vertical and horizontal split. You may see slight angles for other images. 

Then we see other `sin`/`cos` like before in the next few. In particular, the vectors seem to alternate more frequently as the index gets higher."""

# ╔═╡ 5a491365-da8e-43bb-987d-24fccb9a2e6d
Z = mosaicview(map(i->colorview(RGB,
                      permutedims(applydivergingcolourmap(
                        reshape(tU[:,i], (tilesz,tilesz)), cmap("D2"), 0),
                      (3,1,2))),
               1:16)..., nrow=4, npad=5, fillvalue=127) |> Z -> map(clamp01nan, Z) |> Z -> upscale(Z,8)

# ╔═╡ ade6e8f2-bd1b-4569-8efb-aae3d42ee340
md""" ### Introducing the Discrete Cosine Transform

At some point in your life, you either learned about, or will learn more about, the Fourier transform and the discrete cosine transform (DCT). These are mainstays of signal processing and enable much of the modern world to work. 

This isn't the place to introduce them. You have wikipedia and google... and youtube... there are excellent explainations here.

In particular, the DCT underlies the JPG file format. 

### The DCT in JPG

As a rough guide, the way the JPG files are stored is that the image is broken up into little 8x8 tiles (like we did here!) and then they apply a DCT transform to each block to determine how much information to keep. This is basically the same as our low-rank approximation, except we use the SVD basis instead of the DCT. 

### The DCT and SVD

The reason I mention the DCT here is that the discrete cosine transform "looks a lot like" the picture above. (Not exactly the same, but very similar.)

Here's the same picture for the DCT. 
"""

# ╔═╡ 5c53bc42-5f09-482e-81a8-43c1cfa71be6
begin 
	function idct_id(offset,sz)
	  A = zeros(sz,sz)
	  # find the triangular index
	  for j=1:sz
		# jth diagonal... 
		if offset > j
		  offset -= j
		else
		  # in this diagonal...
		  A[j-offset+1,offset] = 1
		  break
		end
	  end
	  return A
	end
	Zdct = mosaicview(map(i->colorview(RGB,
                      permutedims(applydivergingcolourmap(
                        idct(idct_id(i,tilesz)), cmap("D2"), 0),
                      (3,1,2))),
               1:16)..., nrow=4, npad=5, fillvalue=127) |> Z -> map(clamp01nan, Z) |> Z->upscale(Z,8)
end 

# ╔═╡ 73e63f45-1458-40ca-9ec4-bb1c00932300
md"""
Yeah, not the same, but pretty darn close!  If you want it to be closer, try a bigger image. (Upload your own image that's around 1500x1000 and use 8x8 blocks... for instance.) Using a high-res Windows XP wallpaper `bliss` image (at some time availble from this URL <https://i.imgur.com/uGRFZEs.jpg>) showed excellent results.
> In fact, one time when I was looking at this notebook, I thought I had deleted the SVD-based result because I couldn't quickly tell the result on `bliss` apart from the exact result as I had been expected.
"""

# ╔═╡ 211f12bd-6d77-4d78-b486-b8f381bdde8d
md"""
### Quantifying the SVD as DCT approximation.

The DCT is a matrix transformation. That means we an ask the DCT to tell us how much those SVD-like images are `DCT-like`. 

The following picture quantifies this and `it's really good` is the quick summary.  The way you can see this is because each `image` is highly localized. You basically only see a single pixel or handful of pixels with `large values`. Also, when you  ee multiple pixels with large values, they are in the same diagonal band. (It turns out that the diagonal bands are basically energy modes determined by the number of sin/cos oscillations, so we are mixing up nearly indistinguishable energy modes.) This picture would be an excellent starting point to talk more about the DCT.
"""

# ╔═╡ 224dc71b-abdf-4ad4-9821-63b1ec2021ff
dct_of_Z = mosaicview(map(i->colorview(RGB,
                      permutedims(applydivergingcolourmap(
                        dct(reshape(tU[:,i], (tilesz,tilesz))), cmap("D2"), 0),
                      (3,1,2))),
               1:16)..., nrow=4, npad=5, fillvalue=127) |> Z -> map(clamp01nan, Z) |> Z->upscale(Z,8)

# ╔═╡ 4e1f7928-3f7e-4140-a314-a56b727b5de0
md""" Just for reference (and because it's easier) here is the same picture if the result were `exact` and we used the DCT basis itself. Remember again that the sign/color is ambiguous so we can change that."""

# ╔═╡ 28279f1f-d0bc-48d5-ad95-ef0600c774f4
dct_of_Zdct = mosaicview(map(i->colorview(RGB,
                      permutedims(applydivergingcolourmap(
                        dct(idct(idct_id(i,tilesz))), cmap("D2"), 0),
                      (3,1,2))),
               1:16)..., nrow=4, npad=5, fillvalue=127) |> Z -> map(clamp01nan, Z) |> Z->upscale(Z,8)

# ╔═╡ fed9cdd4-4262-4a60-aee6-12b611c373b2
md""" ## (Under construction) Interpreting the right tiled SVD basis. 

The matrix `tV` gives the expansion."""

# ╔═╡ ba5cbe49-05da-438e-b9aa-36e77843e778
heatmap(tV';colors=cmap("D2"))

# ╔═╡ fa381b5f-fee2-4974-9b0d-0b317c6fe921
histogram(vec(tV))

# ╔═╡ 23b74436-7020-4a33-82d6-5b2cb50c2a39
histogram(vec(V))

# ╔═╡ f82f8ba0-c1ae-42d8-a63a-bf605b158fad
plot(abs.(tV[1:5,:]'), linecolor=:black, alpha=0.05, legend=false)

# ╔═╡ fd45332e-2980-474a-9237-31031c4a2c07
begin
	thresh = 0.02
	sum(x->abs(x) > thresh, tV), sum(x->abs(x) > thresh, V)
end

# ╔═╡ ddf50450-5582-4ab0-9e1f-8fe0b9739edb
md""" # Studying failure modes and towards a model.

When you see results like this, you should ask _why_ is it the case that we get the DCT result from the SVD on tiled matrices. Maybe that's the just the general answer? Here is what happens on a matrix of random values instead.

I wish I had an image that showed this instead of just random noise, but we'll work with that! 
"""

# ╔═╡ e26f91f9-82a5-461c-a26d-2671bb17fff4
begin
	# generate random values in a matrix with the same size as the tiled matrix X
	Random.seed!(0)
	Xr = rand(size(X)...)
	rU,rS,rV = svd(Xr)
end;

# ╔═╡ a0ab28cd-ff89-4678-a8c0-a75c384273db
mosaicview(map(i->colorview(RGB,
                      permutedims(applydivergingcolourmap(
                        reshape(rU[:,i], (tilesz,tilesz)), cmap("D2"), 0),
                      (3,1,2))),
               1:16)..., nrow=4, npad=5, fillvalue=127) |> Z -> map(clamp01nan, Z) |> Z -> upscale(Z,8)

# ╔═╡ 3f072542-81b2-445b-abeb-c5d7686e53a8
md""" What goes wrong is that the random data _only_ has high frequency information. So there is no low frequency inforamtion (nearly constant patches) to drive it towards a DCT-like answer."""


# ╔═╡ 0cd0586e-2efc-4f88-89e5-99f813435e3f
md"""We can create low-frequency information by bluring a little tile or patch we generate. 

- `blur_amount` $(@bind blur_amount Slider(0:3, default=1))
"""

# ╔═╡ 2e6dbdfa-71da-4eef-847c-f78879121822
begin
	Random.seed!(0)
	rand_tile = Gray.(rand(tilesz, tilesz))
	blur_tile = imfilter(rand_tile, Kernel.gaussian(blur_amount))
	mosaicview(rand_tile, blur_tile, nrow=1, npad=5, fillvalue=127) |> Z -> upscale(Z,8)
end

# ╔═╡ b7156110-cf28-4df4-8068-9b9131122076
md""" Now what we'll do is generate a matrix like X, but where each column simulates a random patch with a random amount of blurring. """

# ╔═╡ 96587e36-0ded-480d-aa51-3b1177306bd7
begin 
function random_mixed_frequency(m::Int, n::Int)
	@assert tilesz*tilesz == m # need multiple of tile-size
	R = zeros(m,n)
	for j=1:n
		T = rand(tilesz,tilesz) # generate a random tile
		# randomly blur, 0 = no blur, 3 = lots!), see above
		T = imfilter(T, Kernel.gaussian(rand(0:10))) 
		R[:,j] = vec(T)
	end
	return R
end
	
	Xr2 = random_mixed_frequency(size(X)...)
	rU2,rS2,rV2 = svd(Xr2)
end;

# ╔═╡ 5098d01f-bb09-4155-9994-c77e3e742e72
mosaicview(map(i->colorview(RGB,
                      permutedims(applydivergingcolourmap(
                        reshape(rU2[:,i], (tilesz,tilesz)), cmap("D2"), 0),
                      (3,1,2))),
               1:16)..., nrow=4, npad=5, fillvalue=127) |> Z -> map(clamp01nan, Z) |> Z -> upscale(Z,8)

# ╔═╡ c687100b-96cd-4788-8bd5-e53e952a2762
md"""
While this looks like we get the first few modes/vectors correct. We can check with the idct transform as before. 
"""

# ╔═╡ afcee1f2-f5b8-4346-87de-21afc5bbce16
mosaicview(map(i->colorview(RGB,
                      permutedims(applydivergingcolourmap(
                        dct(reshape(rU2[:,i], (tilesz,tilesz))), cmap("D2"), 0),
                      (3,1,2))),
               1:16)..., nrow=4, npad=5, fillvalue=127) |> Z -> map(clamp01nan, Z) |> Z->upscale(Z,8)

# ╔═╡ 10602dbb-d960-4488-bfff-8e4c8c8a46a7
md"""
> You can play around with this a lot. The key line above is `T = imfilter(T, Kernel.gaussian(rand(0:3)))`. See what happens if you use `T = imfilter(T, Kernel.gaussian(rand(0:1)))` to look at only a tiny bit of blurring, or `T = imfilter(T, Kernel.gaussian(rand(5:10)))` to look at tons of blurring...
"""

# ╔═╡ 655d163f-7845-430e-b784-fb0bfa7550c3
md""" # Varimax and sparse bases from SVD 

> This is a more advanced concept. Please skip it unless you want to learn a lot more yourself or you already known about things like Sparse PCA and 1-norm approximations and the modern sparsity concept. I mention it only to show how this idea can be used as a jumpstart for all sorts of ideas in applied math and signal processing and stats. 

There is a method in statistics to `find` more interpretable or `sparse` bases from an orthgonal basis called "varimax" or "orthomax". Karl Rohe pointed out some interesting comparisons to "sparse PCA" (Rohe and Zeng, arXiv 2020 <https://arxiv.org/abs/2004.05387>). Sparse PCA is designed around convex programming, 1-norms, and the like. But varimax is just a fast heurstic, but can be applied to any SVD output. So let's see what the varimax basis here is. 

"""

# ╔═╡ 32064bb3-b50a-42b0-aa2d-5cef91a25398
md""" Compute a rank-16 basis transformation via orthomax. """ 

# ╔═╡ 351840fb-81a9-47c1-8635-1c1887d7f04a
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
	tU2 = orthomax(tU[:,1:16], 1.0, 20, 1000, 1e-12)[1]
end

# ╔═╡ c5f90874-bf6e-4179-ac74-d57d2253b3a2
md"""
The varimax basis is below. We see sparsity in terms of only having small regions of nonzero values. 

The result is typically little "blobs" of nonzeros. This is reasonable. If you had to represent 8x8 images with only 16 _sparse_ images (those with only a few nonzero entries). Then having all 16 4x4 activation blocks is probably the best you can do!

The `coffee` image looks pretty good here.
"""

# ╔═╡ b617fb5b-6f20-4195-a13c-93809bf77539
Z2 = mosaicview(map(i->colorview(RGB,
                      permutedims(applydivergingcolourmap(
                        reshape(tU2[:,i], (tilesz,tilesz)), cmap("D2"), 0),
                      (3,1,2))),
               1:16)..., nrow=4, npad=5, fillvalue=127) |> Z -> map(clamp01nan, Z) |> Z-> upscale(Z,8)

# ╔═╡ 6ae2aaf4-29a3-4da8-947b-044ba3baed26
md"""
This "blob" analysis suggests that if we ask varimax/Orthomax for size-9 basis, we should get bigger blobs.

Which we do! See below. Again, this will depend on the image you are looking at. """


# ╔═╡ 5fe6838b-e244-4135-be25-1f2575b289b5
begin
	tU3 = orthomax(tU[:,1:9], 1.0, 20, 1000, 1e-12)[1]
	Z3 = mosaicview(map(i->colorview(RGB,
                      permutedims(applydivergingcolourmap(
                        reshape(tU3[:,i], (tilesz,tilesz)), cmap("D2"), 0),
                      (3,1,2))),
               1:9)..., nrow=3, npad=5, fillvalue=127) |> Z -> map(clamp01nan, Z) |> Z-> upscale(Z,8)
end

# ╔═╡ Cell order:
# ╟─b9473b23-957b-4d23-8ad5-893c6c4063cf
# ╟─49bd0281-151c-4306-bcf8-2eb8c618e04d
# ╟─cbd2ea36-7ef6-4ea8-82ac-34b3e96ed8d5
# ╟─59c37536-fff2-480a-80bc-f97125e117af
# ╠═ccbc7c67-e16a-48f1-a927-c002257fcc54
# ╟─2c8a2738-77b1-4803-91d3-9b5172e4a94a
# ╟─2ac8cb5d-ffff-4561-bc08-2ffc368338e2
# ╟─3112233a-543d-48f5-823f-5b1e1cafcb61
# ╟─03604f42-66ce-4a39-b1b1-f5fd49bfccdb
# ╟─e6e10ee8-2bcb-4743-86be-ffac6b591b96
# ╟─2d4125ac-5c10-4150-bc00-e2cfcbfaf5ad
# ╟─52d8a212-0b34-4306-b410-e3d3dce7de3f
# ╠═fd4b3fef-9df2-4253-81b4-98a9c59a0176
# ╟─13b28401-d752-4dae-9ddf-bfd59f3ab995
# ╟─08c9f59e-4408-409a-b324-36dedc005c65
# ╟─4632bc52-9106-4d05-8f6f-a189e577388d
# ╟─6a13f71e-983b-4d48-8787-8d58abbebbf5
# ╟─17dae8a4-010a-4816-b5a4-c9448d3cf8b1
# ╟─fddb68e0-adc2-4ff5-908d-e0204bf074e3
# ╠═c372842c-a529-4ad1-8edb-47d755018dbc
# ╠═dbace9f6-6b02-4e91-88a4-92fe27ec063d
# ╟─037bc839-97e0-44d9-a000-c3cb4a5dcf99
# ╟─36dccd6c-0024-4f49-88b0-f6b5b2901f5d
# ╠═822c2c32-a3e1-40a3-a98f-6fa11a660035
# ╠═74cc32bb-dc19-4a1e-9b6a-79fb782e00b3
# ╟─11a12eb5-2928-47c1-87c5-2faed7e1ba02
# ╟─eae7193c-f287-4f62-bb50-cfc998638ba9
# ╟─c3fd4712-7802-4685-aa5b-4b9d8f81aeab
# ╟─cf8ba3f6-3347-49e5-ad5a-92518a4755b7
# ╟─e953db96-42a3-4506-ad68-3e84cabc47bd
# ╠═606d5856-7051-4633-893a-c33f691dd8ac
# ╟─46d589b2-87a1-48a3-86a3-f784a17d6724
# ╟─90233a0d-c04b-4216-8b0b-9d7c1912150f
# ╟─02811d5e-7948-4cf1-9330-dd95e7bcd55a
# ╠═3632bf49-d6bf-4de0-9450-5233517db4ef
# ╟─342cc50a-81e9-4688-829a-e7018cff30d6
# ╠═25215211-7c33-41db-a327-7c2f45500b3c
# ╟─2ae1b465-2f67-483c-bd56-af0b859940f7
# ╟─c2233159-59f4-4516-859c-5c25170d53aa
# ╠═3e7902ab-4f75-4d14-901b-d2e380996c8f
# ╟─8c059291-0564-4ec7-80a0-13432962254a
# ╟─b688c31c-5394-40e8-9e6c-2f3fb64fffaf
# ╠═3d9f7a23-b5d3-49ef-ae6f-d3644ba0bcf8
# ╟─860b01bd-47e0-41f2-bd05-f5f3957b8370
# ╟─9fed5b5e-0106-484f-898c-1d664033b339
# ╟─5b99d2df-fb5a-4fcc-b5f9-f44ca5328814
# ╟─e869b69b-b79f-4bd6-bd2f-6015e22028d9
# ╠═00c0d888-f192-4ddc-9a94-5cf9624ddd26
# ╟─def9d9d3-2185-4d16-a63f-e415bb91eac3
# ╠═5a491365-da8e-43bb-987d-24fccb9a2e6d
# ╟─ade6e8f2-bd1b-4569-8efb-aae3d42ee340
# ╠═5c53bc42-5f09-482e-81a8-43c1cfa71be6
# ╟─73e63f45-1458-40ca-9ec4-bb1c00932300
# ╟─211f12bd-6d77-4d78-b486-b8f381bdde8d
# ╠═224dc71b-abdf-4ad4-9821-63b1ec2021ff
# ╟─4e1f7928-3f7e-4140-a314-a56b727b5de0
# ╠═28279f1f-d0bc-48d5-ad95-ef0600c774f4
# ╟─fed9cdd4-4262-4a60-aee6-12b611c373b2
# ╠═ba5cbe49-05da-438e-b9aa-36e77843e778
# ╠═fa381b5f-fee2-4974-9b0d-0b317c6fe921
# ╠═23b74436-7020-4a33-82d6-5b2cb50c2a39
# ╠═f82f8ba0-c1ae-42d8-a63a-bf605b158fad
# ╠═fd45332e-2980-474a-9237-31031c4a2c07
# ╟─ddf50450-5582-4ab0-9e1f-8fe0b9739edb
# ╠═e26f91f9-82a5-461c-a26d-2671bb17fff4
# ╠═a0ab28cd-ff89-4678-a8c0-a75c384273db
# ╟─3f072542-81b2-445b-abeb-c5d7686e53a8
# ╟─0cd0586e-2efc-4f88-89e5-99f813435e3f
# ╟─2e6dbdfa-71da-4eef-847c-f78879121822
# ╟─b7156110-cf28-4df4-8068-9b9131122076
# ╠═96587e36-0ded-480d-aa51-3b1177306bd7
# ╠═5098d01f-bb09-4155-9994-c77e3e742e72
# ╟─c687100b-96cd-4788-8bd5-e53e952a2762
# ╠═afcee1f2-f5b8-4346-87de-21afc5bbce16
# ╟─10602dbb-d960-4488-bfff-8e4c8c8a46a7
# ╟─655d163f-7845-430e-b784-fb0bfa7550c3
# ╠═32064bb3-b50a-42b0-aa2d-5cef91a25398
# ╠═351840fb-81a9-47c1-8635-1c1887d7f04a
# ╟─c5f90874-bf6e-4179-ac74-d57d2253b3a2
# ╠═b617fb5b-6f20-4195-a13c-93809bf77539
# ╟─6ae2aaf4-29a3-4da8-947b-044ba3baed26
# ╠═5fe6838b-e244-4135-be25-1f2575b289b5
