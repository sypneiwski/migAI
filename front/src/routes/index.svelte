<script context="module">
	export const prerender = true;
</script>

<script>
	import { offset_percent, CAMERA_HEIGHT, CAMERA_WIDTH } from './const'
	import { fly } from 'svelte/transition';
	import ProgressBar from 'svelte-progress-bar'
	import axios from 'axios'
	import {onMount} from 'svelte';
	let progress;
	let shouldStop = false;
	let result = "Result: ";
	onMount(async function() {
		const img_x = 350
		const img_y = 100
		const img_w = 200
		const img_h = 200
		document.querySelector(':root').style.setProperty('--height', `-${CAMERA_HEIGHT}px`)
		let ctx = document.getElementById('video-canvas').getContext('2d')
		ctx.beginPath();
		ctx.lineWidth = "6"
		ctx.strokeStyle = "#F15412"
		ctx.rect(img_x, img_y, img_w, img_h)
		ctx.stroke()

		let camera_button = document.querySelector("#start-camera");
		let video = document.querySelector("#video");
		let click_button = document.querySelector("#click-photo");
		let canvas = document.querySelector("#canvas");

		let stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
		video.srcObject = stream;
		let taking_photos = false

		const getPhoto = () => {
			let canvas_work = document.createElement("canvas");
			canvas_work.width = CAMERA_WIDTH
			canvas_work.height = CAMERA_HEIGHT
			let ctx = canvas_work.getContext('2d')
			//ctx.setTransform(-1,0,0,1,canvas.width,0)
			ctx.drawImage(video, 0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
			let image_data =  ctx.getImageData(100, 100, 200, 200);
			let ctx_can = canvas.getContext('2d')			
			ctx_can.putImageData(image_data, 0, 0);
		}

		const getResult = () => {
			let data = new FormData()
			let image = canvas.toBlob(blob => {
				data.append('data', blob)
				axios.post('http://localhost:8000/', data,{
				headers: {'Content-Type': 'multipart/form-data'}})
					.then(res => result += res.data.result)
			}, 'image/jpeg', 1)
		}

		const startPhotos = seconds => () => {
			if (taking_photos) {
				return
			}
			taking_photos = true
			if (seconds === -1) {
				getPhoto()
				getResult()
				taking_photos = false
				return
			}
			getPhotosInLoop(seconds)()
		}
		
		const getPhotosInLoop = seconds => () => {
			let interval;
			let x = 0
			let INTERVAL_LENGTH = seconds * 100
			interval = setInterval(() => {
				x += INTERVAL_LENGTH / (seconds * 1000)
				progress.setWidthRatio(x)
				if (x >= 1.0) {
					getPhoto();
					getResult();
					getPhotosInLoop(seconds)()
					clearInterval(interval)
				}
				if (shouldStop) {
					shouldStop = false;
					taking_photos = false
					clearInterval(interval)
				}
			}, INTERVAL_LENGTH)
		}
		click_button.addEventListener('click', startPhotos(-1));
		document.getElementById('click-photo1').addEventListener('click', startPhotos(1))
		document.getElementById('click-photo3').addEventListener('click', startPhotos(3))
		document.getElementById('click-photo5').addEventListener('click', startPhotos(5))
		document.getElementById('stop').addEventListener('click', () => {
			if (taking_photos) {
				shouldStop = true
			}
		})
		document.getElementById('help').addEventListener('click', () => {
			document.getElementById('modal').style.visibility = 'visible'
			document.getElementById('modal').style.opacity = 1
			document.getElementById('greyish').style.visibility = 'visible'
			document.getElementById('greyish').style.opacity = 0.7
		})
		document.getElementById('close-modal').addEventListener('click', () => {
			document.getElementById('modal').style.opacity = 0
			document.getElementById('greyish').style.opacity = 0
			document.getElementById('modal').style.visibility = 'hidden'
			document.getElementById('greyish').style.visibility = 'hidden'
		})
		document.getElementById('color-1').value = document.documentElement.style.getPropertyValue('--primary-color')
		document.getElementById('color-2').value = document.documentElement.style.getPropertyValue('--secondary-color')
		document.getElementById('color-3').value = document.documentElement.style.getPropertyValue('--tertiary-color')
		document.getElementById('color-1').addEventListener('input', (ev) => {
			document.documentElement.style.setProperty('--primary-color', ev.target.value);
		})
		document.getElementById('color-2').addEventListener('input', (ev) => {
			document.documentElement.style.setProperty('--secondary-color', ev.target.value);
		})
		document.getElementById('color-3').addEventListener('input', (ev) => {
			document.documentElement.style.setProperty('--tertiary-color', ev.target.value);
		})
		let ctrl = false;
		document.body.addEventListener('keydown', (ev) => {
			if (ev.which == "17") {
				ctrl = true;
			}
			else if (ev.which == "86" && ctrl) {
				document.getElementById('color-1').style.visibility = 'visible';
				document.getElementById('color-2').style.visibility = 'visible';
				document.getElementById('color-3').style.visibility = 'visible';
			}
		})
		document.body.addEventListener('keyup', (ev) => {
			if (ev.which == "17") {
				ctrl = false;
			}
		})
	})
</script>

<svelte:head>
	<title>Home</title>
</svelte:head>

<section>
	<header>
		<h1>migAI</h1>
		<input type='color' class='color-pick' id='color-1'/>
		<input type='color' class='color-pick' id='color-2'/>
		<input type='color' class='color-pick' id='color-3'/>
		<p id="help">Help</p>
	</header>
	<div id='greyish'></div>
	<div class="infoContent form-content" id='modal'>
		<h2>MigAI, version 1.0</h2>
		<p>MigAI is a website designed to recognize selected signs from Polish Sign Language.</p>
		<p>Usage: </p>
		<p>-show the sign in front of your camera and make sure it is placed inside the selected area</p>
		<p>-click one of the buttons to make photo once or every fixed amount of seconds</p>
		<p>-the computed result will appear next to the 'Result' label</p>
    <p>-if you make a mistake, you can edit the rendered result manually</p>
		<button id='close-modal'>Close</button>
	</div>
	<canvas id="video-canvas" width={CAMERA_WIDTH} height={CAMERA_HEIGHT}></canvas>
	<video id="video" width={CAMERA_WIDTH} height={CAMERA_HEIGHT} autoplay></video>
	<div class="flex">
	<button id="click-photo">Click Photo</button>
	<button id="click-photo1">Click Photo every 1s</button>
	<button id="click-photo3">Click Photo every 3s</button>
	<button id="click-photo5">Click Photo every 5s</button>
	</div>
	<div class="flex">
	<button id="stop">Stop</button>
	<button id="clear" on:click={() => result = "Result: "}>Clear</button>
	</div>
	<p id="result" bind:textContent="{result}" contenteditable in:fly="{{ y: 200, duration: 2000 }}"></p>
	<canvas id="canvas" width={200} height={200}></canvas>
	<ProgressBar bind:this={progress} width="100vw" color="#D01B1B"/>
</section>

<style>
	#canvas {
		transform: scaleX(-1);
		visibility: hidden;
	}
	#video {
		transform: scaleX(-1);
		z-index: -1;
	}
	.color-pick {
		border: 0;
		padding: 0;
		visibility: hidden;
	}
	#greyish {
		width: 100vw;
		height: 100vh;
		top: 0;
		background-color: #919090;
		opacity: 0;
		position: fixed;
		visibility: hidden;
		z-index: 1;
		transition: all 2s;
	}
	  .infoContent {
		transition: all 2s;
		position: fixed;
        width: 30%;
        height: 70%;
        top: 15%;
		text-align: center;
		z-index: 1;
        background-color: whitesmoke;
		padding: 10px 20px;
		border-radius: 1rem;
		opacity: 0;
		visibility: hidden;
      }
      .infoContent > p {
        border: none;
        font-size: 1.5rem;
		color: #000000;
		text-align: left;
      }
      .infoContent > h2 {
		text-align: center;
        border: none;
        font-size: 3rem;
		color: #F15412;
      }
#help {
	display: inline-block;
	margin: 0 5vw;
	line-height: 2rem;
	font-size: 1.6rem;
	cursor: pointer;
	color: #fefefe;
}

header {
	background: var(--primary-color);
	width: 100vw;
	margin-bottom: 1rem;
	height: 3rem;
	display: flex;
	justify-content: space-between;
}
h1{
 margin-top: 0.2rem;
 color: #F15412;
 display:block;
 font-size:2rem;
 font-weight:normal;
 letter-spacing:0.1rem;
 line-height:2rem;
 padding-left: 5vw;

 }

	.flex {
		display: flex;
	}
	#video {
		border-radius: 20px;
	}

	p {
 font-family:Georgia,serif;
 color: #000;;
 font-size:30px;
 line-height:1em;
 }

	#video-canvas {
		margin-top: var(--height);
		transform: translateY(100%);
	}
	section {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		flex: 1;
	}


button {
	margin: 0.5rem;
  /*align-items: center;*/
  background-color: #fee6e3;
  border: 2px solid #111;
  border-radius: 8px;
  /*box-sizing: border-box;*/
  color: #111;
  cursor: pointer;
  /*display: flex;*/
  font-family: Inter,sans-serif;
  font-size: 1.3rem;
  height: 3rem;
  /*justify-content: center;*/
  line-height: 1.3rem;
  max-width: 100%;
  padding: 0 1rem;
  position: relative;
/*  text-align: center;*/
  text-decoration: none;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
}

button:after {
  background-color: #111;
  border-radius: 8px;
  content: "";
  display: block;
  height: 48px;
  left: 0;
  width: 100%;
  position: absolute;
  top: -2px;
  transform: translate(8px, 8px);
  transition: transform .2s ease-out;
  z-index: -1;
}

button:hover:after {
  transform: translate(0, 0);
}

button:active {
  background-color: #ffdeda;
  outline: 0;
}

button:hover {
  outline: 0;
}

@media (min-width: 768px) {
  button {
    padding: 0 40px;
  }
}
</style>
