<script context="module">
	export const prerender = true;
</script>

<script>
	let offset_percent = 0.15;
	let CAMERA_WIDTH = 640
	let CAMERA_HEIGHT = 480
	import { fly } from 'svelte/transition';
	import ProgressBar from 'svelte-progress-bar'
	import axios from 'axios'
	let progress;
	let shouldStop = false;
	let result = "Result: ";
	import {onMount} from 'svelte';
	onMount(async function() {
		document.querySelector(':root').style.setProperty('--height', `-${CAMERA_HEIGHT}px`)
		let ctx = document.getElementById('video-canvas').getContext('2d')
	ctx.beginPath();
	ctx.lineWidth = "6"
	ctx.strokeStyle = "#A80C1E"
	ctx.rect(CAMERA_WIDTH * offset_percent, CAMERA_HEIGHT * offset_percent, CAMERA_WIDTH * (1 - 2 * offset_percent), CAMERA_HEIGHT * (1 - 2 * offset_percent))
	ctx.stroke()

let camera_button = document.querySelector("#start-camera");
let video = document.querySelector("#video");
let click_button = document.querySelector("#click-photo");
let canvas = document.querySelector("#canvas");

	let stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
	video.srcObject = stream;

	const getPhoto = () => {
		let canvas_work = document.createElement("canvas");
		canvas_work.width = CAMERA_WIDTH
		canvas_work.height = CAMERA_HEIGHT
		canvas_work.getContext('2d').drawImage(video, 0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
		let image_data =  canvas_work.getContext('2d').getImageData(CAMERA_WIDTH * offset_percent, CAMERA_HEIGHT * offset_percent, CAMERA_WIDTH * (1 - 2 * offset_percent), CAMERA_HEIGHT * (1 - 2 * offset_percent));
		canvas.getContext('2d').putImageData(image_data, 0, 0);

	}
	const getResult = () => {
		let data = new FormData()
		let image = canvas.toBlob(blob => {
			data.append('data', blob)
			axios.post('http://localhost:8000/', data,{
			headers: {'Content-Type': 'multipart/form-data'}})
				.then(res => result += res.data.result)
		})
	}
	let taking_photos = false
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
		setTimeout(() => {
			getPhotosInLoop(seconds)()
			taking_photos = false
		}, seconds * 1000)
	}
	const getPhotosInLoop = seconds => () => {
		let interval;
		if (!shouldStop) {
			getPhoto();
			getResult();
			let x = 0
			let INTERVAL_LENGTH = seconds * 100
			interval = setInterval(() => {
				x += INTERVAL_LENGTH / (seconds * 1000)
				progress.setWidthRatio(x)
				if (x >= 1.0 || shouldStop) {
					clearInterval(interval)
				}
			}, INTERVAL_LENGTH)
			
			setTimeout(getPhotosInLoop(seconds), seconds * 1000)
		}
		else {
			shouldStop = false;
		}
	}
click_button.addEventListener('click', startPhotos(-1));
document.getElementById('click-photo1').addEventListener('click', startPhotos(1))
document.getElementById('click-photo3').addEventListener('click', startPhotos(3))
document.getElementById('click-photo5').addEventListener('click', startPhotos(5))
document.getElementById('stop').addEventListener('click', () => shouldStop = true)
	}
	)
</script>

<svelte:head>
	<title>Home</title>
</svelte:head>

<section>
	<h1>migAI</h1>
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
	<canvas id="canvas" width={CAMERA_WIDTH} height={CAMERA_HEIGHT}></canvas>
	<ProgressBar bind:this={progress} width="100vw" color="#A80C1E"/>
</section>

<style>
h1{
 color:#A80C1E;
 display:block;
 font-size:3rem;
 font-weight:normal;
 letter-spacing:0.1rem;
 line-height:1rem;
 }

	.flex {
		display: flex;
	}
	#video {
		border-radius: 20px;
	}

	p {
 font-family:Georgia,serif;
 color:#666666;
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
  font-size: 1rem;
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
