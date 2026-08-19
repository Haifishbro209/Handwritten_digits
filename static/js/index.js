(() => {
	const canvas = document.getElementById('board');
	const ctx = canvas.getContext('2d', { willReadFrequently: false });
	const clearBtn = document.getElementById('clearBtn');
	const STROKE_STYLE = '#000';
	const LINE_WIDTH = 2;
	const LINE_CAP = 'round';
	const LINE_JOIN = 'round';
	let drawing = false;
	let lastX = 0;
	let lastY = 0;

	function fillWhiteBackground() {
		ctx.save();
		ctx.globalCompositeOperation = 'source-over';
		ctx.fillStyle = '#fff';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		ctx.restore();
	}

	function setupContext() {
		ctx.lineCap = LINE_CAP;
		ctx.lineJoin = LINE_JOIN;
		ctx.strokeStyle = STROKE_STYLE;
		ctx.lineWidth = LINE_WIDTH;
	}

	function getPosFromEvent(e) {
		const rect = canvas.getBoundingClientRect();
		let clientX, clientY;
		if (e.touches && e.touches.length) {
			clientX = e.touches[0].clientX;
			clientY = e.touches[0].clientY;
		} else {
			clientX = e.clientX;
			clientY = e.clientY;
		}
		const x = ((clientX - rect.left) / rect.width) * canvas.width;
		const y = ((clientY - rect.top) / rect.height) * canvas.height;
		return { x, y };
	}

	function startDraw(e) {
		e.preventDefault();
		drawing = true;
		const { x, y } = getPosFromEvent(e);
		lastX = x; lastY = y;
	}

	function draw(e) {
		if (!drawing) return;
		e.preventDefault();
		const { x, y } = getPosFromEvent(e);
		ctx.beginPath();
		ctx.moveTo(lastX, lastY);
		ctx.lineTo(x, y);
		ctx.stroke();
		lastX = x; lastY = y;
	}

	function endDraw(e) {
		if (!drawing) return;
		e && e.preventDefault();
		drawing = false;
	}

	canvas.addEventListener('mousedown', startDraw);
	window.addEventListener('mousemove', draw);
	window.addEventListener('mouseup', endDraw);
	canvas.addEventListener('mouseleave', endDraw);
	canvas.addEventListener('touchstart', startDraw, { passive: false });
	canvas.addEventListener('touchmove', draw, { passive: false });
	canvas.addEventListener('touchend', endDraw, { passive: false });
	canvas.addEventListener('touchcancel', endDraw, { passive: false });

	clearBtn?.addEventListener('click', clearCanvas);
	const sendBtn = document.getElementById('sendBtn');
	sendBtn?.addEventListener('click', sendCanvas);

	function clearCanvas() {
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		fillWhiteBackground();
		setupContext();
	}

	async function sendCanvas() {
		const w = canvas.width;
		const h = canvas.height;
		const imgData = ctx.getImageData(0, 0, w, h);
		const data = imgData.data;
		const pixels = new Array(w * h);
		for (let y = 0; y < h; y++) {
			for (let x = 0; x < w; x++) {
				const i = (y * w + x) * 4;
				const r = data[i], g = data[i + 1], b = data[i + 2];
				const lum = 0.299 * r + 0.587 * g + 0.114 * b;
				pixels[y * w + x] = lum < 128 ? 1.0 : 0.0;
			}
		}
		const resp = await fetch('/upload', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ width: w, height: h, pixels }),
		});
		if (!resp.ok) throw new Error('Upload fehlgeschlagen');
		const result = await resp.json();
		alert(result.message || 'Erfolgreich gesendet');
		return result;
	}

	setupContext();
	fillWhiteBackground();
})();
