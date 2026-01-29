export function setupCanvas(canvasId, onDrawEnd) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  let isDrawing = false;
  const brushRadius = 1.0;

  const getCoords = (e) => {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    return {
      x: (clientX - rect.left) / (rect.width / 28),
      y: (clientY - rect.top) / (rect.height / 28),
    };
  };

  const startDrawing = () => {
    isDrawing = true;
  };

  const stopDrawing = () => {
    if (isDrawing) {
      isDrawing = false;
      onDrawEnd();
    }
  };

  const draw = (e) => {
    if (!isDrawing) return;

    if (e.touches) e.preventDefault();

    const { x, y } = getCoords(e);
    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(x, y, brushRadius, 0, Math.PI * 2);
    ctx.fill();

    onDrawEnd();
  };

  canvas.addEventListener("mousedown", startDrawing);
  window.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mousemove", draw);

  canvas.addEventListener(
    "touchstart",
    (e) => {
      e.preventDefault();
      startDrawing();
      draw(e);
    },
    { passive: false }
  );

  canvas.addEventListener(
    "touchend",
    (e) => {
      e.preventDefault();
      stopDrawing();
    },
    { passive: false }
  );

  canvas.addEventListener("touchmove", draw, { passive: false });

  return ctx;
}

export function getCenteredData(ctx, canvas) {
  const data = ctx.getImageData(0, 0, 28, 28).data;
  let minX = 28,
    maxX = 0,
    minY = 28,
    maxY = 0;
  let sumX = 0,
    sumY = 0,
    count = 0;

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const alpha = data[(y * 28 + x) * 4 + 3];
      if (alpha > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        sumX += x * alpha;
        sumY += y * alpha;
        count += alpha;
      }
    }
  }

  if (count === 0) return new Float32Array(784);

  const dx = 13.5 - sumX / count;
  const dy = 13.5 - sumY / count;

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tCtx = tempCanvas.getContext("2d");
  tCtx.translate(dx, dy);
  tCtx.drawImage(canvas, 0, 0);

  const shiftedData = tCtx.getImageData(0, 0, 28, 28).data;
  return new Float32Array(784).map((_, i) => shiftedData[i * 4 + 3] / 255.0);
}

export function updateInstructions() {
  const instructionEl = document.getElementById("instruction-text");

  const isTouchDevice =
    "ontouchstart" in window || navigator.maxTouchPoints > 0;

  if (isTouchDevice) {
    instructionEl.innerText = "Touch and drag to draw a digit";
  } else {
    instructionEl.innerText = "Click and drag to draw a digit";
  }
}
