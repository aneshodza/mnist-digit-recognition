export function setupCanvas(canvasId, onDrawEnd) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  let isDrawing = false;
  const brushRadius = 1.0;

  canvas.onmousedown = () => (isDrawing = true);
  window.onmouseup = () => {
    if (isDrawing) {
      isDrawing = false;
      onDrawEnd();
    }
  };

  canvas.onmousemove = (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / (rect.width / 28);
    const y = (e.clientY - rect.top) / (rect.height / 28);
    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(x, y, brushRadius, 0, Math.PI * 2);
    ctx.fill();
  };

  return ctx;
}

export function getCenteredData(ctx, canvas) {
  const data = ctx.getImageData(0, 0, 28, 28).data;
  let minX = 28, maxX = 0, minY = 28, maxY = 0;
  let sumX = 0, sumY = 0, count = 0;

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const alpha = data[(y * 28 + x) * 4 + 3];
      if (alpha > 0) {
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        sumX += x * alpha; sumY += y * alpha; count += alpha;
      }
    }
  }

  if (count === 0) return new Float32Array(784);

  const dx = 13.5 - sumX / count;
  const dy = 13.5 - sumY / count;

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28; tempCanvas.height = 28;
  const tCtx = tempCanvas.getContext("2d");
  tCtx.translate(dx, dy);
  tCtx.drawImage(canvas, 0, 0);

  const shiftedData = tCtx.getImageData(0, 0, 28, 28).data;
  return new Float32Array(784).map((_, i) => shiftedData[i * 4 + 3] / 255.0);
}
