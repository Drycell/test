function drawCurve(canvas, y) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!y || y.length === 0) return;
  const minY = Math.min(...y);
  const maxY = Math.max(...y);
  const yr = Math.max(1e-9, maxY - minY);
  ctx.beginPath();
  y.forEach((v, i) => {
    const x = (i / Math.max(1, y.length - 1)) * (w - 20) + 10;
    const yy = h - 10 - ((v - minY) / yr) * (h - 20);
    if (i === 0) ctx.moveTo(x, yy); else ctx.lineTo(x, yy);
  });
  ctx.strokeStyle = '#1f77b4';
  ctx.lineWidth = 2;
  ctx.stroke();
}

fetch('run_data.json').then(r => r.json()).then(data => {
  document.getElementById('summary').textContent = JSON.stringify(data.summary, null, 2);
  drawCurve(document.getElementById('fitness'), data.fitness_curve || []);

  const slider = document.getElementById('step');
  const frame = document.getElementById('frame');
  const traj = data.trajectory || [];
  slider.max = String(Math.max(0, traj.length - 1));
  const render = () => {
    const i = Number(slider.value || 0);
    frame.textContent = JSON.stringify({ step: i, obs: traj[i] || [] }, null, 2);
  };
  slider.addEventListener('input', render);
  render();
});
