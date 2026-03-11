import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

let playing = false;
let timer = null;

function drawFitnessChart(canvas, y) {
  new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: { labels: y.map((_, i) => i), datasets: [{ label: 'best_fitness', data: y, borderColor: '#1f77b4', pointRadius: 0 }] },
    options: { responsive: false, animation: false, plugins: { legend: { display: false } } },
  });
}

function buildScene(container) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.001, 10);
  camera.position.set(0.01, -0.02, 0.015);
  camera.lookAt(0, 0, 0.005);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dl = new THREE.DirectionalLight(0xffffff, 1.0);
  dl.position.set(0.02, -0.01, 0.03);
  scene.add(dl);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(0.05, 0.05), new THREE.MeshStandardMaterial({ color: 0x223322 }));
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.00004, 0.001, 6, 12), new THREE.MeshStandardMaterial({ color: 0xffcc66 }));
  torso.rotation.z = Math.PI / 2;
  scene.add(torso);

  const wingGeom = new THREE.BoxGeometry(0.0005, 0.00002, 0.0005);
  const wingMat = new THREE.MeshStandardMaterial({ color: 0x66ccff });
  const leftWing = new THREE.Mesh(wingGeom, wingMat);
  const rightWing = new THREE.Mesh(wingGeom, wingMat);
  leftWing.position.set(0, 0.0001, 0);
  rightWing.position.set(0, -0.0001, 0);
  torso.add(leftWing);
  torso.add(rightWing);

  const animate = () => renderer.render(scene, camera);
  return { torso, leftWing, rightWing, animate };
}

fetch('run_data.json').then(r => r.json()).then(data => {
  document.getElementById('summary').textContent = JSON.stringify(data.summary, null, 2);
  drawFitnessChart(document.getElementById('fitness'), data.fitness_curve || []);

  const traj = data.trajectory || [];
  const slider = document.getElementById('step');
  const frame = document.getElementById('frame');
  const stepLabel = document.getElementById('stepLabel');
  slider.max = String(Math.max(0, traj.length - 1));

  const scene = buildScene(document.getElementById('scene'));

  const renderFrame = (i) => {
    const obs = traj[i] || [];
    const roll = obs[1] || 0;
    const pitch = obs[2] || 0;
    const height = obs[8] || 0;
    const wl = obs[9] || 0;
    const wr = obs[10] || 0;

    scene.torso.position.set((obs[14] || 0), 0, height);
    scene.torso.rotation.set(roll, pitch, Math.PI / 2);
    scene.leftWing.rotation.x = wl;
    scene.rightWing.rotation.x = wr;
    scene.animate();

    stepLabel.textContent = String(i);
    frame.textContent = JSON.stringify({ step: i, obs }, null, 2);
  };

  slider.addEventListener('input', () => renderFrame(Number(slider.value || 0)));
  document.getElementById('play').addEventListener('click', () => {
    playing = !playing;
    if (playing) {
      timer = setInterval(() => {
        const next = (Number(slider.value || 0) + 1) % Math.max(1, traj.length);
        slider.value = String(next);
        renderFrame(next);
      }, 40);
    } else if (timer) {
      clearInterval(timer);
      timer = null;
    }
  });

  renderFrame(0);
});
