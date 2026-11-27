/**
 * First-person POV Motorcycle Handlebar, Tank, Mirrors & Glowing Cyan Digital Speedometer
 * modeled closely on Reference.png (Royal Enfield Classic style cockpit).
 */
export class MotoCockpit {
  draw(ctx, W, H, state) {
    const {
      speedKmh = 0,
      rollRad = 0,
      systemOn = false,
      helmetDetected = false,
      activeGesture = "NONE",
      timeSec = 0,
    } = state;

    const isBraking = activeGesture === "BOTH_FISTS";
    const isThrottle = activeGesture === "BOTH_OPEN" && systemOn;

    // Subtle speed vibration
    const vibAmp = systemOn ? Math.min(2.2, 0.3 + (speedKmh / 85) * 1.9) : 0;
    const vibX = Math.sin(timeSec * 42) * vibAmp * 0.5;
    const vibY = Math.cos(timeSec * 55) * vibAmp;

    const pivotX = W * 0.5 + vibX;
    const pivotY = H * 1.03 + vibY;
    const scale = Math.min(W / 1280, H / 720) * 1.04;

    ctx.save();
    ctx.translate(pivotX, pivotY);
    ctx.rotate(rollRad);
    ctx.scale(scale, scale);

    // 1. Matte olive-grey fuel tank at bottom center
    const tankGrad = ctx.createRadialGradient(0, -40, 20, 0, 10, 220);
    tankGrad.addColorStop(0, "#7c7a6e");
    tankGrad.addColorStop(0.55, "#5a594f");
    tankGrad.addColorStop(1, "#373630");
    ctx.fillStyle = tankGrad;
    ctx.beginPath();
    ctx.ellipse(0, -18, 190, 96, 0, Math.PI, Math.PI * 2);
    ctx.lineTo(190, 60);
    ctx.lineTo(-190, 60);
    ctx.closePath();
    ctx.fill();

    // Chrome fuel cap on tank
    const capGrad = ctx.createLinearGradient(-28, -90, 28, -45);
    capGrad.addColorStop(0, "#e9edf0");
    capGrad.addColorStop(0.5, "#8d949c");
    capGrad.addColorStop(1, "#f2f5f7");
    ctx.fillStyle = capGrad;
    ctx.beginPath();
    ctx.ellipse(0, -68, 30, 17, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#3c4046";
    ctx.lineWidth = 2;
    ctx.stroke();

    // 2. Twin amber turn indicator pods flanking fork neck
    for (const side of [-1, 1]) {
      const ix = side * 124;
      const iy = -104;
      // Chrome pod shell
      ctx.fillStyle = "#b8c0c8";
      ctx.beginPath();
      ctx.ellipse(ix, iy, 22, 26, side * 0.25, 0, Math.PI * 2);
      ctx.fill();
      // Amber lens
      ctx.fillStyle = systemOn ? "#e07a14" : "#9c5512";
      ctx.beginPath();
      ctx.ellipse(ix + side * 2, iy - 12, 16, 9, side * 0.15, 0, Math.PI * 2);
      ctx.fill();
    }

    // 3. Black headlight nacelle / instrument housing
    ctx.fillStyle = "#16181c";
    ctx.beginPath();
    ctx.roundRect(-118, -282, 236, 172, 68);
    ctx.fill();
    ctx.strokeStyle = "#363b42";
    ctx.lineWidth = 3;
    ctx.stroke();

    // 4. Rear-view mirrors rising from handlebar mounts
    this._drawMirror(ctx, -275, -315, -1, speedKmh);
    this._drawMirror(ctx, 275, -315, 1, speedKmh);

    // 5. Chrome handlebars sweeping left & right
    this._drawHandlebars(ctx, isBraking, isThrottle, systemOn);

    // 6. Central chrome risers + steering stem hex nut
    ctx.fillStyle = "#c9d0d6";
    ctx.beginPath();
    ctx.roundRect(-56, -198, 22, 38, 6);
    ctx.roundRect(34, -198, 22, 38, 6);
    ctx.fill();

    // Hex steering nut
    ctx.fillStyle = "#d8dfe5";
    ctx.beginPath();
    ctx.arc(0, -182, 19, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#6c747c";
    ctx.lineWidth = 3;
    ctx.stroke();

    // 7. Ignition switch + hanging key (left side of cluster)
    ctx.fillStyle = "#22252a";
    ctx.beginPath();
    ctx.arc(-62, -195, 18, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#cfd6dc";
    ctx.lineWidth = 3;
    ctx.stroke();
    // Key head + ring
    ctx.fillStyle = "#101214";
    ctx.beginPath();
    ctx.roundRect(-72, -192, 18, 28, 4);
    ctx.fill();

    // 8. Secondary small fuel/trip circular sub-gauge (bottom-right of cluster)
    ctx.fillStyle = "#12161b";
    ctx.beginPath();
    ctx.arc(62, -195, 21, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#cfd6dc";
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.fillStyle = systemOn ? "#26d9ff" : "#3a4a56";
    ctx.font = "bold 9px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(helmetDetected ? "SAFE" : "CAP", 62, -192);

    // 9. Center Round Glowing Cyan Digital Speedometer
    this._drawSpeedometer(ctx, 0, -256, speedKmh, systemOn, helmetDetected);

    ctx.restore();
  }

  _drawMirror(ctx, mx, my, side, speedKmh) {
    // Chrome stalk
    ctx.strokeStyle = "#bcc4cc";
    ctx.lineWidth = 7;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(side * 210, -172);
    ctx.quadraticCurveTo(side * 245, -245, mx, my);
    ctx.stroke();

    // Mirror outer housing
    ctx.fillStyle = "#1d2024";
    ctx.beginPath();
    ctx.ellipse(mx, my, 54, 30, side * -0.1, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#aeb6be";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Mirror glass reflection (sky + passing road hint)
    const refGrad = ctx.createLinearGradient(mx, my - 22, mx, my + 22);
    refGrad.addColorStop(0, "#b8c8d4");
    refGrad.addColorStop(0.55, "#6b737c");
    refGrad.addColorStop(1, "#3f444a");
    ctx.fillStyle = refGrad;
    ctx.beginPath();
    ctx.ellipse(mx, my, 47, 23, side * -0.1, 0, Math.PI * 2);
    ctx.fill();

    if (speedKmh > 10) {
      ctx.strokeStyle = "rgba(255,255,255,0.22)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(mx - 20, my + 12);
      ctx.lineTo(mx + 20, my - 4);
      ctx.stroke();
    }
  }

  _drawHandlebars(ctx, isBraking, isThrottle, systemOn) {
    const chromeGrad = ctx.createLinearGradient(0, -205, 0, -110);
    chromeGrad.addColorStop(0, "#f2f5f8");
    chromeGrad.addColorStop(0.45, "#9da5ae");
    chromeGrad.addColorStop(1, "#e2e7ec");

    // Left & right handlebar chrome tubes
    for (const side of [-1, 1]) {
      ctx.strokeStyle = "#2b3036";
      ctx.lineWidth = 26;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(side * 38, -182);
      ctx.bezierCurveTo(side * 125, -185, side * 145, -142, side * 230, -155);
      ctx.lineTo(side * 390, -130);
      ctx.stroke();

      ctx.strokeStyle = chromeGrad;
      ctx.lineWidth = 20;
      ctx.beginPath();
      ctx.moveTo(side * 38, -182);
      ctx.bezierCurveTo(side * 125, -185, side * 145, -142, side * 230, -155);
      ctx.lineTo(side * 315, -142);
      ctx.stroke();
    }

    // Brake & Clutch levers (pull inward slightly when braking)
    const pull = isBraking ? 14 : 0;
    for (const side of [-1, 1]) {
      ctx.strokeStyle = "#17191d";
      ctx.lineWidth = 10;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(side * 245, -180);
      ctx.quadraticCurveTo(side * 330, -198 + pull, side * 418, -172 + pull * 1.4);
      ctx.stroke();
      // Lever ball end
      ctx.fillStyle = "#1f2228";
      ctx.beginPath();
      ctx.arc(side * 422, -171 + pull * 1.4, 7, 0, Math.PI * 2);
      ctx.fill();
    }

    // Right brake master cylinder reservoir (matching Reference.png)
    ctx.fillStyle = "#1f2227";
    ctx.beginPath();
    ctx.roundRect(210, -214, 58, 30, 5);
    ctx.fill();
    ctx.strokeStyle = "#3f454e";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Left switchgear block
    ctx.fillStyle = "#181a1f";
    ctx.beginPath();
    ctx.roundRect(-305, -180, 62, 60, 10);
    ctx.fill();
    ctx.fillStyle = "#32373f";
    ctx.fillRect(-292, -162, 16, 14);

    // Right switchgear block + prominent RED engine kill switch
    ctx.fillStyle = "#181a1f";
    ctx.beginPath();
    ctx.roundRect(243, -180, 64, 60, 10);
    ctx.fill();

    // Red kill switch rocker
    ctx.fillStyle = systemOn ? "#e03131" : "#ff4444";
    ctx.beginPath();
    ctx.roundRect(262, -170, 26, 22, 4);
    ctx.fill();
    if (!systemOn) {
      ctx.strokeStyle = "#ff8585";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Left & right textured rubber handgrips
    for (const side of [-1, 1]) {
      ctx.save();
      ctx.translate(side < 0 ? -308 : 308, -142);
      ctx.rotate(side * 0.14);
      const gripGrad = ctx.createLinearGradient(0, -18, 0, 18);
      gripGrad.addColorStop(0, "#2f333a");
      gripGrad.addColorStop(0.5, "#17191d");
      gripGrad.addColorStop(1, "#0d0e11");
      ctx.fillStyle = gripGrad;
      ctx.beginPath();
      ctx.roundRect(side < 0 ? -156 : 0, -17, 156, 34, 8);
      ctx.fill();

      // Ribbed grip rings
      ctx.strokeStyle = isThrottle && side > 0 ? "rgba(34,230,255,0.45)" : "rgba(255,255,255,0.09)";
      ctx.lineWidth = 2;
      for (let r = 14; r < 142; r += 12) {
        const rx = side < 0 ? -r : r;
        ctx.beginPath();
        ctx.moveTo(rx, -15);
        ctx.lineTo(rx, 15);
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  _drawSpeedometer(ctx, cx, cy, speedKmh, systemOn, helmetDetected) {
    const r = 56;
    // Outer bright chrome bezel
    const bezelGrad = ctx.createLinearGradient(cx - r, cy - r, cx + r, cy + r);
    bezelGrad.addColorStop(0, "#eef2f6");
    bezelGrad.addColorStop(0.35, "#7d8690");
    bezelGrad.addColorStop(0.7, "#f5f8fa");
    bezelGrad.addColorStop(1, "#555d66");
    ctx.fillStyle = bezelGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();

    // Inner dark glass face
    ctx.fillStyle = "#08121b";
    ctx.beginPath();
    ctx.arc(cx, cy, r - 7, 0, Math.PI * 2);
    ctx.fill();

    // Cyan ambient glow inside dial
    const dialGlow = ctx.createRadialGradient(cx, cy, 4, cx, cy, r - 8);
    dialGlow.addColorStop(0, systemOn ? "rgba(18, 210, 255, 0.25)" : "rgba(40, 70, 90, 0.14)");
    dialGlow.addColorStop(1, "rgba(6, 15, 24, 0.0)");
    ctx.fillStyle = dialGlow;
    ctx.beginPath();
    ctx.arc(cx, cy, r - 8, 0, Math.PI * 2);
    ctx.fill();

    // Segmented cyan speed arc (from 135 deg to 405 deg)
    const startAng = Math.PI * 0.78;
    const totalSpan = Math.PI * 1.44;
    const maxDisplaySpeed = 100;
    const ratio = Math.max(0, Math.min(1, speedKmh / maxDisplaySpeed));

    // Background track arc
    ctx.strokeStyle = "rgba(60, 120, 145, 0.30)";
    ctx.lineWidth = 6.5;
    ctx.lineCap = "butt";
    ctx.beginPath();
    ctx.arc(cx, cy, r - 15, startAng, startAng + totalSpan);
    ctx.stroke();

    // Active glowing cyan arc
    if (systemOn && ratio > 0.005) {
      ctx.save();
      ctx.shadowColor = "#00e5ff";
      ctx.shadowBlur = 12;
      const arcGrad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
      arcGrad.addColorStop(0, "#1ee6ff");
      arcGrad.addColorStop(0.75, "#38b6ff");
      arcGrad.addColorStop(1, helmetDetected ? "#00ff9d" : "#ff6b3d");
      ctx.strokeStyle = arcGrad;
      ctx.lineWidth = 6.5;
      ctx.beginPath();
      ctx.arc(cx, cy, r - 15, startAng, startAng + totalSpan * ratio);
      ctx.stroke();
      ctx.restore();
    }

    // Digital numeric readout ("0" when stationary/off)
    const displayNum = Math.round(speedKmh);
    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    if (systemOn) {
      ctx.shadowColor = "rgba(30, 230, 255, 0.85)";
      ctx.shadowBlur = 10;
      ctx.fillStyle = "#dffaff";
    } else {
      ctx.fillStyle = "#5c7585";
    }
    ctx.font = "700 31px 'Orbitron', 'Rajdhani', monospace";
    ctx.fillText(String(displayNum), cx, cy + 1);

    // "km/h" micro label
    ctx.shadowBlur = 0;
    ctx.fillStyle = systemOn ? "#79e2f5" : "#4b606e";
    ctx.font = "600 9.5px 'Inter', sans-serif";
    ctx.fillText("km/h", cx, cy + 23);

    // Top tiny status dot
    ctx.fillStyle = !systemOn ? "#ff4d4d" : helmetDetected ? "#2ce68a" : "#ff9933";
    ctx.beginPath();
    ctx.arc(cx, cy - 26, 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}
