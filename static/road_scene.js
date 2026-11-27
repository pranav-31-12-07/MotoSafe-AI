/**
 * Urban Indian 3-lane street & traffic renderer modeled on Reference.png.
 */
export class RoadScene {
  constructor() {
    this.scrollDist = 0; // meters traveled
    this.vehicles = [];
    this.roadsideProps = [];
    this.spawnTimer = 0;
    this.collisionFlash = 0;
    this._initRoadside();
    this._initTraffic();
  }

  _initRoadside() {
    const buildingColors = [
      { wall: "#d7cbb8", trim: "#b7a994", win: "#3c434a" },
      { wall: "#e2ded5", trim: "#c3beb4", win: "#353c43" },
      { wall: "#c7c3ba", trim: "#9e9a92", win: "#2f363d" },
      { wall: "#d9c8b4", trim: "#a89681", win: "#383f45" },
    ];
    for (let i = 0; i < 14; i++) {
      const z = (i + 0.5) / 14;
      this.roadsideProps.push({
        z,
        leftBld: buildingColors[i % buildingColors.length],
        rightBld: buildingColors[(i + 2) % buildingColors.length],
        leftFloors: 3 + (i % 3),
        rightFloors: 3 + ((i + 1) % 3),
        hasPole: i % 2 === 0,
      });
    }
  }

  _initTraffic() {
    // Seed initial traffic resembling Reference.png (left hatchback, auto-rickshaw, right silver SUV)
    this.vehicles = [
      { lane: 0, z: 0.44, speedKmh: 36, type: "hatchback", color: "#5e646c" },
      { lane: 0.78, z: 0.36, speedKmh: 31, type: "autorickshaw", color: "#f2b705" },
      { lane: 2, z: 0.46, speedKmh: 40, type: "suv", color: "#cfd6dc" },
      { lane: 1, z: 0.16, speedKmh: 44, type: "sedan", color: "#e8ecef" },
    ];
  }

  update(dt, playerSpeedKmh, playerLanePos) {
    // Convert km/h to normalized world distance per second
    const speedMps = playerSpeedKmh / 3.6;
    this.scrollDist += speedMps * dt;
    this.collisionFlash = Math.max(0, this.collisionFlash - dt * 2.4);

    // Move roadside scenery toward camera
    const propSpeed = (speedMps / 150) * dt;
    for (const p of this.roadsideProps) {
      p.z += propSpeed;
      if (p.z > 1.08) {
        p.z -= 1.08;
      }
    }

    // Move traffic relative to player speed
    for (let i = this.vehicles.length - 1; i >= 0; i--) {
      const v = this.vehicles[i];
      const relKmh = playerSpeedKmh - v.speedKmh;
      // Positive relKmh -> player catches up (z increases toward 1.0)
      v.z += (relKmh / 145) * dt;

      // Gentle collision detection in foreground zone
      if (v.z > 0.72 && v.z < 0.87 && Math.abs(v.lane - playerLanePos) < 0.44) {
        this.collisionFlash = 1.0;
        // Push vehicle slightly ahead / clear lane gently so rider isn't trapped
        v.z = 0.69;
        v.speedKmh = Math.max(v.speedKmh, playerSpeedKmh + 8);
      }

      if (v.z > 1.15 || v.z < 0.01) {
        this.vehicles.splice(i, 1);
      }
    }

    // Keep 4–6 vehicles smoothly populated across the 3 lanes
    this.spawnTimer -= dt;
    if (this.vehicles.length < 5 && this.spawnTimer <= 0) {
      this.spawnTimer = 1.1 + Math.random() * 1.1;
      const lane = Math.floor(Math.random() * 3);
      const types = ["autorickshaw", "hatchback", "suv", "sedan"];
      const type = types[Math.floor(Math.random() * types.length)];
      const spawnZ = playerSpeedKmh > 38 ? 0.05 : 0.72;

      // Avoid overlapping an existing vehicle in same lane
      const crowded = this.vehicles.some(
        (v) => Math.abs(v.lane - lane) < 0.55 && Math.abs(v.z - spawnZ) < 0.16
      );
      if (!crowded) {
        const speedByType = {
          autorickshaw: 26 + Math.random() * 9,
          hatchback: 34 + Math.random() * 14,
          suv: 38 + Math.random() * 16,
          sedan: 36 + Math.random() * 15,
        };
        this.vehicles.push({
          lane,
          z: spawnZ,
          speedKmh: speedByType[type],
          type,
          color: type === "hatchback" ? "#5a6068" : type === "suv" ? "#d3d9df" : "#eceff2",
        });
      }
    }
  }

  /**
   * Project normalized road coords (laneOffset in [-1.5..1.5], depth z in [0..1])
   * onto screen pixels.
   */
  _project(W, H, laneCenterOffset, z, camOffset) {
    const horizonY = H * 0.315;
    const roadH = H - horizonY;
    // Perspective depth curve
    const p = Math.pow(Math.max(0.005, z), 1.65);
    const y = horizonY + p * roadH;

    // Road half-width expands from narrow horizon to > screen half-width at bottom
    const halfW = W * (0.018 + p * 0.62);
    const vanishX = W * 0.5 - camOffset * W * 0.045;
    // Slight curve shift near foreground with camera lateral offset
    const centerX = vanishX - camOffset * p * W * 0.24;
    const x = centerX + (laneCenterOffset / 1.5) * halfW;
    return { x, y, halfW, p, scale: 0.06 + p * 0.94 };
  }

  draw(ctx, W, H, playerLanePos, speedKmh, rollRad) {
    // camOffset in [-1..1] corresponding to lane 0..2 centered around lane 1
    const camOffset = playerLanePos - 1.0;
    const horizonY = H * 0.315;

    ctx.save();
    // Subtle horizon roll when motorcycle leans
    ctx.translate(W * 0.5, H * 0.55);
    ctx.rotate(-rollRad * 0.25);
    ctx.translate(-W * 0.5, -H * 0.55);

    // 1. Overcast Indian daylight sky
    const skyGrad = ctx.createLinearGradient(0, 0, 0, horizonY + 12);
    skyGrad.addColorStop(0, "#cfd7de");
    skyGrad.addColorStop(0.65, "#e2e7eb");
    skyGrad.addColorStop(1, "#eef1f3");
    ctx.fillStyle = skyGrad;
    ctx.fillRect(-40, -40, W + 80, horizonY + 60);

    // Soft clouds
    ctx.fillStyle = "rgba(255,255,255,0.28)";
    for (let c = 0; c < 4; c++) {
      const cx = W * (0.15 + c * 0.24) - camOffset * 18;
      const cy = horizonY * (0.35 + (c % 2) * 0.2);
      ctx.beginPath();
      ctx.ellipse(cx, cy, W * 0.14, horizonY * 0.16, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // 2. Ground / Sidewalk base below horizon
    const groundGrad = ctx.createLinearGradient(0, horizonY, 0, H);
    groundGrad.addColorStop(0, "#8f9490");
    groundGrad.addColorStop(1, "#55585c");
    ctx.fillStyle = groundGrad;
    ctx.fillRect(-40, horizonY, W + 80, H - horizonY + 40);

    // 3. Roadside buildings, trees & poles (sorted back-to-front)
    const sortedProps = [...this.roadsideProps].sort((a, b) => a.z - b.z);
    for (const prop of sortedProps) {
      if (prop.z < 0.02 || prop.z > 1.02) continue;
      const leftEdge = this._project(W, H, -1.55, prop.z, camOffset);
      const rightEdge = this._project(W, H, 1.55, prop.z, camOffset);
      this._drawBuilding(ctx, leftEdge.x, leftEdge.y, leftEdge.scale, -1, prop.leftBld, prop.leftFloors);
      this._drawBuilding(ctx, rightEdge.x, rightEdge.y, rightEdge.scale, 1, prop.rightBld, prop.rightFloors);
      this._drawTree(ctx, leftEdge.x - 10 * leftEdge.scale, leftEdge.y, leftEdge.scale);
      this._drawTree(ctx, rightEdge.x + 12 * rightEdge.scale, rightEdge.y, rightEdge.scale);
      if (prop.hasPole) {
        this._drawStreetPole(ctx, rightEdge.x + 4 * rightEdge.scale, rightEdge.y, rightEdge.scale);
      }
    }

    // 4. Asphalt 3-lane road polygon
    const topL = this._project(W, H, -1.52, 0.005, camOffset);
    const topR = this._project(W, H, 1.52, 0.005, camOffset);
    const botL = this._project(W, H, -1.52, 1.02, camOffset);
    const botR = this._project(W, H, 1.52, 1.02, camOffset);

    const roadGrad = ctx.createLinearGradient(0, horizonY, 0, H);
    roadGrad.addColorStop(0, "#75797f");
    roadGrad.addColorStop(0.35, "#5c6066");
    roadGrad.addColorStop(1, "#494d52");
    ctx.fillStyle = roadGrad;
    ctx.beginPath();
    ctx.moveTo(topL.x, topL.y);
    ctx.lineTo(topR.x, topR.y);
    ctx.lineTo(botR.x, botR.y);
    ctx.lineTo(botL.x, botL.y);
    ctx.closePath();
    ctx.fill();

    // 5. Black-and-white Indian curbs along both road shoulders
    this._drawCurbs(ctx, W, H, camOffset);

    // 6. Scrolling 3-lane white dashed dividers & edge lines
    this._drawLaneMarkings(ctx, W, H, camOffset);

    // 7. Traffic vehicles sorted back-to-front
    const sortedVehicles = [...this.vehicles].sort((a, b) => a.z - b.z);
    for (const v of sortedVehicles) {
      const laneOffset = v.lane - 1.0; // lane 0->-1, 1->0, 2->+1
      const pos = this._project(W, H, laneOffset, v.z, camOffset);
      this._drawVehicle(ctx, v, pos.x, pos.y, pos.scale, speedKmh);
    }

    // Subtle speed wind streaks at high speed
    if (speedKmh > 48) {
      const alpha = Math.min(0.24, ((speedKmh - 48) / 40) * 0.24);
      ctx.strokeStyle = `rgba(255,255,255,${alpha.toFixed(3)})`;
      ctx.lineWidth = 1.5;
      for (let i = 0; i < 8; i++) {
        const ang = (i / 8) * Math.PI * 2 + (this.scrollDist * 0.15) % 1;
        const r1 = W * 0.34;
        const r2 = W * 0.48;
        ctx.beginPath();
        ctx.moveTo(W * 0.5 + Math.cos(ang) * r1, horizonY + Math.sin(ang) * r1 * 0.45);
        ctx.lineTo(W * 0.5 + Math.cos(ang) * r2, horizonY + Math.sin(ang) * r2 * 0.45);
        ctx.stroke();
      }
    }

    // Collision alert vignette
    if (this.collisionFlash > 0.02) {
      ctx.fillStyle = `rgba(255, 50, 40, ${(this.collisionFlash * 0.28).toFixed(3)})`;
      ctx.fillRect(-40, -40, W + 80, H + 80);
    }

    ctx.restore();
  }

  _drawCurbs(ctx, W, H, camOffset) {
    const segments = 26;
    const phase = (this.scrollDist * 0.06) % 1;
    for (let i = 0; i < segments; i++) {
      const z1 = Math.max(0.01, (i + phase) / segments);
      const z2 = Math.min(1.03, (i + 0.85 + phase) / segments);
      if (z2 <= z1) continue;
      const isWhite = (i + Math.floor(this.scrollDist * 0.06)) % 2 === 0;
      const col = isWhite ? "#eceff2" : "#22252a";

      for (const side of [-1, 1]) {
        const a1 = this._project(W, H, side * 1.52, z1, camOffset);
        const a2 = this._project(W, H, side * 1.52, z2, camOffset);
        const b1 = this._project(W, H, side * 1.62, z1, camOffset);
        const b2 = this._project(W, H, side * 1.62, z2, camOffset);
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.moveTo(a1.x, a1.y);
        ctx.lineTo(b1.x, b1.y);
        ctx.lineTo(b2.x, b2.y);
        ctx.lineTo(a2.x, a2.y);
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  _drawLaneMarkings(ctx, W, H, camOffset) {
    // Solid outer edge markings
    for (const edgeOffset of [-1.42, 1.42]) {
      const pTop = this._project(W, H, edgeOffset, 0.01, camOffset);
      const pBot = this._project(W, H, edgeOffset, 1.02, camOffset);
      ctx.strokeStyle = "rgba(245, 247, 250, 0.72)";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(pTop.x, pTop.y);
      ctx.lineTo(pBot.x, pBot.y);
      ctx.stroke();
    }

    // Dashed lane dividers between Left-Center (-0.5) and Center-Right (+0.5)
    const dashes = 13;
    const cycle = (this.scrollDist * 0.085) % 1;
    for (const divider of [-0.5, 0.5]) {
      for (let i = -1; i < dashes; i++) {
        const zStart = (i + cycle) / dashes;
        const zEnd = zStart + 0.42 / dashes;
        if (zEnd < 0.01 || zStart > 1.02) continue;
        const zs = Math.max(0.01, zStart);
        const ze = Math.min(1.02, zEnd);
        const p1 = this._project(W, H, divider, zs, camOffset);
        const p2 = this._project(W, H, divider, ze, camOffset);
        const w1 = Math.max(1.5, p1.scale * 5);
        const w2 = Math.max(2.5, p2.scale * 11);

        ctx.fillStyle = "rgba(248, 250, 252, 0.90)";
        ctx.beginPath();
        ctx.moveTo(p1.x - w1 * 0.5, p1.y);
        ctx.lineTo(p1.x + w1 * 0.5, p1.y);
        ctx.lineTo(p2.x + w2 * 0.5, p2.y);
        ctx.lineTo(p2.x - w2 * 0.5, p2.y);
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  _drawBuilding(ctx, edgeX, groundY, scale, dir, palette, floors) {
    const bw = 210 * scale;
    const bh = (68 * floors) * scale;
    const bx = dir < 0 ? edgeX - bw - 48 * scale : edgeX + 48 * scale;
    const by = groundY - bh + 6 * scale;

    // Main concrete facade
    ctx.fillStyle = palette.wall;
    ctx.fillRect(bx, by, bw, bh);

    // Side shaded depth
    ctx.fillStyle = palette.trim;
    const sideW = 24 * scale;
    ctx.fillRect(dir < 0 ? bx + bw - sideW : bx, by, sideW, bh);

    // Rooftop parapet
    ctx.fillRect(bx - 2 * scale, by - 8 * scale, bw + 4 * scale, 9 * scale);

    // Windows / Balconies grid
    ctx.fillStyle = palette.win;
    const cols = 3;
    const padX = bw * 0.14;
    const winW = (bw - padX * 2) / cols * 0.65;
    const winH = (bh / floors) * 0.42;
    for (let f = 0; f < floors; f++) {
      for (let c = 0; c < cols; c++) {
        const wx = bx + padX + c * ((bw - padX * 2) / cols);
        const wy = by + (f + 0.25) * (bh / floors);
        ctx.fillRect(wx, wy, winW, winH);
      }
    }

    // Ground floor Indian shop shutter / awning strip
    ctx.fillStyle = dir < 0 ? "#8c3b32" : "#2f5d80";
    ctx.fillRect(bx, groundY - 28 * scale, bw, 10 * scale);
  }

  _drawTree(ctx, x, groundY, scale) {
    const trunkH = 54 * scale;
    const trunkW = Math.max(2, 10 * scale);
    ctx.fillStyle = "#4b3b2d";
    ctx.fillRect(x - trunkW * 0.5, groundY - trunkH, trunkW, trunkH);

    // Lush layered foliage canopy
    const r = 46 * scale;
    ctx.fillStyle = "#1f4f24";
    ctx.beginPath();
    ctx.arc(x, groundY - trunkH - r * 0.45, r * 0.92, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#2d6e31";
    ctx.beginPath();
    ctx.arc(x - r * 0.35, groundY - trunkH - r * 0.65, r * 0.72, 0, Math.PI * 2);
    ctx.arc(x + r * 0.35, groundY - trunkH - r * 0.58, r * 0.72, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#418c40";
    ctx.beginPath();
    ctx.arc(x, groundY - trunkH - r * 0.9, r * 0.65, 0, Math.PI * 2);
    ctx.fill();
  }

  _drawStreetPole(ctx, x, groundY, scale) {
    const poleH = 220 * scale;
    ctx.strokeStyle = "#8a929a";
    ctx.lineWidth = Math.max(1.2, 4 * scale);
    ctx.beginPath();
    ctx.moveTo(x, groundY);
    ctx.lineTo(x, groundY - poleH);
    // Curved arm toward road
    ctx.lineTo(x - 58 * scale, groundY - poleH - 12 * scale);
    ctx.stroke();

    // Lamp head
    ctx.fillStyle = "#cfd6dc";
    ctx.fillRect(x - 68 * scale, groundY - poleH - 14 * scale, 16 * scale, 5 * scale);
  }

  _drawVehicle(ctx, v, cx, baseY, scale) {
    ctx.save();
    ctx.translate(cx, baseY);
    ctx.scale(scale, scale);

    // Contact shadow
    ctx.fillStyle = "rgba(0,0,0,0.34)";
    ctx.beginPath();
    ctx.ellipse(0, 2, 74, 12, 0, 0, Math.PI * 2);
    ctx.fill();

    if (v.type === "autorickshaw") {
      this._drawAutoRickshaw(ctx);
    } else if (v.type === "suv") {
      this._drawInnovaSUV(ctx, v.color);
    } else {
      this._drawHatchback(ctx, v.color);
    }

    ctx.restore();
  }

  _drawAutoRickshaw(ctx) {
    // Rear tires
    ctx.fillStyle = "#191b1e";
    ctx.fillRect(-54, -22, 16, 24);
    ctx.fillRect(38, -22, 16, 24);

    // Yellow lower body
    ctx.fillStyle = "#efb810";
    ctx.beginPath();
    ctx.roundRect(-52, -76, 104, 62, 10);
    ctx.fill();

    // Black canvas roof & rear curtain
    ctx.fillStyle = "#1e2024";
    ctx.beginPath();
    ctx.roundRect(-48, -142, 96, 74, [24, 24, 6, 6]);
    ctx.fill();

    // Rear plastic window
    ctx.fillStyle = "#48515c";
    ctx.beginPath();
    ctx.roundRect(-28, -118, 56, 24, 5);
    ctx.fill();

    // Yellow trim strip & number plate
    ctx.fillStyle = "#d99d00";
    ctx.fillRect(-46, -42, 92, 6);
    ctx.fillStyle = "#f2c94c";
    ctx.fillRect(-16, -34, 32, 12);

    // Twin round red tail lights + amber indicators
    ctx.fillStyle = "#e62e2d";
    ctx.beginPath();
    ctx.arc(-38, -55, 6, 0, Math.PI * 2);
    ctx.arc(38, -55, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ff9900";
    ctx.beginPath();
    ctx.arc(-44, -67, 4, 0, Math.PI * 2);
    ctx.arc(44, -67, 4, 0, Math.PI * 2);
    ctx.fill();

    // Black rear bumper bar
    ctx.fillStyle = "#141619";
    ctx.fillRect(-56, -20, 112, 8);
  }

  _drawHatchback(ctx, color) {
    // Tires
    ctx.fillStyle = "#181a1d";
    ctx.fillRect(-74, -28, 22, 30);
    ctx.fillRect(52, -28, 22, 30);

    // Main sculpted hatchback body
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(-76, -16);
    ctx.lineTo(-78, -62);
    ctx.lineTo(-56, -112);
    ctx.lineTo(56, -112);
    ctx.lineTo(78, -62);
    ctx.lineTo(76, -16);
    ctx.closePath();
    ctx.fill();

    // Rear sloped windshield
    ctx.fillStyle = "#2b333b";
    ctx.beginPath();
    ctx.moveTo(-50, -104);
    ctx.lineTo(50, -104);
    ctx.lineTo(62, -68);
    ctx.lineTo(-62, -68);
    ctx.closePath();
    ctx.fill();

    // Tail lamps
    ctx.fillStyle = "#d92b2b";
    ctx.beginPath();
    ctx.roundRect(-72, -66, 22, 16, 4);
    ctx.roundRect(50, -66, 22, 16, 4);
    ctx.fill();

    // License plate & lower bumper
    ctx.fillStyle = "#eaecef";
    ctx.fillRect(-18, -42, 36, 12);
    ctx.fillStyle = "#363a40";
    ctx.fillRect(-72, -20, 144, 8);
  }

  _drawInnovaSUV(ctx, color) {
    // Tires
    ctx.fillStyle = "#16181b";
    ctx.fillRect(-84, -32, 24, 34);
    ctx.fillRect(60, -32, 24, 34);

    // Tall MPV/SUV body
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(-84, -142, 168, 124, [18, 18, 8, 8]);
    ctx.fill();

    // Rear glass
    ctx.fillStyle = "#2c343d";
    ctx.beginPath();
    ctx.roundRect(-66, -130, 132, 48, 8);
    ctx.fill();

    // Chrome tailgate bar
    ctx.fillStyle = "#f2f5f7";
    ctx.fillRect(-48, -78, 96, 5);

    // Vertical tail lights
    ctx.fillStyle = "#d92626";
    ctx.beginPath();
    ctx.roundRect(-80, -84, 18, 34, 4);
    ctx.roundRect(62, -84, 18, 34, 4);
    ctx.fill();

    // License plate & bumper
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-24, -62, 48, 14);
    ctx.fillStyle = "#9da4ac";
    ctx.fillRect(-84, -30, 168, 14);
  }
}
